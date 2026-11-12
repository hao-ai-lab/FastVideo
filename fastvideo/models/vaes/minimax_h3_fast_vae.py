# SPDX-License-Identifier: Apache-2.0
"""NVlabs' fast 24-layer MiniMax-H3 video decoder (StreamRender-H3) behind the released-VAE decode surface.

Opt-in through ``FASTVIDEO_H3_FAST_VAE_DECODER=<fast_h3_vae_noncausal_24l.safetensors>``. The decoder is the plain
PyTorch one in ``minimax_h3_fast_vae_decoder`` (SDPA over 16x16-token windows), so it runs on any CUDA GPU, including
GB10 (sm_121) where NVlabs' FlashAttention-4 deployment engine does not. It replaces only the video decoder and is
lossy against the released 36-layer decoder (33.9 dB on a 2K Open-H3-Regenerate-2K clip), so it is approximate tier.

The adapter exposes the part of ``AutoencoderKLMiniMaxH3`` the decoding stage drives: ``denormalize_latents``,
``decoded_pixel_shape``, ``decode_autocast_dtype`` and ``decode_to_pixels``. ``denormalize_latents`` only reorders
``(B, C, T, H, W)`` to ``(B, T, C, H, W)``: the decoder applies ``z * latents_std + latents_mean`` itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file

import fastvideo.envs as envs
from fastvideo.logger import init_logger
from fastvideo.models.vaes.minimax_h3_fast_vae_decoder import KEEP, NonCausalViTDecoder

logger = init_logger(__name__)

# The released VAE's ImageNet pixel statistics: the decoder outputs ImageNet-normalized pixels.
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
# Native temporal geometry: 5 latent frames per block, 4 frames per latent frame, the first 3 frames of every block
# dropped (17 frames per block); 16x spatial compression.
BLOCK = 5
PATCH_T = 4
FRAMES_TO_TRIM = 3
SPATIAL_COMPRESSION = 16
NUM_LAYERS = 24
# The released fast checkpoint (fast_h3_vae_noncausal_24l.safetensors) stores exactly NVlabs' KEEP_24 layers,
# reindexed 0..23. A smaller KEEP[n] subset is a subset of those 24, so it is expressed in checkpoint coordinates and
# the block weights are re-stacked (see ``FastH3VaeDecoder.__init__``).
CHECKPOINT_LAYERS = KEEP[NUM_LAYERS]
LATENT_CHANNELS = 24


class FastH3VaeDecoder(nn.Module):
    """The fast 24-layer decoder with the released VAE's decode surface."""

    decode_autocast_dtype = torch.bfloat16

    def __init__(self, weights_path: str | Path, base_model_path: str | Path, device: torch.device | str = "cuda"):
        super().__init__()
        config_path = Path(base_model_path) / "vae" / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"The fast H3 VAE decoder needs the base model's {config_path} for its latent "
                                    "statistics; set FASTVIDEO_H3_FAST_VAE_BASE_MODEL.")
        config = json.loads(config_path.read_text())
        num_layers = int(envs.FASTVIDEO_H3_FAST_VAE_LAYERS.get())
        if num_layers not in KEEP:
            raise ValueError(f"FASTVIDEO_H3_FAST_VAE_LAYERS must be one of {sorted(KEEP)}, got {num_layers}.")
        keep_layers = KEEP[num_layers]
        missing = [idx for idx in keep_layers if idx not in CHECKPOINT_LAYERS]
        if missing:
            raise ValueError(f"The fast decoder checkpoint does not contain original layers {missing}.")
        model = NonCausalViTDecoder(config["latents_mean"],
                                    config["latents_std"],
                                    keep_layers,
                                    time_norm=BLOCK,
                                    heads=config["decoder_num_attention_heads"],
                                    dim_head=config["decoder_attention_head_dim"],
                                    rope_theta=config["decoder_rope_theta"],
                                    rope_dim_ratio=config["decoder_rope_dim_ratio"],
                                    eps=config["decoder_norm_eps"],
                                    latent_channels=config["latent_channels"])
        state = load_file(str(weights_path), device="cpu")
        if num_layers != NUM_LAYERS:
            # Reindex the checkpoint's KEEP_24 blocks into the selected subset, keeping the new order; drop the
            # checkpoint blocks the selected subset does not use.
            positions = [CHECKPOINT_LAYERS.index(idx) for idx in keep_layers]
            remap = {pos: new for new, pos in enumerate(positions)}
            remapped = {}
            for key, value in state.items():
                if key.startswith("blocks."):
                    _, index, rest = key.split(".", 2)
                    new_index = remap.get(int(index))
                    if new_index is None:
                        continue
                    remapped[f"blocks.{new_index}.{rest}"] = value
                else:
                    remapped[key] = value
            state = remapped
            logger.info("MiniMax-H3 fast decoder: %d of %d checkpoint layers (original indices %s)",
                        num_layers, NUM_LAYERS, keep_layers)
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval().requires_grad_(False)
        self.latent_channels = int(config["latent_channels"])
        self.spatial_compression_ratio = SPATIAL_COMPRESSION
        self.register_buffer("pixel_mean",
                             torch.tensor(PIXEL_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1, 1),
                             persistent=False)
        self.register_buffer("pixel_std",
                             torch.tensor(PIXEL_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1, 1),
                             persistent=False)
        self._forward = self.model
        if envs.FASTVIDEO_H3_FAST_VAE_FP8.get():
            # Approximate tier: the block attention/FFN linears run as per-tensor FP8 E4M3. The conv/embedder and
            # the final proj_out stay in their original precision.
            from fastvideo.models.vaes.fast_vae_fp8 import convert_linears_to_fp8
            replaced = convert_linears_to_fp8(self.model, skip_names=("x_embedder", "proj_out"))
            logger.info("MiniMax-H3 fast decoder: %d linears in per-tensor FP8", replaced)
        if envs.FASTVIDEO_H3_FAST_VAE_COMPILE.get():
            # Eager peaks at ~95 GiB at 2K on GB10; compiled fuses the casts, norms and window partition glue and
            # runs at ~36 GiB, ~37 s instead of ~60 s. The first call of a new latent shape pays the compile.
            self._forward = torch.compile(self.model)

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, H, W)`` normalized -> ``(B, T, C, H, W)``; the decoder denormalizes internally."""
        return latents.transpose(1, 2).contiguous()

    def decoded_pixel_shape(self, latent_shape: torch.Size | tuple[int, ...]) -> tuple[int, int, int, int, int]:
        if len(latent_shape) != 5:
            raise ValueError(f"Fast H3 VAE latents must be five-dimensional, got {tuple(latent_shape)}.")
        batch_size, latent_num_frames, channels, latent_height, latent_width = map(int, latent_shape)
        if channels != self.latent_channels:
            raise ValueError(f"Fast H3 VAE latents must have {self.latent_channels} channels, got {channels}.")
        num_blocks = (latent_num_frames + BLOCK - 1) // BLOCK
        decoded_num_frames = latent_num_frames * PATCH_T - FRAMES_TO_TRIM * num_blocks
        return (batch_size, 3, decoded_num_frames, latent_height * SPATIAL_COMPRESSION,
                latent_width * SPATIAL_COMPRESSION)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, T, C, H, W)`` normalized latents -> ``(B, 3, F, 16H, 16W)`` pixels in [0, 1]."""
        with torch.autocast("cuda", dtype=self.decode_autocast_dtype):
            pixels = self._forward(z.float()).float()
        return (pixels * self.pixel_std + self.pixel_mean).clamp_(0, 1)

    def decode_to_pixels(self, z: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Decode into a caller-owned CPU buffer, like ``AutoencoderKLMiniMaxH3.decode_to_pixels``."""
        expected_shape = self.decoded_pixel_shape(z.shape)
        if (output.device.type != "cpu" or output.dtype not in (torch.float32, torch.uint8)
                or tuple(output.shape) != expected_shape):
            raise ValueError("`output` must be a CPU float32 or uint8 tensor with shape "
                             f"{expected_shape}, got device={output.device}, dtype={output.dtype}, "
                             f"shape={tuple(output.shape)}.")
        pixels = self.decode(z)
        if output.dtype == torch.uint8:
            pixels = pixels.mul_(255).clamp_(0, 255).to(torch.uint8)
        output.copy_(pixels)
        return output


_DECODERS: dict[tuple[str, str], FastH3VaeDecoder] = {}


def fast_vae_decoder(fastvideo_args: Any, device: torch.device) -> FastH3VaeDecoder | None:
    """The opt-in fast decoder, built once per (weights, base model) and kept resident; ``None`` when off."""
    weights = envs.FASTVIDEO_H3_FAST_VAE_DECODER.get()
    if not weights:
        return None
    base_model = envs.FASTVIDEO_H3_FAST_VAE_BASE_MODEL.get() or str(Path(fastvideo_args.model_path) / "base_model")
    num_layers = int(envs.FASTVIDEO_H3_FAST_VAE_LAYERS.get())
    key = (str(weights), str(base_model), num_layers)
    decoder = _DECODERS.get(key)
    if decoder is None:
        logger.info("MiniMax-H3 video decode: fast %d-layer decoder %s (latent statistics from %s)", num_layers, weights,
                    base_model)
        decoder = FastH3VaeDecoder(weights, base_model, device=device)
        _DECODERS[key] = decoder
    return decoder


__all__ = ["FastH3VaeDecoder", "fast_vae_decoder"]
