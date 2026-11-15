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
            static = envs.FASTVIDEO_H3_FAST_VAE_FP8_STATIC.get()
            replaced = convert_linears_to_fp8(self.model, skip_names=("x_embedder", "proj_out"),
                                              static_activation=static)
            logger.info("MiniMax-H3 fast decoder: %d linears in per-tensor FP8 (static_activation=%s)",
                        replaced, static)
        if envs.FASTVIDEO_H3_FAST_VAE_COMPILE.get():
            # Eager peaks at ~95 GiB at 2K on GB10; compiled fuses the casts, norms and window partition glue and
            # runs at ~36 GiB, ~37 s instead of ~60 s. The first call of a new latent shape pays the compile.
            self._forward = torch.compile(self.model)

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, H, W)`` normalized -> ``(B, T, C, H, W)``; the decoder denormalizes internally."""
        return latents.transpose(1, 2).contiguous()

    def denormalize_latents_for_export(self, latents: torch.Tensor) -> torch.Tensor:
        """The released VAE's latent-export convention: denormalized, channel-first ``(B, C, T, H, W)``.

        ``denormalize_latents`` prepares this decoder's time-first input instead, so the export path keeps the
        released convention and an opt-in decoder does not change what ``output_type="latent"`` returns.
        """
        mean = self.model.latents_mean.view(1, -1, 1, 1, 1)
        std = self.model.latents_std.view(1, -1, 1, 1, 1)
        return latents * std + mean

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
        # In-place denormalization: the naive ``(pixels * std + mean)`` materializes two more
        # 8.5 GiB fp32 tensors at 2K, which is the whole stage's memory peak on a Spark and is
        # what made a resident DiT + decode not fit. Same ops, same order, same values.
        pixels.mul_(self.pixel_std).add_(self.pixel_mean).clamp_(0, 1)
        return pixels

    # ------------------------------------------------------------------ sequence-parallel slice decode
    # The decoder's temporal attention is windowed over the native 5-latent blocks with odd layers shifted by 2, and
    # each window's global tokens belong to the native block it overlaps, so a block-aligned slice with a one-block
    # halo on each side reproduces the owned blocks' pixels. It is not bit-identical to the full decode (the bf16
    # SDPA accumulation order depends on the tensor shape) but measured 41.6 dB PSNR at 2K, inside the pipeline's own
    # cross-process spread. On the output rank the full-buffer copy path is unchanged; the other ranks ship only
    # their owned frames over the sequence-parallel group.
    supports_sequence_parallel_decode = True

    @torch.no_grad()
    def decode_range(self, z: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        """Decode the block-aligned latent slice ``z[:, start:stop]`` -> ``(B, 3, F, 16H, 16W)`` pixels in [0, 1]."""
        with torch.autocast("cuda", dtype=self.decode_autocast_dtype):
            pixels = self._forward(z[:, start:stop].contiguous().float()).float()
        pixels.mul_(self.pixel_std).add_(self.pixel_mean).clamp_(0, 1)
        return pixels

    @torch.no_grad()
    def decode_to_pixels_parallel(self,
                                  z: torch.Tensor,
                                  output: torch.Tensor | None,
                                  sp_group: Any,
                                  strategy: str | None = None) -> torch.Tensor | None:
        """Decode the native-block slices across ``sp_group`` and assemble on the ``output`` rank.

        Every rank decodes the blocks it owns plus a one-block halo on each side and keeps only its owned frames; the
        frames are padded to the largest owned segment and all-gathered, and the assembling rank writes each rank's
        segment into its full CPU ``output`` buffer. ``output`` is ``None`` on the other ranks. The collective is
        called by every rank with identical shapes.
        """
        world = sp_group.world_size
        rank = sp_group.rank_in_group
        t_latent = z.shape[1]
        num_blocks = (t_latent + BLOCK - 1) // BLOCK
        total_frames = t_latent * PATCH_T - FRAMES_TO_TRIM * num_blocks
        block_frames = BLOCK * PATCH_T - FRAMES_TO_TRIM
        # Frame ownership is deterministic from the geometry, so every rank builds the same plan.
        spans = []
        for r in range(world):
            own_lo = r * num_blocks // world
            own_hi = (r + 1) * num_blocks // world
            frames = min(own_hi * block_frames, total_frames) - own_lo * block_frames
            spans.append((own_lo, own_hi, frames))
        max_frames = max(frames for _, _, frames in spans)

        own_lo, own_hi, own_frames = spans[rank]
        slice_lo = max(0, own_lo - 1)
        slice_hi = min(num_blocks, own_hi + 1)
        pixels = self.decode_range(z, slice_lo * BLOCK, min(slice_hi * BLOCK, t_latent))
        frame_lo = (own_lo - slice_lo) * block_frames
        segment = pixels[:, :, frame_lo:frame_lo + own_frames]
        if segment.shape[2] < max_frames:
            segment = torch.nn.functional.pad(segment, (0, 0, 0, 0, 0, max_frames - segment.shape[2]))
        # Both ranks must post the same dtype/count to the collective. The assembling rank's output buffer is uint8
        # under FASTVIDEO_H3_VAE_UINT8_PIXELS, but the peer has output=None, so gate the quantize on the env (known
        # identically on every rank) rather than on the local buffer: shipping fp32 from one rank and uint8 from the
        # other makes the all_gather hang.
        ship_uint8 = bool(envs.FASTVIDEO_H3_VAE_UINT8_PIXELS.get()) and (output is None or output.dtype == torch.uint8)
        if ship_uint8:
            segment = segment.mul(255.0).clamp_(0, 255).to(torch.uint8)
        segment = segment.contiguous()
        gathered = sp_group.all_gather(segment, dim=2)
        if output is None:
            return None
        # Contiguous block ownership makes the concatenated owned frames the prefix of the gathered tensor: every
        # rank but the last owns a whole number of 17-frame blocks, and the last one's padding trails its valid
        # frames. Copy per (batch, channel) so both sides are contiguous -- a single strided view of ``gathered``
        # would make the device-to-host copy re-fault pages again.
        total_owned = sum(frames for _, _, frames in spans)
        for b in range(output.shape[0]):
            for c in range(output.shape[1]):
                part = gathered[b, c, :total_owned]
                if part.dtype != output.dtype:
                    part = part.to(output.dtype)
                output[b, c].copy_(part)
        return output

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


_DECODERS: dict[tuple[Any, ...], FastH3VaeDecoder] = {}
# One live profile plus the previous one. The key covers the construction settings, so without a bound a
# profile switch would keep a GPU-resident decoder for every combination of FP8, compile and mask settings.
_MAX_DECODERS = 2


def fast_vae_decoder(fastvideo_args: Any, device: torch.device) -> FastH3VaeDecoder | None:
    """The opt-in fast decoder, built once per (weights, base model, construction settings); ``None`` when off."""
    weights = envs.FASTVIDEO_H3_FAST_VAE_DECODER.get()
    if not weights:
        return None
    base_model = envs.FASTVIDEO_H3_FAST_VAE_BASE_MODEL.get() or str(Path(fastvideo_args.model_path) / "base_model")
    num_layers = int(envs.FASTVIDEO_H3_FAST_VAE_LAYERS.get())
    key = (
        str(weights),
        str(base_model),
        num_layers,
        bool(envs.FASTVIDEO_H3_FAST_VAE_FP8.get()),
        bool(envs.FASTVIDEO_H3_FAST_VAE_FP8_STATIC.get()),
        bool(envs.FASTVIDEO_H3_FAST_VAE_COMPILE.get()),
        bool(envs.FASTVIDEO_H3_FAST_VAE_ATTN_NOMASK.get()),
    )
    decoder = _DECODERS.get(key)
    if decoder is None:
        logger.info("MiniMax-H3 video decode: fast %d-layer decoder %s (latent statistics from %s)", num_layers, weights,
                    base_model)
        decoder = FastH3VaeDecoder(weights, base_model, device=device)
        _DECODERS[key] = decoder
        while len(_DECODERS) > _MAX_DECODERS:
            _DECODERS.pop(next(iter(_DECODERS)))
    return decoder


__all__ = ["FastH3VaeDecoder", "fast_vae_decoder"]
