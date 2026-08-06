# SPDX-License-Identifier: Apache-2.0
"""MAGI-2 Stable Audio Open decoder loading and source-key conversion."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from fastvideo.models.vaes.oobleck import OobleckDecoder

_SOURCE_PREFIX = "pretransform.model."


class Magi2AudioVAE(nn.Module):
    """Decode MAGI-2 audio latents with the Stable Audio Oobleck decoder."""

    def __init__(
        self,
        decoder: OobleckDecoder,
        sampling_rate: int,
        downsampling_ratio: int,
    ) -> None:
        """Store the decoder and the published audio sampling geometry."""
        super().__init__()
        self.decoder = decoder
        self.sampling_rate = sampling_rate
        self.hop_length = downsampling_ratio

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode channel-major audio latents into channel-major waveforms."""
        if latent.ndim == 2:
            latent = latent.unsqueeze(0)
        return self.decoder(latent)


def _source_decoder_to_fastvideo_key(
    source_name: str,
    decoder_block_count: int,
) -> str:
    """Map one Stable Audio sequential decoder key to an Oobleck decoder key."""
    parts = source_name.split(".")
    if parts[:2] != ["decoder", "layers"] or len(parts) < 4:
        raise ValueError(f"Unsupported Stable Audio decoder key: {source_name}")

    top_level_index = int(parts[2])
    parameter_parts = parts[3:]
    if top_level_index == 0:
        return ".".join(["decoder", "conv1", *parameter_parts])
    if top_level_index == decoder_block_count + 1:
        return ".".join(["decoder", "snake1", *parameter_parts])
    if top_level_index == decoder_block_count + 2:
        return ".".join(["decoder", "conv2", *parameter_parts])
    if not 1 <= top_level_index <= decoder_block_count:
        raise ValueError(f"Unsupported Stable Audio decoder key: {source_name}")

    if parameter_parts[:1] != ["layers"] or len(parameter_parts) < 3:
        raise ValueError(f"Unsupported Stable Audio decoder block key: {source_name}")
    block_index = top_level_index - 1
    block_layer_index = int(parameter_parts[1])
    block_parameter_parts = parameter_parts[2:]
    if block_layer_index == 0:
        target_parts = ["decoder", "block", str(block_index), "snake1"]
    elif block_layer_index == 1:
        target_parts = ["decoder", "block", str(block_index), "conv_t1"]
    elif 2 <= block_layer_index <= 4:
        if block_parameter_parts[:1] != ["layers"] or len(block_parameter_parts) < 3:
            raise ValueError(f"Unsupported Stable Audio residual key: {source_name}")
        residual_layer_index = int(block_parameter_parts[1])
        residual_names = ("snake1", "conv1", "snake2", "conv2")
        if residual_layer_index >= len(residual_names):
            raise ValueError(f"Unsupported Stable Audio residual key: {source_name}")
        target_parts = [
            "decoder",
            "block",
            str(block_index),
            f"res_unit{block_layer_index - 1}",
            residual_names[residual_layer_index],
        ]
        block_parameter_parts = block_parameter_parts[2:]
    else:
        raise ValueError(f"Unsupported Stable Audio decoder block key: {source_name}")
    return ".".join([*target_parts, *block_parameter_parts])


def _load_decoder_state(
    decoder_model: Magi2AudioVAE,
    checkpoint_path: Path,
    decoder_block_count: int,
) -> None:
    """Read only decoder tensors and load the mapped state strictly."""
    target_state = decoder_model.state_dict()
    mapped_state: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        for checkpoint_name in list(checkpoint.keys()):
            if not checkpoint_name.startswith(f"{_SOURCE_PREFIX}decoder."):
                continue
            source_name = checkpoint_name.removeprefix(_SOURCE_PREFIX)
            target_name = _source_decoder_to_fastvideo_key(
                source_name,
                decoder_block_count,
            )
            if target_name not in target_state:
                raise KeyError(
                    f"Stable Audio key {checkpoint_name} maps to unknown key {target_name}"
                )
            source_tensor = checkpoint.get_tensor(checkpoint_name)
            target_tensor = target_state[target_name]
            if source_tensor.numel() != target_tensor.numel():
                raise ValueError(
                    f"Stable Audio tensor size differs for {checkpoint_name}: "
                    f"source={tuple(source_tensor.shape)}, target={tuple(target_tensor.shape)}"
                )
            mapped_state[target_name] = source_tensor.reshape(target_tensor.shape)
    incompatible = decoder_model.load_state_dict(mapped_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Stable Audio strict load failed: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def load_magi2_audio_vae(
    component_dir: str | Path,
    device: torch.device | str,
) -> Magi2AudioVAE:
    """Build and strictly load MAGI-2's published FP32 audio decoder."""
    component_path = Path(component_dir)
    config_path = component_path / "model_config.json"
    checkpoint_path = component_path / "model.safetensors"
    with config_path.open(encoding="utf-8") as config_file:
        full_config = json.load(config_file)
    sampling_rate = int(full_config["sample_rate"])
    vae_config = full_config["model"]["pretransform"]["config"]
    decoder_spec = vae_config["decoder"]
    decoder_config = decoder_spec["config"]
    if decoder_spec["type"] != "oobleck" or not decoder_config["use_snake"]:
        raise ValueError("MAGI-2 audio decoding requires the published Oobleck Snake decoder")
    if decoder_config.get("use_nearest_upsample", False):
        raise ValueError("MAGI-2 audio decoding requires transposed-convolution upsampling")
    if decoder_config.get("final_tanh", True):
        raise ValueError("MAGI-2 audio decoding requires final_tanh=false")

    strides = [int(stride) for stride in decoder_config["strides"]]
    downsampling_ratio = int(vae_config["downsampling_ratio"])
    if math.prod(strides) != downsampling_ratio:
        raise ValueError("Stable Audio strides do not match the declared downsampling ratio")
    decoder = OobleckDecoder(
        channels=int(decoder_config["channels"]),
        input_channels=int(decoder_config["latent_dim"]),
        audio_channels=int(decoder_config["out_channels"]),
        upsampling_ratios=list(reversed(strides)),
        channel_multiples=[int(value) for value in decoder_config["c_mults"]],
    )
    audio_vae = Magi2AudioVAE(
        decoder=decoder,
        sampling_rate=sampling_rate,
        downsampling_ratio=downsampling_ratio,
    )
    _load_decoder_state(audio_vae, checkpoint_path, len(strides))
    audio_vae.to(device=device, dtype=torch.float32)
    audio_vae.requires_grad_(False)
    return audio_vae.eval()


__all__ = ["Magi2AudioVAE", "load_magi2_audio_vae"]
