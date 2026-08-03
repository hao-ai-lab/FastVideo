#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert raw MiniMax-H3 component weights to FastVideo-native state dicts.

The filename follows the upstream Diffusers conversion PR that defines the
checkpoint contract. FastVideo's native MiniMax-H3 components intentionally use
that same target key surface, so the emitted component folders load directly via
FastVideo's component loader.

The converter reads tensors directly from safetensors. Transformer and video-VAE
weights are streamed and sharded; the audio-VAE mapping is an identity. No model
implementation is imported or instantiated by this script.
"""

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SAFE_WEIGHTS_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"

MINIMAX_H3_TRANSFORMER_CONFIG = {
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "hidden_size": 5376,
    "num_layers": 50,
    "num_refiner_layers": 2,
    "ffn_dim": 14336,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": [1, 2, 2],
    "text_dim": 5120,
    "freq_dim": 256,
    "time_embed_hidden_dim": 5376,
    "time_embed_dim": 2688,
    "rope_freq_dim": 16,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
    "final_norm_eps": 1e-5,
}

# Raw MiniMax-H3 uses FP32 only for these source subtrees. All other
# transformer tensors, including AdaLN projections, are BF16.
MINIMAX_H3_FP32_SOURCE_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
)

# Recomputed exactly by MiniMaxH3RotaryPosEmbed as a non-persistent buffer.
MINIMAX_H3_TRANSFORMER_DROPPED_KEYS = ("rope.inv_freq", )

MINIMAX_H3_VIDEO_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 24,
    "block_out_channels": [128, 256, 256, 512, 512, 1024],
    "layers_per_block": 2,
    "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
    "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
    "norm_num_groups": 32,
    "norm_eps": 1e-6,
    "spatial_padding_mode": "reflect",
    "decoder_num_layers": 36,
    "decoder_num_attention_heads": 32,
    "decoder_attention_head_dim": 64,
    "decoder_num_register_tokens": 4,
    "decoder_ffn_mult": 4,
    "decoder_rope_theta": 100.0,
    "decoder_rope_dim_ratio": 0.75,
    "decoder_norm_eps": 1e-5,
    "clip_length": 17,
    "token_drop": 3,
}

# Training-only, all-zero masked-autoencoding buffer; the released decoder
# never reads it and the native decoder does not persist it.
MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS = ("decoder.mask_token", )

MINIMAX_H3_AUDIO_VAE_FIXED_CONFIG = {
    "num_attention_heads": 8,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
}


def reorder_interleaved_qkv(weight: torch.Tensor, num_attention_heads: int,
                            attention_head_dim: int) -> torch.Tensor:
    """Change raw ``[head0:qkv, head1:qkv, ...]`` rows to ``[q_all;k_all;v_all]``."""
    expected_rows = num_attention_heads * 3 * attention_head_dim
    if weight.shape[0] != expected_rows:
        raise ValueError(
            f"fused qkv tensor has {weight.shape[0]} rows, expected {expected_rows} "
            f"({num_attention_heads} heads * 3 * {attention_head_dim}).")
    grouped = weight.reshape(num_attention_heads, 3 * attention_head_dim,
                             *weight.shape[1:])
    query, key, value = grouped.split(attention_head_dim, dim=1)
    return torch.cat([
        tensor.reshape(num_attention_heads * attention_head_dim,
                       *weight.shape[1:])
        for tensor in (query, key, value)
    ],
                     dim=0).contiguous()


def split_fused_qkv(weight: torch.Tensor, num_attention_heads: int,
                    attention_head_dim: int) -> tuple[torch.Tensor, ...]:
    """Split contiguous ``[q_all;k_all;v_all]`` rows into three projections."""
    inner_dim = num_attention_heads * attention_head_dim
    if weight.shape[0] != 3 * inner_dim:
        raise ValueError(
            f"fused qkv tensor has {weight.shape[0]} rows, expected {3 * inner_dim}.")
    return tuple(tensor.contiguous()
                 for tensor in weight.split(inner_dim, dim=0))


def get_transformer_key_plan(
        config: dict[str, Any]) -> dict[str, list[tuple[str, list[int]]]]:
    """Return the complete raw-to-native Transformer key and shape contract."""
    hidden_size = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = config["attention_head_dim"]
    inner_dim = heads * head_dim
    ffn_dim = config["ffn_dim"]
    time_embed_dim = config["time_embed_dim"]
    video_patch_dim = (config["in_channels"] * math.prod(config["patch_size"]))

    plan: dict[str, list[tuple[str, list[int]]]] = {
        "video_patch_proj.weight": [("proj_in.weight", [hidden_size, video_patch_dim])],
        "video_patch_proj.bias": [("proj_in.bias", [hidden_size])],
        "audio_patch_proj.weight": [("audio_proj_in.weight", [hidden_size, config["audio_in_channels"]])],
        "audio_patch_proj.bias": [("audio_proj_in.bias", [hidden_size])],
        "condition_proj.weight": [("context_embedder.weight", [hidden_size, config["text_dim"]])],
        "condition_proj.bias": [("context_embedder.bias", [hidden_size])],
        "time_embedder.proj_in.weight": [("time_embedder.linear_1.weight", [
            config["time_embed_hidden_dim"], config["freq_dim"]
        ])],
        "time_embedder.proj_in.bias": [("time_embedder.linear_1.bias", [config["time_embed_hidden_dim"]])],
        "time_embedder.proj_out.weight": [("time_embedder.linear_2.weight", [
            time_embed_dim, config["time_embed_hidden_dim"]
        ])],
        "time_embedder.proj_out.bias": [("time_embedder.linear_2.bias", [time_embed_dim])],
        "token_refiner.final_norm.weight": [("token_refiner.final_norm.weight", [hidden_size])],
        "final_layer.norm.weight": [("norm_out.norm.weight", [hidden_size])],
        "final_layer.adaln_proj.linear.weight": [("norm_out.linear.weight", [2 * hidden_size, time_embed_dim])],
        "final_layer.adaln_proj.linear.bias": [("norm_out.linear.bias", [2 * hidden_size])],
        "final_layer.video_out.weight": [("proj_out.weight", [video_patch_dim, hidden_size])],
        "final_layer.video_out.bias": [("proj_out.bias", [video_patch_dim])],
        "final_layer.audio_out.weight": [("audio_proj_out.weight", [config["audio_in_channels"], hidden_size])],
        "final_layer.audio_out.bias": [("audio_proj_out.bias", [config["audio_in_channels"]])],
    }
    for key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
        plan[key] = []

    block_specs = (
        ("blocks", "transformer_blocks", config["num_layers"], True),
        ("token_refiner.blocks", "token_refiner.refiner_blocks",
         config["num_refiner_layers"], False),
    )
    for source_prefix, target_prefix, num_layers, has_adaln in block_specs:
        for index in range(num_layers):
            source = f"{source_prefix}.{index}"
            target = f"{target_prefix}.{index}"
            plan[f"{source}.norm1.weight"] = [(f"{target}.norm1.weight", [hidden_size])]
            plan[f"{source}.norm2.weight"] = [(f"{target}.norm2.weight", [hidden_size])]
            plan[f"{source}.attn.qkv_proj.weight"] = [
                (f"{target}.attn.to_q.weight", [inner_dim, hidden_size]),
                (f"{target}.attn.to_k.weight", [inner_dim, hidden_size]),
                (f"{target}.attn.to_v.weight", [inner_dim, hidden_size]),
            ]
            plan[f"{source}.attn.q_norm.weight"] = [(f"{target}.attn.norm_q.weight", [head_dim])]
            plan[f"{source}.attn.k_norm.weight"] = [(f"{target}.attn.norm_k.weight", [head_dim])]
            plan[f"{source}.attn.out_proj.weight"] = [(f"{target}.attn.to_out.0.weight", [hidden_size, inner_dim])]
            plan[f"{source}.mlp.fc1.weight"] = [(f"{target}.ff.net.0.proj.weight", [2 * ffn_dim, hidden_size])]
            plan[f"{source}.mlp.fc2.weight"] = [(f"{target}.ff.net.2.weight", [hidden_size, ffn_dim])]
            if has_adaln:
                plan[f"{source}.adaln_proj.linear.weight"] = [(
                    f"{target}.adaln_proj.linear.weight",
                    [18 * hidden_size, time_embed_dim],
                )]
                plan[f"{source}.adaln_proj.linear.bias"] = [(f"{target}.adaln_proj.linear.bias", [18 * hidden_size])]
    return plan


def convert_transformer_key(source_key: str, tensor: torch.Tensor,
                            config: dict[str, Any]) -> list[tuple[str, torch.Tensor]]:
    """Convert one in-memory raw Transformer tensor after QKV de-interleaving."""
    if source_key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
        return []

    target_key = source_key
    if target_key.startswith("token_refiner.blocks."):
        target_key = target_key.replace("token_refiner.blocks.",
                                        "token_refiner.refiner_blocks.", 1)
    elif target_key.startswith("blocks."):
        target_key = target_key.replace("blocks.", "transformer_blocks.", 1)
    replacements = (
        ("time_embedder.proj_in.", "time_embedder.linear_1."),
        ("time_embedder.proj_out.", "time_embedder.linear_2."),
        ("video_patch_proj.", "proj_in."),
        ("audio_patch_proj.", "audio_proj_in."),
        ("condition_proj.", "context_embedder."),
        ("final_layer.norm.", "norm_out.norm."),
        ("final_layer.adaln_proj.linear.", "norm_out.linear."),
        ("final_layer.video_out.", "proj_out."),
        ("final_layer.audio_out.", "audio_proj_out."),
        (".attn.q_norm.", ".attn.norm_q."),
        (".attn.k_norm.", ".attn.norm_k."),
        (".attn.out_proj.", ".attn.to_out.0."),
    )
    for source, target in replacements:
        target_key = target_key.replace(source, target)

    if target_key.endswith(".attn.qkv_proj.weight"):
        query, key, value = split_fused_qkv(tensor,
                                            config["num_attention_heads"],
                                            config["attention_head_dim"])
        prefix = target_key.removesuffix("qkv_proj.weight")
        return [(f"{prefix}to_q.weight", query),
                (f"{prefix}to_k.weight", key),
                (f"{prefix}to_v.weight", value)]

    if target_key.endswith(".mlp.fc1.weight"):
        gate, value = tensor.chunk(2, dim=0)
        target_key = target_key.replace(".mlp.fc1.weight",
                                        ".ff.net.0.proj.weight")
        return [(target_key, torch.cat([value, gate], dim=0).contiguous())]

    return [(target_key.replace(".mlp.fc2.", ".ff.net.2."), tensor)]


def _rename_video_vae_key(source_key: str) -> str:
    target_key = source_key
    if target_key.startswith("encoder.down."):
        level, rest = target_key.removeprefix("encoder.down.").split(".", 1)
        rest = rest.replace("block.", "resnets.", 1)
        rest = rest.replace("nin_shortcut.", "conv_shortcut.", 1)
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        target_key = f"encoder.down_blocks.{level}.{rest}"
    target_key = target_key.replace("decoder.x_embedder.", "decoder.proj_in.")
    target_key = target_key.replace(".attn.to_out.", ".attn.to_out.0.")
    target_key = target_key.replace(".ff.w1.", ".ff.net.0.proj.")
    return target_key.replace(".ff.w2.", ".ff.net.2.")


def convert_video_vae_key(source_key: str, tensor: torch.Tensor,
                          config: dict[str, Any]) -> list[tuple[str, torch.Tensor]]:
    """Convert one original video-VAE key/tensor pair to native key(s)."""
    if source_key in MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS:
        return []
    if ".attn.to_qkv." in source_key:
        reordered = reorder_interleaved_qkv(
            tensor, config["decoder_num_attention_heads"],
            config["decoder_attention_head_dim"])
        query, key, value = split_fused_qkv(
            reordered, config["decoder_num_attention_heads"],
            config["decoder_attention_head_dim"])
        prefix, suffix = source_key.split(".attn.to_qkv.")
        return [(f"{prefix}.attn.to_q.{suffix}", query),
                (f"{prefix}.attn.to_k.{suffix}", key),
                (f"{prefix}.attn.to_v.{suffix}", value)]

    target_key = _rename_video_vae_key(source_key)
    if ".ff.w1." in source_key:
        gate, up = tensor.chunk(2, dim=0)
        tensor = torch.cat([up, gate], dim=0).contiguous()
    return [(target_key, tensor)]


def get_video_vae_key_plan(config: dict[str, Any]) -> dict[str, list[str]]:
    """Return the complete raw-to-native video-VAE key contract."""
    channels = list(config["block_out_channels"])
    input_channels = [channels[0]] + channels[:-1]
    plan: dict[str, list[str]] = {}

    def renamed(*keys: str) -> None:
        for key in keys:
            plan[key] = [_rename_video_vae_key(key)]

    renamed("quant_conv.weight", "quant_conv.bias", "post_quant_conv.weight",
            "post_quant_conv.bias", "encoder.conv_in.weight",
            "encoder.conv_in.bias")
    for level, (in_channels,
                out_channels) in enumerate(zip(input_channels, channels)):
        for index in range(config["layers_per_block"]):
            prefix = f"encoder.down.{level}.block.{index}"
            for name in ("norm1", "conv1", "norm2", "conv2"):
                renamed(f"{prefix}.{name}.weight", f"{prefix}.{name}.bias")
            if (in_channels if index == 0 else out_channels) != out_channels:
                renamed(f"{prefix}.nin_shortcut.weight",
                        f"{prefix}.nin_shortcut.bias")
        if (config["spatial_downsample_factors"][level] *
                config["temporal_downsample_factors"][level] > 1):
            renamed(f"encoder.down.{level}.downsample.conv.weight",
                    f"encoder.down.{level}.downsample.conv.bias")
    renamed("encoder.norm_out.weight", "encoder.norm_out.bias",
            "encoder.conv_out.weight", "encoder.conv_out.bias",
            "decoder.x_embedder.weight", "decoder.x_embedder.bias",
            "decoder.register_tokens", "decoder.norm_out.weight",
            "decoder.norm_out.bias", "decoder.proj_out.weight",
            "decoder.proj_out.bias")

    for index in range(config["decoder_num_layers"]):
        prefix = f"decoder.transformer_blocks.{index}"
        renamed(f"{prefix}.norm1.weight", f"{prefix}.norm2.weight",
                f"{prefix}.scale1", f"{prefix}.scale2")
        for suffix in ("weight", "bias"):
            plan[f"{prefix}.attn.to_qkv.{suffix}"] = [
                f"{prefix}.attn.to_q.{suffix}",
                f"{prefix}.attn.to_k.{suffix}",
                f"{prefix}.attn.to_v.{suffix}",
            ]
            plan[f"{prefix}.attn.to_out.{suffix}"] = [
                f"{prefix}.attn.to_out.0.{suffix}"
            ]
            plan[f"{prefix}.ff.w1.{suffix}"] = [
                f"{prefix}.ff.net.0.proj.{suffix}"
            ]
            plan[f"{prefix}.ff.w2.{suffix}"] = [
                f"{prefix}.ff.net.2.{suffix}"
            ]
    for key in MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS:
        plan[key] = []
    return plan


def get_audio_vae_config(checkpoint_path: str | Path) -> dict[str, Any]:
    """Derive the native audio-VAE config from original metadata and wrapper config."""
    source_dir = Path(checkpoint_path) / "audio_vae"
    with (source_dir / "metadata.json").open() as handle:
        kwargs = json.load(handle)["metadata"]["kwargs"]
    with (source_dir / "config.json").open() as handle:
        wrapper = json.load(handle)

    if kwargs["decoder_type"] != "bigvgan":
        raise ValueError(
            f"Only the BigVGAN decoder is supported, got {kwargs['decoder_type']!r}.")
    if not kwargs["attn_proj"]:
        raise ValueError("MiniMax-H3 audio VAE requires the causal-attention latent projection.")
    latent_channels = kwargs["vae_latent_channels"]
    if wrapper["latent_channels"] != latent_channels:
        raise ValueError("audio latent width differs between metadata.json and config.json.")
    if wrapper["sample_rate"] != kwargs["sample_rate"]:
        raise ValueError("audio sample rate differs between metadata.json and config.json.")

    config = {
        "encoder_dim": kwargs["encoder_dim"],
        "encoder_rates": kwargs["encoder_rates"],
        "latent_dim": kwargs["latent_dim"],
        "latent_channels": latent_channels,
        "decoder_dim": kwargs["decoder_dim"],
        "decoder_rates": kwargs["decoder_rates"],
        "decoder_kernel_sizes": [
            2 * rate - (rate % 2) for rate in kwargs["decoder_rates"]
        ],
        **MINIMAX_H3_AUDIO_VAE_FIXED_CONFIG,
        "sampling_rate": kwargs["sample_rate"],
        "latents_mean": wrapper["latents_mean"],
        "latents_std": wrapper["latents_std"],
    }
    _validate_audio_vae_config(config)
    return config


def _validate_audio_vae_config(config: dict[str, Any]) -> None:
    latent_channels = config["latent_channels"]
    for key in ("latents_mean", "latents_std"):
        if len(config[key]) != latent_channels:
            raise ValueError(
                f"audio {key} has {len(config[key])} values, expected {latent_channels}.")
    if len(config["decoder_rates"]) != len(config["decoder_kernel_sizes"]):
        raise ValueError("audio decoder_rates and decoder_kernel_sizes must have equal length.")


def _prepare_output_directory(output_path: str | Path) -> Path:
    output = Path(output_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


class _ShardWriter:
    def __init__(self, output_path: str | Path, max_shard_size: int):
        if max_shard_size <= 0:
            raise ValueError("max_shard_size must be positive.")
        self.output_path = _prepare_output_directory(output_path)
        self.max_shard_size = max_shard_size
        self.buffer: dict[str, torch.Tensor] = {}
        self.buffer_size = 0
        self.total_size = 0
        self.temporary_paths: list[Path] = []
        self.temporary_weight_map: dict[str, Path] = {}
        self.target_keys: set[str] = set()

    def add(self, key: str, tensor: torch.Tensor) -> None:
        if key in self.target_keys:
            raise KeyError(f"Duplicate converted target key: {key}")
        tensor = tensor.detach().cpu().contiguous()
        tensor_size = tensor.numel() * tensor.element_size()
        if self.buffer and self.buffer_size + tensor_size > self.max_shard_size:
            self._flush()
        self.buffer[key] = tensor
        self.buffer_size += tensor_size
        self.total_size += tensor_size
        self.target_keys.add(key)

    def _flush(self) -> None:
        if not self.buffer:
            return
        path = self.output_path / f".tmp-shard-{len(self.temporary_paths):05d}.safetensors"
        save_file(self.buffer, str(path), metadata={"format": "pt"})
        for key in self.buffer:
            self.temporary_weight_map[key] = path
        self.temporary_paths.append(path)
        self.buffer = {}
        self.buffer_size = 0

    def finish(self) -> dict[str, int]:
        self._flush()
        if not self.temporary_paths:
            raise ValueError("No tensors were produced by the conversion.")
        shard_count = len(self.temporary_paths)
        renames = {
            path: self.output_path /
            f"diffusion_pytorch_model-{index + 1:05d}-of-{shard_count:05d}.safetensors"
            for index, path in enumerate(self.temporary_paths)
        }
        for source, target in renames.items():
            source.rename(target)
        index = {
            "metadata": {
                "total_size": self.total_size
            },
            "weight_map": {
                key: renames[path].name
                for key, path in self.temporary_weight_map.items()
            },
        }
        with (self.output_path / SAFE_WEIGHTS_INDEX_NAME).open("w") as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
        return {
            "target_keys": len(self.target_keys),
            "shards": shard_count,
            "total_size": self.total_size,
        }


def _validate_planned_source_keys(paths: list[str], plan: dict[str, Any]) -> None:
    seen: set[str] = set()
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in seen:
                    raise KeyError(f"Duplicate source key across shards: {key}")
                if key not in plan:
                    raise KeyError(f"Unexpected key in {os.path.basename(path)}: {key}")
                seen.add(key)
    missing = sorted(set(plan) - seen)
    if missing:
        raise KeyError(
            f"{len(missing)} planned source key(s) missing, e.g. {missing[:5]}.")


def _convert_planned_safetensors(
        paths: list[str], output_path: str | Path, plan: dict[str, Any],
        convert: Callable[[str, torch.Tensor], list[tuple[str, torch.Tensor]]],
        expected_dtype: Callable[[str], torch.dtype],
        max_shard_size: int) -> dict[str, int]:
    _validate_planned_source_keys(paths, plan)
    writer = _ShardWriter(output_path, max_shard_size)
    source_count = 0
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for source_key in handle.keys():
                source_count += 1
                converted = convert(source_key, handle.get_tensor(source_key))
                expected_targets = plan[source_key]
                if expected_targets and isinstance(expected_targets[0], tuple):
                    expected_names = [item[0] for item in expected_targets]
                    expected_shapes = {
                        item[0]: tuple(item[1])
                        for item in expected_targets
                    }
                else:
                    expected_names = list(expected_targets)
                    expected_shapes = {}
                actual_names = [key for key, _ in converted]
                if actual_names != expected_names:
                    raise KeyError(
                        f"{source_key}: produced {actual_names}, expected {expected_names}.")
                for target_key, tensor in converted:
                    dtype = expected_dtype(source_key)
                    if tensor.dtype != dtype:
                        raise ValueError(
                            f"{source_key}: expected {dtype}, got {tensor.dtype}.")
                    if target_key in expected_shapes and tuple(
                            tensor.shape) != expected_shapes[target_key]:
                        raise ValueError(
                            f"{source_key} -> {target_key}: got shape {tuple(tensor.shape)}, "
                            f"expected {expected_shapes[target_key]}.")
                    writer.add(target_key, tensor)
    report = writer.finish()
    report["source_keys"] = source_count
    return report


def _write_component_config(output_path: str | Path, class_name: str,
                            config: dict[str, Any]) -> None:
    with (Path(output_path) / "config.json").open("w") as handle:
        json.dump({"_class_name": class_name, **config}, handle, indent=2)


def convert_transformer(checkpoint_path: str | Path, output_path: str | Path,
                        config: dict[str, Any],
                        max_shard_size: int) -> dict[str, int]:
    """Stream one raw MiniMax-H3 Transformer variant to native keys."""
    transformer_dir = Path(checkpoint_path) / "transformer"
    shards = sorted(glob.glob(str(transformer_dir / "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"No `*.safetensors` shards found under {transformer_dir}.")
    plan = get_transformer_key_plan(config)

    def convert(source_key: str,
                tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        if source_key.endswith(".attn.qkv_proj.weight"):
            tensor = reorder_interleaved_qkv(
                tensor, config["num_attention_heads"],
                config["attention_head_dim"])
        return convert_transformer_key(source_key, tensor, config)

    def expected_dtype(source_key: str) -> torch.dtype:
        return (torch.float32 if source_key.startswith(
            MINIMAX_H3_FP32_SOURCE_PREFIXES) else torch.bfloat16)

    report = _convert_planned_safetensors(shards, output_path, plan, convert,
                                          expected_dtype, max_shard_size)
    _write_component_config(output_path, "MiniMaxH3Transformer3DModel", config)
    return report


def _resolve_video_vae_source(checkpoint_path: str | Path,
                              config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    source_dir = Path(checkpoint_path) / "video_vae"
    with (source_dir / "config.json").open() as handle:
        wrapper = json.load(handle)
    weights_path = source_dir / wrapper.get("source_path", "source") / wrapper.get(
        "source_safetensors_path", "model.safetensors")
    resolved = dict(config)
    for key in ("latents_mean", "latents_std"):
        if key not in resolved:
            if key not in wrapper:
                raise KeyError(f"{source_dir / 'config.json'} does not carry `{key}`.")
            resolved[key] = wrapper[key]
        if len(resolved[key]) != resolved["latent_channels"]:
            raise ValueError(
                f"video {key} has {len(resolved[key])} values, expected {resolved['latent_channels']}.")
    return weights_path, resolved


def convert_video_vae(checkpoint_path: str | Path, output_path: str | Path,
                      config: dict[str, Any],
                      max_shard_size: int) -> dict[str, int]:
    """Stream the raw video VAE to native keys."""
    weights_path, resolved_config = _resolve_video_vae_source(
        checkpoint_path, config)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    plan = get_video_vae_key_plan(resolved_config)
    report = _convert_planned_safetensors(
        [str(weights_path)], output_path, plan,
        lambda key, tensor: convert_video_vae_key(key, tensor, resolved_config),
        lambda _key: torch.float32, max_shard_size)
    _write_component_config(output_path, "AutoencoderKLMiniMaxH3",
                            resolved_config)
    return report


def convert_audio_vae(checkpoint_path: str | Path, output_path: str | Path,
                      config: dict[str, Any] | None = None) -> dict[str, int]:
    """Copy the name-for-name FP32 audio-VAE state dict and emit its config."""
    resolved_config = (get_audio_vae_config(checkpoint_path)
                       if config is None else dict(config))
    _validate_audio_vae_config(resolved_config)
    weights_path = Path(checkpoint_path) / "audio_vae" / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    output = _prepare_output_directory(output_path)
    state_dict: dict[str, torch.Tensor] = {}
    total_size = 0
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if tensor.dtype != torch.float32:
                raise ValueError(f"{key}: expected torch.float32, got {tensor.dtype}.")
            state_dict[key] = tensor.contiguous()
            total_size += tensor.numel() * tensor.element_size()
    if not state_dict:
        raise ValueError(f"No tensors found in {weights_path}.")
    save_file(state_dict,
              str(output / "diffusion_pytorch_model.safetensors"),
              metadata={"format": "pt"})
    _write_component_config(output, "AutoencoderKLMiniMaxH3Audio",
                            resolved_config)
    return {
        "source_keys": len(state_dict),
        "target_keys": len(state_dict),
        "shards": 1,
        "total_size": total_size,
    }


def write_scheduler_configs(checkpoint_path: str | Path,
                            output_path: str | Path) -> dict[str, float]:
    """Write role-specific video/audio flow-scheduler configs from the raw index."""
    model_index_path = Path(checkpoint_path) / "model_index.json"
    with model_index_path.open() as handle:
        shift_scales = json.load(handle)["_minimax_h3"]["sigma_shift_scales"]

    shifts = {
        "scheduler": float(shift_scales["video"]),
        "audio_scheduler": float(shift_scales["audio"]),
    }
    for role, shift in shifts.items():
        role_path = _prepare_output_directory(Path(output_path) / role)
        with (role_path / "scheduler_config.json").open("w") as handle:
            json.dump({
                "_class_name": "MiniMaxH3Scheduler",
                "shift": shift,
            },
                      handle,
                      indent=2)
    return shifts


def load_safetensors_directory(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a converted component directory; intended for validation/tests."""
    state_dict: dict[str, torch.Tensor] = {}
    for weights_path in sorted(Path(path).glob("*.safetensors")):
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in state_dict:
                    raise KeyError(f"Duplicate key across converted shards: {key}")
                state_dict[key] = handle.get_tensor(key)
    if not state_dict:
        raise FileNotFoundError(f"No converted safetensors files under {path}.")
    return state_dict


def convert_checkpoint(
        checkpoint_path: str | Path,
        output_path: str | Path,
        transformer_config: dict[str, Any] | None = None,
        video_vae_config: dict[str, Any] | None = None,
        audio_vae_config: dict[str, Any] | None = None,
        max_shard_size: int = 5 * 1024**3) -> dict[str, dict[str, int]]:
    """Convert the three native MiniMax-H3 components in one variant folder."""
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    reports = {
        "transformer": convert_transformer(
            checkpoint_path, output / "transformer",
            transformer_config or MINIMAX_H3_TRANSFORMER_CONFIG,
            max_shard_size),
        "vae": convert_video_vae(checkpoint_path, output / "vae",
                                 video_vae_config
                                 or MINIMAX_H3_VIDEO_VAE_CONFIG,
                                 max_shard_size),
        "audio_vae": convert_audio_vae(checkpoint_path,
                                       output / "audio_vae",
                                       audio_vae_config),
    }
    write_scheduler_configs(checkpoint_path, output)
    return reports


def _read_config_override(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path) as handle:
        config = json.load(handle)
    return {key: value for key, value in config.items() if not key.startswith("_")}


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw MiniMax-H3 weights to FastVideo-native component folders.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--transformer_config",
        help="Optional JSON override; defaults to the released H3 architecture.")
    parser.add_argument(
        "--video_vae_config",
        help="Optional JSON override; defaults to the released H3 video VAE.")
    parser.add_argument(
        "--audio_vae_config",
        help="Optional JSON override; otherwise derived from audio_vae metadata.")
    parser.add_argument("--max_shard_size",
                        type=int,
                        default=5 * 1024**3,
                        help="Maximum output shard size in bytes.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = get_args(argv)
    reports = convert_checkpoint(
        args.checkpoint_path,
        args.output_path,
        transformer_config=_read_config_override(args.transformer_config),
        video_vae_config=_read_config_override(args.video_vae_config),
        audio_vae_config=_read_config_override(args.audio_vae_config),
        max_shard_size=args.max_shard_size,
    )
    for component, report in reports.items():
        print(
            f"{component}: {report['source_keys']} raw keys -> {report['target_keys']} native keys, "
            f"{report['shards']} shard(s), {report['total_size'] / 1024**3:.2f} GiB.")


if __name__ == "__main__":
    main()
