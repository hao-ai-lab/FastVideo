#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a WanTrack initialization checkpoint from Wan2.1-Fun Control weights.

The VideoX-Fun Control DiT receives 48 input channels in this order:

    noisy latent (16) | spatial control (16) | reference latent (16)

WanTrack receives 52 input channels:

    noisy latent (16) | I2V mask (4) | first-frame latent (16) | track map (16)

This converter preserves the pretrained control pathway by moving the Fun
Control channels into WanTrack's track slot. The four new I2V mask channels are
initialized to zero. The remainder of the model is renamed to the Diffusers
key layout and checked against a compatible Fun-InP Diffusers checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn


SOURCE_IN_CHANNELS = 48
TARGET_IN_CHANNELS = 52
LATENT_CHANNELS = 16
TRACK_CHANNELS = 16
VAE_SPATIAL_COMPRESSION = 8
VAE_TEMPORAL_COMPRESSION = 4

# VideoX-Fun Wan names to the Diffusers/FastVideo Wan state-dict surface.
# patch_embedding is handled separately because its input channels are remapped.
PARAM_NAME_MAP: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in {
        r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.linear_1.\1",
        r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.linear_2.\1",
        r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.linear_1.\1",
        r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.linear_2.\1",
        r"^time_projection\.1\.(.*)$": r"condition_embedder.time_proj.\1",
        r"^img_emb\.proj\.0\.(.*)$": r"condition_embedder.image_embedder.norm1.\1",
        r"^img_emb\.proj\.1\.(.*)$": r"condition_embedder.image_embedder.ff.net.0.proj.\1",
        r"^img_emb\.proj\.3\.(.*)$": r"condition_embedder.image_embedder.ff.net.2.\1",
        r"^img_emb\.proj\.4\.(.*)$": r"condition_embedder.image_embedder.norm2.\1",
        r"^head\.modulation$": r"scale_shift_table",
        r"^head\.head\.(.*)$": r"proj_out.\1",
        r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.attn1.to_q.\2",
        r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.attn1.to_k.\2",
        r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.attn1.to_v.\2",
        r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.attn1.to_out.0.\2",
        r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.attn1.norm_q.\2",
        r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.attn1.norm_k.\2",
        r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
        r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
        r"^blocks\.(\d+)\.cross_attn\.k_img\.(.*)$": r"blocks.\1.attn2.add_k_proj.\2",
        r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
        r"^blocks\.(\d+)\.cross_attn\.v_img\.(.*)$": r"blocks.\1.attn2.add_v_proj.\2",
        r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.0.\2",
        r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
        r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
        r"^blocks\.(\d+)\.cross_attn\.norm_k_img\.(.*)$": r"blocks.\1.attn2.norm_added_k.\2",
        r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.net.0.proj.\2",
        r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.net.2.\2",
        r"^blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
        r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.norm2.\2",
    }.items()
)


def load_sharded_safetensors(directory: Path) -> dict[str, torch.Tensor]:
    """Load every safetensors shard in a directory and reject duplicate keys."""
    shard_paths = sorted(directory.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No safetensors files found under {directory}")

    state_dict: dict[str, torch.Tensor] = {}
    for shard_path in shard_paths:
        shard = load_file(str(shard_path), device="cpu")
        duplicate_keys = state_dict.keys() & shard.keys()
        if duplicate_keys:
            examples = ", ".join(sorted(duplicate_keys)[:8])
            raise RuntimeError(f"Duplicate parameter keys across base shards: {examples}")
        state_dict.update(shard)
    return state_dict


def rename_control_state_dict(source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename all VideoX-Fun parameters and reject unknown or colliding keys."""
    converted: dict[str, torch.Tensor] = {}
    for source_key, value in source.items():
        if source_key.startswith("patch_embedding."):
            target_key = source_key
        else:
            target_key = ""
            for pattern, replacement in PARAM_NAME_MAP:
                if pattern.match(source_key):
                    target_key = pattern.sub(replacement, source_key)
                    break
            if not target_key:
                raise ValueError(f"No target mapping for source parameter: {source_key}")

        if target_key in converted:
            raise RuntimeError(f"Multiple source parameters map to target parameter: {target_key}")
        converted[target_key] = value
    return converted


def synthesize_image_query_norms(
    converted: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> None:
    """Create Diffusers image-query norm tensors absent from VideoX-Fun."""
    expected_keys = sorted(key for key in reference if key.endswith("attn2.norm_added_q.weight"))
    for query_key in expected_keys:
        key_norm_key = query_key.replace("norm_added_q", "norm_added_k")
        if key_norm_key not in converted:
            raise RuntimeError(f"Cannot synthesize {query_key}: converted state lacks {key_norm_key}")
        if query_key in converted:
            raise RuntimeError(f"Refusing to overwrite converted parameter: {query_key}")
        converted[query_key] = torch.zeros_like(converted[key_norm_key])


def remap_patch_embedding(source_weight: torch.Tensor) -> torch.Tensor:
    """Remap Fun Control's 48-channel patch projection into WanTrack's 52 channels."""
    if source_weight.ndim != 5:
        raise ValueError(f"Expected a 5D patch embedding weight, got shape {tuple(source_weight.shape)}")
    if source_weight.shape[1] != SOURCE_IN_CHANNELS:
        raise ValueError(
            f"Expected patch_embedding.weight input channels={SOURCE_IN_CHANNELS}, "
            f"got {source_weight.shape[1]}"
        )

    target_shape = (source_weight.shape[0], TARGET_IN_CHANNELS, *source_weight.shape[2:])
    target_weight = source_weight.new_zeros(target_shape)
    target_weight[:, 0:16] = source_weight[:, 0:16]
    target_weight[:, 20:36] = source_weight[:, 32:48]
    target_weight[:, 36:52] = source_weight[:, 16:32]

    if torch.count_nonzero(target_weight[:, 16:20]).item() != 0:
        raise RuntimeError("WanTrack I2V mask channels must be initialized to zero")
    return target_weight


def validate_converted_body(
    converted: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> None:
    """Require the converted Wan body to match the Fun-InP key and shape surface."""
    converted_keys = set(converted)
    reference_keys = set(reference)
    missing = sorted(reference_keys - converted_keys)
    unexpected = sorted(converted_keys - reference_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Converted transformer keys do not match the Fun-InP reference. "
            f"missing={missing[:8]} ({len(missing)} total), "
            f"unexpected={unexpected[:8]} ({len(unexpected)} total)"
        )

    patch_key = "patch_embedding.weight"
    if patch_key not in converted:
        raise RuntimeError(f"Converted transformer lacks required parameter: {patch_key}")
    reference_patch = reference[patch_key]
    converted_patch = converted[patch_key]
    if reference_patch.ndim != converted_patch.ndim:
        raise RuntimeError(
            f"{patch_key} rank mismatch: reference={reference_patch.ndim}, converted={converted_patch.ndim}"
        )
    if reference_patch.shape[0] != converted_patch.shape[0] or reference_patch.shape[2:] != converted_patch.shape[2:]:
        raise RuntimeError(
            f"{patch_key} non-input dimensions differ: "
            f"reference={tuple(reference_patch.shape)}, converted={tuple(converted_patch.shape)}"
        )
    if converted_patch.shape[1] != TARGET_IN_CHANNELS:
        raise RuntimeError(f"{patch_key} must have {TARGET_IN_CHANNELS} input channels")

    shape_mismatches = []
    for key in sorted(reference_keys - {patch_key}):
        if converted[key].shape != reference[key].shape:
            shape_mismatches.append((key, tuple(reference[key].shape), tuple(converted[key].shape)))
    if shape_mismatches:
        raise RuntimeError(
            "Converted transformer shapes do not match the Fun-InP reference: "
            f"{shape_mismatches[:8]} ({len(shape_mismatches)} total)"
        )


def build_track_encoder_state(
    id_dim: int,
    dtype: torch.dtype,
    seed: int,
    zero_init_head: bool,
) -> dict[str, torch.Tensor]:
    """Initialize the two bias-free TrackEncoder convolution weights reproducibly."""
    if id_dim <= 0:
        raise ValueError(f"--id-dim must be positive, got {id_dim}")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        temporal_conv = nn.Conv3d(
            id_dim,
            TRACK_CHANNELS,
            kernel_size=(VAE_TEMPORAL_COMPRESSION, 1, 1),
            stride=(VAE_TEMPORAL_COMPRESSION, 1, 1),
            bias=False,
        )
        projection = nn.Conv3d(TRACK_CHANNELS, TRACK_CHANNELS, kernel_size=1, bias=False)
        if zero_init_head:
            nn.init.zeros_(projection.weight)

    state = {
        "track_encoder.temporal_conv.weight": temporal_conv.weight.detach().to(dtype=dtype).contiguous(),
        "track_encoder.proj.weight": projection.weight.detach().to(dtype=dtype).contiguous(),
    }
    expected_shapes = {
        "track_encoder.temporal_conv.weight": (
            TRACK_CHANNELS,
            id_dim,
            VAE_TEMPORAL_COMPRESSION,
            1,
            1,
        ),
        "track_encoder.proj.weight": (TRACK_CHANNELS, TRACK_CHANNELS, 1, 1, 1),
    }
    for key, expected_shape in expected_shapes.items():
        if state[key].shape != expected_shape:
            raise RuntimeError(f"Unexpected initialized shape for {key}: {tuple(state[key].shape)}")
    return state


def copy_passthrough_components(source_dir: Path, output_dir: Path) -> None:
    """Copy every base bundle entry except the transformer directory."""
    for source_path in sorted(source_dir.iterdir(), key=lambda path: path.name):
        if source_path.name == "transformer":
            continue
        target_path = output_dir / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path, symlinks=True)
        else:
            shutil.copy2(source_path, target_path, follow_symlinks=False)


def update_model_index(output_dir: Path, transformer_class: str) -> None:
    """Point the copied bundle index at the selected FastVideo transformer."""
    model_index_path = output_dir / "model_index.json"
    if not model_index_path.is_file():
        raise FileNotFoundError(f"Fun-InP base does not provide {model_index_path.name}")

    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    transformer_entry = model_index.get("transformer")
    if not isinstance(transformer_entry, list) or len(transformer_entry) != 2:
        raise ValueError(f"Unsupported transformer entry in {model_index_path}: {transformer_entry!r}")
    transformer_entry[1] = transformer_class
    model_index_path.write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")


def write_transformer_config(
    base_dir: Path,
    output_dir: Path,
    transformer_class: str,
    id_dim: int,
    max_track_id: int,
    zero_init_head: bool,
) -> None:
    """Write the widened, loader-compatible WanTrack transformer config."""
    source_config_path = base_dir / "transformer" / "config.json"
    if not source_config_path.is_file():
        raise FileNotFoundError(f"Missing Fun-InP transformer config: {source_config_path}")

    config = json.loads(source_config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "_class_name": transformer_class,
            "in_channels": TARGET_IN_CHANNELS,
            "out_channels": LATENT_CHANNELS,
            "track_config": {
                "id_dim": id_dim,
                "track_channels": TRACK_CHANNELS,
                "vae_spatial_compression": VAE_SPATIAL_COMPRESSION,
                "vae_temporal_compression": VAE_TEMPORAL_COMPRESSION,
                "max_track_id": max_track_id,
                "zero_init_head": zero_init_head,
                "use_bias": False,
            },
        }
    )
    config_path = output_dir / "transformer" / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def validate_paths(base_dir: Path, control_checkpoint: Path, output_dir: Path) -> None:
    """Validate input paths and refuse ambiguous or destructive output layouts."""
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Fun-InP base directory does not exist: {base_dir}")
    if not control_checkpoint.is_file():
        raise FileNotFoundError(f"Fun Control checkpoint does not exist: {control_checkpoint}")
    if control_checkpoint.suffix != ".safetensors":
        raise ValueError(f"Fun Control checkpoint must be a .safetensors file: {control_checkpoint}")

    base_resolved = base_dir.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == base_resolved:
        raise ValueError("Output directory must differ from the Fun-InP base directory")
    if output_resolved.is_relative_to(base_resolved) or base_resolved.is_relative_to(output_resolved):
        raise ValueError("Output directory and Fun-InP base directory must not contain one another")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert VideoX-Fun Wan Control weights into a FastVideo WanTrack bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inp-base",
        required=True,
        type=Path,
        help="Compatible Fun-InP Diffusers bundle used for config, key validation, and passthrough components.",
    )
    parser.add_argument(
        "--control-ckpt",
        required=True,
        type=Path,
        help="VideoX-Fun Wan Control diffusion_pytorch_model.safetensors.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the converted WanTrack bundle.")
    parser.add_argument("--id-dim", type=int, default=128, help="Sinusoidal track-ID embedding dimension.")
    parser.add_argument(
        "--max-track-id",
        type=int,
        default=100_000,
        help="Exclusive upper bound for sampled track IDs.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used to initialize TrackEncoder weights.")
    parser.add_argument(
        "--zero-init-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero-initialize TrackEncoder's final projection.",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Emit CausalTrackWanTransformer3DModel instead of the bidirectional transformer class.",
    )
    return parser.parse_args()


def main() -> None:
    """Convert the checkpoint and create a complete WanTrack bundle."""
    args = parse_args()
    if args.max_track_id <= 0:
        raise ValueError(f"--max-track-id must be positive, got {args.max_track_id}")
    validate_paths(args.inp_base, args.control_ckpt, args.out)

    reference_state = load_sharded_safetensors(args.inp_base / "transformer")
    control_state = load_file(str(args.control_ckpt), device="cpu")
    converted_state = rename_control_state_dict(control_state)
    synthesize_image_query_norms(converted_state, reference_state)

    patch_key = "patch_embedding.weight"
    if patch_key not in converted_state:
        raise RuntimeError(f"Fun Control checkpoint lacks required parameter: {patch_key}")
    converted_state[patch_key] = remap_patch_embedding(converted_state[patch_key])
    validate_converted_body(converted_state, reference_state)

    converted_state.update(
        build_track_encoder_state(
            id_dim=args.id_dim,
            dtype=converted_state[patch_key].dtype,
            seed=args.seed,
            zero_init_head=args.zero_init_head,
        )
    )

    transformer_class = "CausalTrackWanTransformer3DModel" if args.causal else "TrackWanTransformer3DModel"
    args.out.mkdir(parents=True, exist_ok=True)
    copy_passthrough_components(args.inp_base, args.out)
    (args.out / "transformer").mkdir()
    update_model_index(args.out, transformer_class)
    write_transformer_config(
        base_dir=args.inp_base,
        output_dir=args.out,
        transformer_class=transformer_class,
        id_dim=args.id_dim,
        max_track_id=args.max_track_id,
        zero_init_head=args.zero_init_head,
    )

    output_weights = args.out / "transformer" / "diffusion_pytorch_model.safetensors"
    contiguous_state = {key: value.contiguous() for key, value in converted_state.items()}
    save_file(contiguous_state, str(output_weights), metadata={"format": "pt"})
    print(
        f"Converted {len(reference_state)} Wan parameters and 2 TrackEncoder parameters "
        f"to {output_weights} ({transformer_class})."
    )


if __name__ == "__main__":
    main()
