#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Convert GEN3C checkpoint (nvidia/GEN3C-Cosmos-7B) to FastVideo (diffusers) format.

Usage:
    # Convert from local checkpoint (memory-efficient mode for limited RAM)
    python convert_gen3c_to_fastvideo.py --source ./official_weights/GEN3C-Cosmos-7B/model.pt --output ./gen3c_fastvideo

    # Download and convert from HuggingFace
    python convert_gen3c_to_fastvideo.py --download nvidia/GEN3C-Cosmos-7B --output ./gen3c_fastvideo

    # Analyze checkpoint structure only (low memory)
    python convert_gen3c_to_fastvideo.py --source ./model.pt --analyze

    # Convert with fp16 to reduce output size (and memory during save)
    python convert_gen3c_to_fastvideo.py --source ./model.pt --output ./gen3c_fastvideo --dtype fp16
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import torch
from safetensors.torch import save_file

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    hf_hub_download = None
    snapshot_download = None


# Parameter name mapping from official GEN3C checkpoint to FastVideo format
# Based on fastvideo/configs/models/dits/gen3c.py
# The actual GEN3C checkpoint uses AdaLN-LoRA with decomposed projections (index 0 = base, 1 = LoRA)
PARAM_NAMES_MAPPING: dict[str, str] = {
    # Patch embedding: net.x_embedder.proj.1.weight -> patch_embed.proj.weight
    r"^net\.x_embedder\.proj\.1\.(.*)$": r"patch_embed.proj.\1",

    # Time embedding
    r"^net\.t_embedder\.0\.(.*)$": r"time_embed.time_proj.\1",
    r"^net\.t_embedder\.1\.linear_1\.(.*)$": r"time_embed.t_embedder.linear_1.\1",
    r"^net\.t_embedder\.1\.linear_2\.(.*)$": r"time_embed.t_embedder.linear_2.\1",
    
    # Augment sigma embedding (GEN3C-specific)
    r"^net\.augment_sigma_embedder\.0\.(.*)$": r"augment_sigma_embed.time_proj.\1",
    r"^net\.augment_sigma_embedder\.1\.linear_1\.(.*)$": r"augment_sigma_embed.t_embedder.linear_1.\1",
    r"^net\.augment_sigma_embedder\.1\.linear_2\.(.*)$": r"augment_sigma_embed.t_embedder.linear_2.\1",

    # Affine embedding norm
    r"^net\.affline_norm\.(.*)$": r"affine_norm.\1",

    # Extra positional embeddings (learnable per-axis)
    r"^net\.extra_pos_embedder\.pos_emb_t$": r"learnable_pos_embed.pos_emb_t",
    r"^net\.extra_pos_embedder\.pos_emb_h$": r"learnable_pos_embed.pos_emb_h",
    r"^net\.extra_pos_embedder\.pos_emb_w$": r"learnable_pos_embed.pos_emb_w",

    # Transformer blocks: net.blocks.blockN -> transformer_blocks.N
    # GEN3C uses attn.to_q.0/1 pattern (0=base weight, 1=LoRA weight)
    
    # Self-attention (block index 0)
    # Q projection: to_q.0 is linear, to_q.1 is QK norm (RMSNorm applied per-head)
    r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.attn\.to_q\.0\.(.*)$": r"transformer_blocks.\1.attn1.to_q.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.attn\.to_q\.1\.(.*)$": r"transformer_blocks.\1.attn1.norm_q.\2",
    # K projection: to_k.0 is linear, to_k.1 is QK norm
    r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.attn\.to_k\.0\.(.*)$": r"transformer_blocks.\1.attn1.to_k.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.attn\.to_k\.1\.(.*)$": r"transformer_blocks.\1.attn1.norm_k.\2",
    # V projection
    r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.attn\.to_v\.0\.(.*)$": r"transformer_blocks.\1.attn1.to_v.\2",
    # Output projection
    r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.attn\.to_out\.0\.(.*)$": r"transformer_blocks.\1.attn1.to_out.\2",
    # AdaLN modulation for self-attention
    r"^net\.blocks\.block(\d+)\.blocks\.0\.adaLN_modulation\.(.*)$": r"transformer_blocks.\1.adaln_modulation_self_attn.\2",

    # Cross-attention (block index 1)
    # Q projection: to_q.0 is linear, to_q.1 is QK norm
    r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.attn\.to_q\.0\.(.*)$": r"transformer_blocks.\1.attn2.to_q.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.attn\.to_q\.1\.(.*)$": r"transformer_blocks.\1.attn2.norm_q.\2",
    # K projection: to_k.0 is linear, to_k.1 is QK norm
    r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.attn\.to_k\.0\.(.*)$": r"transformer_blocks.\1.attn2.to_k.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.attn\.to_k\.1\.(.*)$": r"transformer_blocks.\1.attn2.norm_k.\2",
    # V projection
    r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.attn\.to_v\.0\.(.*)$": r"transformer_blocks.\1.attn2.to_v.\2",
    # Output projection
    r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.attn\.to_out\.0\.(.*)$": r"transformer_blocks.\1.attn2.to_out.\2",
    # AdaLN modulation for cross-attention
    r"^net\.blocks\.block(\d+)\.blocks\.1\.adaLN_modulation\.(.*)$": r"transformer_blocks.\1.adaln_modulation_cross_attn.\2",

    # MLP (block index 2) - simpler naming: layer1, layer2 directly
    r"^net\.blocks\.block(\d+)\.blocks\.2\.block\.layer1\.(.*)$": r"transformer_blocks.\1.mlp.fc_in.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.2\.block\.layer2\.(.*)$": r"transformer_blocks.\1.mlp.fc_out.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.2\.adaLN_modulation\.(.*)$": r"transformer_blocks.\1.adaln_modulation_mlp.\2",

    # Final layer
    r"^net\.final_layer\.linear\.(.*)$": r"final_layer.proj_out.\1",
    r"^net\.final_layer\.adaLN_modulation\.(.*)$": r"final_layer.adaln_modulation.\1",
}

# Keys to skip (dynamically computed or training metadata)
SKIP_PATTERNS = [
    "net.pos_embedder.",  # RoPE computed dynamically
    "net.accum_",  # Training accumulation metadata
    "logvar.",  # Training-only logvar module (not used for inference)
]


def apply_mapping(key: str) -> str | None:
    """Apply parameter name mapping to convert from official to FastVideo format."""
    # Check if key should be skipped
    for pattern in SKIP_PATTERNS:
        if key.startswith(pattern):
            return None
    
    # Apply mapping patterns
    for pattern, replacement in PARAM_NAMES_MAPPING.items():
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key)
    
    # If no mapping found, return original key (will be reported)
    return key


def load_checkpoint(path: Path, mmap: bool = True) -> dict[str, torch.Tensor]:
    """Load checkpoint from .pt file.
    
    Args:
        path: Path to checkpoint file
        mmap: Use memory-mapped loading (PyTorch 2.1+) for reduced RAM usage
    """
    # Try memory-mapped loading first (PyTorch 2.1+)
    load_kwargs: dict = {"map_location": "cpu", "weights_only": False}
    
    if mmap:
        # Check if PyTorch version supports mmap (2.1+)
        try:
            version_str = torch.__version__.split("+")[0]  # Remove +cu118 suffix etc.
            major, minor = version_str.split(".")[:2]
            torch_version = (int(major), int(minor))
            if torch_version >= (2, 1):
                load_kwargs["mmap"] = True
                print("  Using memory-mapped loading for reduced RAM usage")
            else:
                print(f"  Note: PyTorch {torch.__version__} doesn't support mmap, using standard loading")
        except (ValueError, IndexError):
            print(f"  Warning: Could not parse PyTorch version {torch.__version__}, skipping mmap")
    
    checkpoint = torch.load(path, **load_kwargs)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "ema"):
            if key in checkpoint:
                return checkpoint[key]
        return checkpoint
    
    return checkpoint


def iterate_checkpoint_keys(path: Path) -> Iterator[str]:
    """Iterate over checkpoint keys without loading all tensors.
    
    This is useful for analysis when memory is limited.
    """
    # Load with weights_only=True to just get the structure
    # Unfortunately PyTorch doesn't have a great way to do this
    # So we load the full checkpoint but only keep keys
    checkpoint = torch.load(path, map_location="meta", weights_only=False)
    
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "ema"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    
    return checkpoint.keys()


def analyze_checkpoint(state_dict: dict[str, torch.Tensor]) -> None:
    """Print checkpoint structure analysis."""
    print("\n" + "=" * 80)
    print("CHECKPOINT ANALYSIS")
    print("=" * 80)
    
    # Count parameters
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\nTotal parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"Total keys: {len(state_dict)}")
    
    # Analyze key prefixes
    prefixes: dict[str, int] = {}
    for key in state_dict:
        parts = key.split(".")
        prefix = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    
    print("\nKey prefixes:")
    for prefix, count in sorted(prefixes.items()):
        print(f"  {prefix}: {count}")
    
    # Print first 100 keys with shapes
    print("\nFirst 100 keys with shapes:")
    for i, (key, value) in enumerate(state_dict.items()):
        if i >= 100:
            print(f"  ... and {len(state_dict) - 100} more keys")
            break
        print(f"  {key}: {list(value.shape)}")
    
    # Identify GEN3C-specific layers
    print("\nGEN3C-specific layers:")
    gen3c_patterns = [
        "augment_sigma",
        "x_embedder",  # Has more input channels than standard Cosmos
        "extra_pos_embedder",
    ]
    for key in state_dict:
        for pattern in gen3c_patterns:
            if pattern in key:
                print(f"  {key}: {list(state_dict[key].shape)}")
                break
    
    print("=" * 80 + "\n")


def convert_weights(
    state_dict: dict[str, torch.Tensor],
    verbose: bool = False,
    dtype: torch.dtype | None = None,
    memory_efficient: bool = True,
) -> tuple[OrderedDict[str, torch.Tensor], list[str], list[str]]:
    """Convert weights from official format to FastVideo format.
    
    Args:
        state_dict: Source state dict
        verbose: Print detailed conversion info
        dtype: Convert tensors to this dtype (e.g., torch.float16)
        memory_efficient: Delete source tensors after processing to free memory
    
    Returns:
        converted: Converted state dict
        unmapped: List of keys that weren't mapped (kept as-is)
        skipped: List of keys that were skipped
    """
    converted = OrderedDict()
    unmapped = []
    skipped = []
    
    # Get all keys first (so we can delete as we go)
    keys = list(state_dict.keys())
    total = len(keys)
    
    for i, key in enumerate(keys):
        value = state_dict[key]
        new_key = apply_mapping(key)
        
        if new_key is None:
            skipped.append(key)
            if verbose:
                print(f"  Skipped: {key}")
        elif new_key == key:
            # No mapping found, but not in skip list
            unmapped.append(key)
            if dtype is not None and value.is_floating_point():
                converted[key] = value.to(dtype).contiguous()
            else:
                converted[key] = value.contiguous() if memory_efficient else value
            if verbose:
                print(f"  Unmapped: {key}")
        else:
            if dtype is not None and value.is_floating_point():
                converted[new_key] = value.to(dtype).contiguous()
            else:
                converted[new_key] = value.contiguous() if memory_efficient else value
            if verbose:
                print(f"  {key} -> {new_key}")
        
        # Free memory by deleting processed tensor from source
        if memory_efficient:
            del state_dict[key]
            if i % 100 == 0:
                gc.collect()
        
        # Progress indicator
        if (i + 1) % 200 == 0 or i == total - 1:
            print(f"  Processed {i + 1}/{total} tensors...")
    
    # Final garbage collection
    if memory_efficient:
        gc.collect()
    
    return converted, unmapped, skipped


def write_component(
    output_dir: Path,
    name: str,
    weights: OrderedDict[str, torch.Tensor],
    config: dict | None = None,
) -> None:
    """Write component weights and config to output directory."""
    component_dir = output_dir / name
    component_dir.mkdir(parents=True, exist_ok=True)
    
    # Save weights
    output_file = component_dir / "model.safetensors"
    save_file(weights, str(output_file))
    print(f"Saved {name} weights to {output_file}")
    print(f"  {len(weights)} tensors, {sum(t.numel() for t in weights.values()):,} parameters")
    
    # Save config
    if config is not None:
        config_path = component_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        print(f"Saved {name} config to {config_path}")


def build_transformer_config() -> dict:
    """Build transformer config for Gen3CTransformer3DModel.
    
    Architecture based on checkpoint analysis:
    - hidden_size = 4096 (32 heads * 128 head_dim)
    - num_layers = 28
    - in_channels = 82 (16 VAE + 1 mask + 64 buffer + 1 padding)
    - MLP ratio = 4.0
    """
    return {
        "_class_name": "Gen3CTransformer3DModel",
        "in_channels": 16,  # Base VAE channels (full input computed at runtime)
        "out_channels": 16,
        "num_attention_heads": 32,  # 4096 / 128 = 32
        "attention_head_dim": 128,
        "num_layers": 28,
        "mlp_ratio": 4.0,
        "text_embed_dim": 1024,  # T5 embedding dim
        "adaln_lora_dim": 256,
        "use_adaln_lora": True,
        "add_augment_sigma_embedding": False,  # Not present in this checkpoint
        "frame_buffer_max": 2,  # 2 buffers for 3D cache
        "max_size": [128, 240, 240],  # Max T, H, W for positional embeddings
        "patch_size": [1, 2, 2],
        "rope_scale": [2.0, 1.0, 1.0],
        "extra_pos_embed_type": "learnable",
        "concat_padding_mask": True,
        "affine_emb_norm": True,
        "qk_norm": "rms_norm",
        "eps": 1e-6,
    }


def build_model_index() -> dict:
    """Build model_index.json for the converted model."""
    return {
        "_class_name": "Gen3CPipeline",
        "_diffusers_version": "0.33.0.dev0",
        "transformer": ["diffusers", "Gen3CTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLGen3CTokenizer"],
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "T5Tokenizer"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }


def download_checkpoint(
    repo_id: str,
    filename: str = "model.pt",
    token: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download checkpoint from HuggingFace Hub."""
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is required for --download. Install with: pip install huggingface_hub")
    
    print(f"Downloading {filename} from {repo_id}...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    print(f"Downloaded to {path}")
    return Path(path)


def resolve_model_dir(
    model_name_or_path: str,
    cache_dir: Path | None = None,
) -> Path:
    """Resolve a local model directory from path or HuggingFace repo id."""
    model_path = Path(model_name_or_path)
    if model_path.exists():
        return model_path

    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is required to download component source. "
            "Install with: pip install huggingface_hub")

    print(f"Downloading component source repo: {model_name_or_path}")
    downloaded = snapshot_download(
        repo_id=model_name_or_path,
        allow_patterns=[
            "model_index.json",
            "vae/*",
            "text_encoder/*",
            "tokenizer/*",
            "scheduler/*",
        ],
        local_dir=str(cache_dir) if cache_dir else None,
        local_dir_use_symlinks=False,
    )
    print(f"Component source downloaded to {downloaded}")
    return Path(downloaded)


def add_inference_components(
    source_dir: Path,
    output_dir: Path,
    link_components: bool = False,
) -> None:
    """Copy or symlink VAE/text encoder/tokenizer/scheduler into output dir."""
    required_components = ("vae", "text_encoder", "tokenizer", "scheduler")
    missing: list[str] = []

    for component in required_components:
        src = source_dir / component
        dst = output_dir / component

        if not src.exists():
            missing.append(component)
            continue

        if dst.exists():
            print(f"  Skipping {component}: already exists at {dst}")
            continue

        if link_components:
            dst.symlink_to(src.resolve(), target_is_directory=True)
            print(f"  Linked {component}: {dst} -> {src.resolve()}")
        else:
            shutil.copytree(src, dst, dirs_exist_ok=False)
            print(f"  Copied {component}: {src} -> {dst}")

    if missing:
        raise FileNotFoundError(
            f"Missing required components in source repo {source_dir}: {missing}"
        )


def patch_gen3c_vae_config(output_dir: Path) -> None:
    """Tag copied VAE config with the Gen3C tokenizer-backed class name."""
    vae_cfg_path = output_dir / "vae" / "config.json"
    if not vae_cfg_path.exists():
        return
    with vae_cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_class_name"] = "AutoencoderKLGen3CTokenizer"
    with vae_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"  Patched VAE config class to AutoencoderKLGen3CTokenizer: {vae_cfg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GEN3C checkpoint to FastVideo format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert from local checkpoint (recommended for limited RAM)
    python convert_gen3c_to_fastvideo.py --source ./official_weights/GEN3C-Cosmos-7B/model.pt --output ./gen3c_fastvideo

    # Download and convert from HuggingFace
    python convert_gen3c_to_fastvideo.py --download nvidia/GEN3C-Cosmos-7B --output ./gen3c_fastvideo

    # Analyze checkpoint structure only
    python convert_gen3c_to_fastvideo.py --source ./model.pt --analyze

    # Convert to fp16 for smaller output (and lower memory during save)
    python convert_gen3c_to_fastvideo.py --source ./model.pt --output ./gen3c_fastvideo --dtype fp16
        """,
    )
    
    parser.add_argument(
        "--source",
        type=str,
        help="Path to input .pt checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for converted weights",
    )
    parser.add_argument(
        "--download",
        type=str,
        help="HuggingFace repo ID to download checkpoint from (e.g., nvidia/GEN3C-Cosmos-7B)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="model.pt",
        help="Filename to download from HuggingFace (default: model.pt)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze checkpoint structure, don't convert",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed conversion info",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it exists",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Output dtype for weights (default: bf16, use fp16 for smaller output)",
    )
    parser.add_argument(
        "--no-mmap",
        action="store_true",
        help="Disable memory-mapped loading (not recommended, uses more RAM)",
    )
    parser.add_argument(
        "--components-source",
        type=str,
        default=None,
        help=(
            "Optional local path or HF repo id containing diffusers components "
            "(vae/text_encoder/tokenizer/scheduler) to copy into output. "
            "Example: nvidia/Cosmos-Predict2-2B-Video2World"
        ),
    )
    parser.add_argument(
        "--components-cache-dir",
        type=str,
        default=None,
        help="Optional cache/local directory used when downloading --components-source.",
    )
    parser.add_argument(
        "--link-components",
        action="store_true",
        help="Create symlinks for components instead of copying (for local sources).",
    )
    parser.add_argument(
        "--components-only",
        action="store_true",
        help=(
            "Skip transformer conversion and only add "
            "vae/text_encoder/tokenizer/scheduler into --output."
        ),
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.components_only:
        if args.output is None:
            raise ValueError("--output is required with --components-only")
        if args.components_source is None:
            raise ValueError(
                "--components-source is required with --components-only")
    else:
        if args.download and args.source:
            raise ValueError("Use either --download or --source, not both")
        if not args.download and not args.source:
            raise ValueError("Either --download or --source is required")
        if not args.analyze and not args.output:
            raise ValueError("--output is required when not using --analyze")

    if args.components_only:
        output_dir = Path(args.output)
        if not output_dir.exists():
            raise FileNotFoundError(
                f"--components-only expected existing output directory: {output_dir}"
            )
        component_source_dir = resolve_model_dir(
            args.components_source,
            cache_dir=Path(args.components_cache_dir)
            if args.components_cache_dir else None,
        )
        print("Adding inference components only...")
        add_inference_components(
            source_dir=component_source_dir,
            output_dir=output_dir,
            link_components=args.link_components,
        )
        patch_gen3c_vae_config(output_dir)
        print(f"Done. Components added to {output_dir}")
        return
    
    # Parse dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    target_dtype = dtype_map[args.dtype]
    
    # Get checkpoint path
    if args.download:
        checkpoint_path = download_checkpoint(
            repo_id=args.download,
            filename=args.filename,
            token=args.token,
        )
    else:
        checkpoint_path = Path(args.source)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint with memory-mapped loading
    print(f"Loading checkpoint from {checkpoint_path}...")
    use_mmap = not args.no_mmap
    state_dict = load_checkpoint(checkpoint_path, mmap=use_mmap)
    
    # Analyze if requested
    if args.analyze:
        analyze_checkpoint(state_dict)
        return
    
    # Analyze before conversion (quick summary only to save memory)
    print(f"\nCheckpoint has {len(state_dict)} tensors")
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    
    # Convert weights (memory-efficient mode)
    print(f"\nConverting weights to {args.dtype}...")
    converted, unmapped, skipped = convert_weights(
        state_dict, 
        verbose=args.verbose,
        dtype=target_dtype,
        memory_efficient=True,
    )
    
    # Force garbage collection after conversion
    del state_dict
    gc.collect()
    
    print(f"\nConversion summary:")
    print(f"  Converted: {len(converted)} tensors")
    print(f"  Skipped: {len(skipped)} tensors (dynamic/metadata)")
    print(f"  Unmapped: {len(unmapped)} tensors (kept original names)")
    
    if unmapped:
        print("\nWarning: The following keys were not mapped:")
        for key in unmapped[:20]:
            print(f"  {key}")
        if len(unmapped) > 20:
            print(f"  ... and {len(unmapped) - 20} more")
    
    # Write output
    output_dir = Path(args.output)
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"Output directory exists: {output_dir}. Use --force to overwrite.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write transformer weights
    print(f"\nSaving converted weights...")
    transformer_config = build_transformer_config()
    write_component(output_dir, "transformer", converted, transformer_config)
    
    # Free memory after saving
    del converted
    gc.collect()
    
    # Write model_index.json
    model_index = build_model_index()
    model_index_path = output_dir / "model_index.json"
    with model_index_path.open("w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2)
        f.write("\n")
    print(f"\nSaved model_index.json to {model_index_path}")

    if args.components_source is not None:
        print("\nAdding inference components...")
        component_source_dir = resolve_model_dir(
            args.components_source,
            cache_dir=Path(args.components_cache_dir)
            if args.components_cache_dir else None,
        )
        add_inference_components(
            source_dir=component_source_dir,
            output_dir=output_dir,
            link_components=args.link_components,
        )
        patch_gen3c_vae_config(output_dir)
    else:
        print("\nNote: Only transformer/model_index were written.")
        print("FastVideo local loading also requires: vae/, text_encoder/, tokenizer/, scheduler/.")
        print("Re-run with --components-source to add them automatically.")
    
    print(f"\nConversion complete! Output saved to {output_dir}")
    print("\nTo use with FastVideo:")
    print(f"  from fastvideo.models.dits.gen3c import Gen3CTransformer3DModel")
    print(f"  model = Gen3CTransformer3DModel.from_pretrained('{output_dir}/transformer')")


if __name__ == "__main__":
    main()
