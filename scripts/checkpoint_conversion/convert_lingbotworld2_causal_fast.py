# SPDX-License-Identifier: Apache-2.0
"""Create a FastVideo-native LingBot World 2 causal-fast bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


LATENTS_MEAN = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
LATENTS_STD = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
]


def _write_json(path: Path, payload: dict) -> None:
    """Write deterministic JSON config files for a local bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _symlink_or_replace(source: Path, target: Path) -> None:
    """Create a relative symlink, replacing stale symlinks only."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        raise FileExistsError(f"Refusing to replace existing path: {target}")
    target.symlink_to(os.path.relpath(source, target.parent))


def convert_vae(source_dir: Path, output_dir: Path) -> None:
    """Link the exact LingBot World 2 Wan VAE checkpoint consumed by LingBotWorld2WanVAE."""
    _symlink_or_replace(source_dir / "Wan2.1_VAE.pth", output_dir / "vae" / "Wan2.1_VAE.pth")


def build_bundle(source_dir: Path, output_dir: Path) -> None:
    """Create configs and symlinks for FastVideo-native LingBot World 2 loading."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "model_index.json",
        {
            "_class_name": "LingBotWorld2CausalFastPipeline",
            "_diffusers_version": "0.30.0",
            "scheduler": ["diffusers", "FlowUniPCMultistepScheduler"],
            "text_encoder": ["transformers", "LingBotWorld2T5EncoderModel"],
            "tokenizer": ["transformers", "AutoTokenizer"],
            "transformer": ["diffusers", "LingBotWorld2CausalFastTransformer3DModel"],
            "vae": ["diffusers", "LingBotWorld2WanVAE"],
        },
    )
    _write_json(
        output_dir / "transformer" / "config.json",
        {
            "_class_name": "LingBotWorld2CausalFastTransformer3DModel",
            "_diffusers_version": "0.30.0",
            "model_type": "i2v",
            "patch_size": [1, 2, 2],
            "text_len": 512,
            "in_dim": 36,
            "dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 40,
            "num_layers": 40,
            "local_attn_size": 18,
            "sink_size": 6,
            "qk_norm": True,
            "cross_attn_norm": True,
            "eps": 1e-6,
        },
    )
    _write_json(
        output_dir / "text_encoder" / "config.json",
        {
            "architectures": ["LingBotWorld2T5EncoderModel"],
            "vocab_size": 256384,
            "dim": 4096,
            "dim_attn": 4096,
            "dim_ffn": 10240,
            "num_heads": 64,
            "num_layers": 24,
            "num_buckets": 32,
            "text_len": 512,
            "hidden_size": 4096,
            "dropout": 0.1,
        },
    )
    _write_json(
        output_dir / "vae" / "config.json",
        {
            "_class_name": "LingBotWorld2WanVAE",
            "base_dim": 96,
            "z_dim": 16,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "temperal_downsample": [False, True, True],
            "dropout": 0.0,
            "latents_mean": LATENTS_MEAN,
            "latents_std": LATENTS_STD,
        },
    )
    _write_json(
        output_dir / "scheduler" / "scheduler_config.json",
        {
            "_class_name": "FlowUniPCMultistepScheduler",
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "use_dynamic_shifting": False,
            "prediction_type": "flow_prediction",
            "solver_order": 2,
            "solver_type": "bh2",
            "final_sigmas_type": "zero",
        },
    )
    _write_json(output_dir / "scheduler" / "config.json", json.loads((output_dir / "scheduler" / "scheduler_config.json").read_text()))

    _symlink_or_replace(source_dir / "models_t5_umt5-xxl-enc-bf16.pth", output_dir / "text_encoder" / "pytorch_model.pt")
    _symlink_or_replace(source_dir / "google" / "umt5-xxl", output_dir / "tokenizer")
    for shard in sorted((source_dir / "transformers").glob("*.safetensors")):
        _symlink_or_replace(shard, output_dir / "transformer" / shard.name)
    index_path = source_dir / "transformers" / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        _symlink_or_replace(index_path, output_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json")
    convert_vae(source_dir, output_dir)


def main() -> None:
    """Parse CLI arguments and build the local converted bundle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    build_bundle(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
