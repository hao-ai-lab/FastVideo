#!/usr/bin/env python3
"""Prepare LingBot-World checkpoints into a FastVideo-compatible repo layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

UMT5_XXL_CONFIG = {
    "vocab_size": 250112,
    "d_model": 4096,
    "d_kv": 64,
    "d_ff": 10240,
    "num_layers": 24,
    "num_heads": 64,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "dropout_rate": 0.1,
    "layer_norm_epsilon": 1e-6,
    "feed_forward_proj": "gated-gelu",
    "is_encoder_decoder": True,
    "use_cache": True,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "text_len": 512,
}

WAN_VAE_CONFIG = {
    "_class_name": "AutoencoderKLWan",
    "_diffusers_version": "0.33.0",
}


def _safe_copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _ensure_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)


def _normalize_transformer_config(config_path: Path) -> None:
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    config["_class_name"] = config.get("_class_name", "WanModel")
    config["_diffusers_version"] = config.get("_diffusers_version", "0.33.0")
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _maybe_copy_shared_component(component: str, shared_root: Path | None,
                                 output_root: Path) -> bool:
    if shared_root is None:
        return False
    src = shared_root / component
    if not src.exists():
        return False
    _safe_copytree(src, output_root / component)
    return True


def convert(
    source_dir: Path,
    output_dir: Path,
    shared_components_dir: Path | None,
    pipeline_class_name: str,
    boundary_ratio: float,
) -> None:
    high_noise = source_dir / "high_noise_model"
    low_noise = source_dir / "low_noise_model"

    if not high_noise.exists() or not low_noise.exists():
        raise FileNotFoundError(
            "source_dir must contain both high_noise_model/ and low_noise_model/"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    _safe_copytree(high_noise, output_dir / "transformer")
    _safe_copytree(low_noise, output_dir / "transformer_2")
    _normalize_transformer_config(output_dir / "transformer" / "config.json")
    _normalize_transformer_config(output_dir / "transformer_2" / "config.json")

    copied_tokenizer = _maybe_copy_shared_component("tokenizer",
                                                    shared_components_dir,
                                                    output_dir)
    if not copied_tokenizer:
        tokenizer_src = source_dir / "google" / "umt5-xxl"
        if not tokenizer_src.exists():
            raise FileNotFoundError(
                "Missing tokenizer directory. Provide --shared-components-dir "
                "or ensure source_dir/google/umt5-xxl exists.")
        _safe_copytree(tokenizer_src, output_dir / "tokenizer")

    copied_text_encoder = _maybe_copy_shared_component("text_encoder",
                                                       shared_components_dir,
                                                       output_dir)
    if not copied_text_encoder:
        text_encoder_dir = output_dir / "text_encoder"
        text_encoder_dir.mkdir(parents=True, exist_ok=True)
        ckpt = source_dir / "models_t5_umt5-xxl-enc-bf16.pth"
        if not ckpt.exists():
            raise FileNotFoundError(
                "Missing text encoder checkpoint models_t5_umt5-xxl-enc-bf16.pth. "
                "Provide --shared-components-dir with a prepared text_encoder/ "
                "or place the checkpoint under source_dir.")
        shutil.copy2(ckpt, text_encoder_dir / ckpt.name)
        _ensure_json(text_encoder_dir / "config.json", UMT5_XXL_CONFIG)

    copied_vae = _maybe_copy_shared_component("vae", shared_components_dir,
                                              output_dir)
    if not copied_vae:
        vae_dir = output_dir / "vae"
        vae_dir.mkdir(parents=True, exist_ok=True)
        vae_ckpt = source_dir / "Wan2.1_VAE.pth"
        if not vae_ckpt.exists():
            raise FileNotFoundError(
                "Missing VAE checkpoint Wan2.1_VAE.pth. "
                "Provide --shared-components-dir with a prepared vae/ "
                "or place the checkpoint under source_dir.")
        shutil.copy2(vae_ckpt, vae_dir / vae_ckpt.name)
        _ensure_json(vae_dir / "config.json", WAN_VAE_CONFIG)

    model_index = {
        "_class_name": pipeline_class_name,
        "_diffusers_version": "0.33.0",
        "workload_type": "i2v",
        "boundary_ratio": boundary_ratio,
        "transformer": ["diffusers", "WanModel"],
        "transformer_2": ["diffusers", "WanModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
        "tokenizer": ["transformers", "AutoTokenizer"],
    }
    _ensure_json(output_dir / "model_index.json", model_index)

    print("Prepared FastVideo model repo at:", output_dir)
    print("Pipeline:", pipeline_class_name)
    print("Boundary ratio:", boundary_ratio)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LingBot-World layout to FastVideo-compatible layout."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Path to downloaded lingbot-world-base-cam directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for FastVideo-compatible model layout.",
    )
    parser.add_argument(
        "--shared-components-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing prepared text_encoder/, tokenizer/, "
            "and vae/ folders (for example from a Wan2.2 Diffusers repo)."
        ),
    )
    parser.add_argument(
        "--pipeline-class-name",
        type=str,
        default="WanCamImageToVideoPipeline",
        help="Pipeline class name written into model_index.json.",
    )
    parser.add_argument(
        "--boundary-ratio",
        type=float,
        default=0.9,
        help="boundary_ratio written into model_index.json for dual-transformer switching.",
    )
    args = parser.parse_args()

    convert(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        shared_components_dir=args.shared_components_dir,
        pipeline_class_name=args.pipeline_class_name,
        boundary_ratio=args.boundary_ratio,
    )


if __name__ == "__main__":
    main()
