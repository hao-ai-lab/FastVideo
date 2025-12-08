"""Verify merged LoRA model matches finetuned model numerically.

Usage:
    python verify_lora.py \
        --merged merged_model \
        --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

LOG = logging.getLogger(__name__)


def configure_logging(level: str = "INFO"):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    LOG.addHandler(handler)
    LOG.setLevel(level)


def load_transformer_weights(model_path: str | Path) -> dict:
    """Load transformer weights from model directory."""
    model_path = Path(model_path)
    
    if not model_path.exists() or not model_path.is_dir():
        from huggingface_hub import snapshot_download
        LOG.info(f"Downloading {model_path} from HuggingFace Hub...")
        model_path = Path(snapshot_download(
            repo_id=str(model_path),
            ignore_patterns=["*.onnx", "*.msgpack"]
        ))
    
    transformer_dir = model_path / "transformer"
    
    if not transformer_dir.exists():
        raise FileNotFoundError(f"Transformer directory not found: {transformer_dir}")
    
    weight_files = sorted(transformer_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files in {transformer_dir}")
    
    LOG.info(f"Loading {len(weight_files)} file(s) from {transformer_dir}")
    
    state_dict = {}
    for f in weight_files:
        if "custom" in f.name:
            continue
        state_dict.update(load_file(str(f)))
    
    return state_dict


def compare_models(merged_sd: dict, ft_sd: dict) -> dict:
    """Compare merged and finetuned model weights."""
    
    common_keys = set(merged_sd.keys()) & set(ft_sd.keys())
    merged_only = set(merged_sd.keys()) - set(ft_sd.keys())
    ft_only = set(ft_sd.keys()) - set(merged_sd.keys())
    
    LOG.info(f"Common keys: {len(common_keys)}")
    if merged_only:
        LOG.warning(f"Keys only in merged: {len(merged_only)}")
    if ft_only:
        LOG.warning(f"Keys only in finetuned: {len(ft_only)}")
    
    results = []
    for key in sorted(common_keys):
        merged_param = merged_sd[key]
        ft_param = ft_sd[key]
        
        if merged_param.shape != ft_param.shape:
            LOG.error(f"{key}: shape mismatch {merged_param.shape} vs {ft_param.shape}")
            continue
        
        diff = (merged_param.float() - ft_param.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        
        merged_norm = merged_param.float().norm().item()
        rel_mean = (mean_abs / merged_norm * 100) if merged_norm > 0 else 0
        
        results.append({
            "key": key,
            "shape": tuple(merged_param.shape),
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "rel_mean": rel_mean
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify merged LoRA model matches finetuned model")
    parser.add_argument("--merged", required=True, help="Merged model directory")
    parser.add_argument("--ft", required=True, help="Finetuned model ID or path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()
    
    configure_logging(args.log_level)
    
    LOG.info(f"Loading merged model: {args.merged}")
    merged_sd = load_transformer_weights(args.merged)
    LOG.info(f"Loaded {len(merged_sd)} parameters")
    
    LOG.info(f"Loading finetuned model: {args.ft}")
    ft_sd = load_transformer_weights(args.ft)
    LOG.info(f"Loaded {len(ft_sd)} parameters")
    
    LOG.info("Comparing models...")
    results = compare_models(merged_sd, ft_sd)
    
    results.sort(key=lambda x: x["max_abs"], reverse=True)
    
    LOG.info(f"\nTop 10 mismatches by max_abs_error:")
    for i, r in enumerate(results[:10], 1):
        LOG.info(f"{i:2d}. {r['key']}")
        LOG.info(f"    shape={r['shape']}, max_abs={r['max_abs']:.3e}, mean_abs={r['mean_abs']:.3e}, rel_mean={r['rel_mean']:.4f}%")
    
    overall_mean = sum(r["mean_abs"] for r in results) / len(results)
    overall_max = max(r["max_abs"] for r in results)
    
    LOG.info(f"\nOverall metrics:")
    LOG.info(f"  Layers compared: {len(results)}")
    LOG.info(f"  Mean(mean_abs): {overall_mean:.3e}")
    LOG.info(f"  Max(max_abs): {overall_max:.3e}")
    
    if overall_mean < 1e-4:
        LOG.info("\nVerification PASSED: Merge is numerically accurate")
    else:
        LOG.warning(f"\nVerification WARNING: Mean error {overall_mean:.3e} > 1e-4")


if __name__ == "__main__":
    main()
