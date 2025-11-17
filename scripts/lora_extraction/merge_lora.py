"""merge_lora.py
Merge LoRA adapter into base transformer weights and write merged transformer weights.
Usage:
  python merge_lora.py --base <base> --adapter <adapter.safetensors> --output <out_dir>

example : python merge_lora.py \
  --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --adapter lora_adapter_rank16.safetensors \
  --output ./merged_model

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import shutil

import torch

# safetensors optional
_HAVE_SAFETENSORS = True
try:
    from safetensors.torch import load_file as safetensors_load, save_file as safetensors_save  # type: ignore
except Exception:
    _HAVE_SAFETENSORS = False

from extract_lora import load_transformer_state_dict_from_model  # type: ignore


def load_adapter(path: Path) -> Dict[str, Any]:
    p = str(path)
    if path.suffix == ".safetensors":
        if not _HAVE_SAFETENSORS:
            raise RuntimeError("safetensors not available")
        return safetensors_load(p)
    else:
        return torch.load(p, map_location="cpu")


def group_adapter(adapter: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    grouped = {}
    for k, v in adapter.items():
        if k.endswith(".lora_A.weight"):
            grouped.setdefault(k[: -len(".lora_A.weight")], {})["A"] = v
        elif k.endswith(".lora_B.weight"):
            grouped.setdefault(k[: -len(".lora_B.weight")], {})["B"] = v
    return grouped


def compute_delta(A_raw, B_raw, expected_shape):
    A = torch.tensor(A_raw) if not isinstance(A_raw, torch.Tensor) else A_raw
    B = torch.tensor(B_raw) if not isinstance(B_raw, torch.Tensor) else B_raw
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    out_dim, in_dim = expected_shape

    if A.ndim == 2 and B.ndim == 2:
        # canonical: A (r, in), B (out, r)
        if A.shape[1] == in_dim and B.shape[0] == out_dim and A.shape[0] == B.shape[1]:
            return B @ A
        # alternative: A (in, r), B (r, out)
        if A.shape[0] == in_dim and B.shape[1] == out_dim and A.shape[1] == B.shape[0]:
            return B.T @ A.T
        # fallback: try B @ A.T
        try:
            return B @ A.T
        except Exception:
            pass
    raise ValueError("Unable to compute delta: unexpected A/B shapes")


def merge(base_sd: Dict[str, torch.Tensor], adapter: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    grouped = group_adapter(adapter)
    merged = dict(base_sd)
    merged_count = 0
    for base_name, parts in grouped.items():
        weight_key = base_name + ".weight"
        if weight_key not in base_sd:
            continue
        A = parts.get("A")
        B = parts.get("B")
        if A is None or B is None:
            continue
        base_w = base_sd[weight_key].to(torch.float32)
        delta = compute_delta(A, B, tuple(base_w.shape))
        merged[weight_key] = (base_w + delta).to(base_sd[weight_key].dtype)
        merged_count += 1

    print(f"Merged {merged_count} layers")
    return merged


def save_merged_transformer(merged_sd: Dict[str, torch.Tensor], out_dir: Path):
    transformer_dir = out_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    out_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    # convert tensors to CPU numpy/torch friendly dict
    to_save = {k: v.detach().cpu() for k, v in merged_sd.items()}
    if _HAVE_SAFETENSORS:
        safetensors_save(to_save, str(out_path))
    else:
        torch.save(to_save, str(out_path.with_suffix(".pt")))
    print(f"Saved merged transformer weights to: {out_path}")


def copy_base_artifacts(base_path: str, out_dir: Path):
    src = Path(base_path)
    if not src.exists() or not src.is_dir():
        return
    # copy model_index.json and transformer/config.json if present
    for name in ("model_index.json",):
        s = src / name
        if s.exists():
            shutil.copy2(s, out_dir / name)
    cfg = src / "transformer" / "config.json"
    if cfg.exists():
        dst_dir = out_dir / "transformer"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg, dst_dir / "config.json")


def parse_args():
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base transformer weights.")
    p.add_argument("--base", required=True, help="Base model id or local path")
    p.add_argument("--adapter", required=True, type=Path, help="Adapter file (.safetensors or .pt)")
    p.add_argument("--output", required=True, type=Path, help="Output directory for merged model")
    return p.parse_args()


def main():
    args = parse_args()

    adapter = load_adapter(args.adapter)
    base_sd = load_transformer_state_dict_from_model(args.base)
    merged_sd = merge(base_sd, adapter)

    args.output.mkdir(parents=True, exist_ok=True)
    save_merged_transformer(merged_sd, args.output)
    copy_base_artifacts(args.base, args.output)
    print("Merge complete. You can run inference against the merged model directory.")


if __name__ == "__main__":
    main()
