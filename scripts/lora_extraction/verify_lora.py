#!/usr/bin/env python3
"""verify_lora.py
Verify that base_weight + delta (from adapter) ≈ fine_tuned_weight for sampled layers.
Usage: python verify_lora.py --base <base> --ft <ft> --adapter <adapter.safetensors> [--samples 10]

example: python verify_lora.py \
  --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --adapter artifacts/fastvideo_adapter.r16.safetensors \
  --samples 50 

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Iterable
import sys

import torch

# safetensors optional
_HAVE_SAFETENSORS = True
try:
    from safetensors.torch import load_file as safetensors_load  # type: ignore
except Exception:
    _HAVE_SAFETENSORS = False

# reuse loader from your extract_lora module
from extract_lora import load_transformer_state_dict_from_model  # type: ignore


def load_adapter(path: Path) -> Dict[str, Any]:
    p = str(path)
    if path.suffix == ".safetensors":
        if not _HAVE_SAFETENSORS:
            raise RuntimeError("safetensors not available in this environment")
        raw = safetensors_load(p)
        # safetensors returns numpy arrays; convert lazily later
        return raw
    else:
        return torch.load(p, map_location="cpu")


def as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x)


def group_adapter(adapter: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    grouped = {}
    for k, v in adapter.items():
        if k.endswith(".lora_A.weight"):
            base = k[: -len(".lora_A.weight")]
            grouped.setdefault(base, {})["A"] = v
        elif k.endswith(".lora_B.weight"):
            base = k[: -len(".lora_B.weight")]
            grouped.setdefault(base, {})["B"] = v
        elif k.endswith(".lora_rank"):
            base = k[: -len(".lora_rank")]
            grouped.setdefault(base, {})["rank"] = v
        elif k.endswith(".lora_alpha"):
            base = k[: -len(".lora_alpha")]
            grouped.setdefault(base, {})["alpha"] = v
    return grouped


def compute_delta_from_AB(A_raw, B_raw, expected_shape: Tuple[int, int]) -> Optional[torch.Tensor]:
    A = as_tensor(A_raw).to(torch.float32)
    B = as_tensor(B_raw).to(torch.float32)
    out_dim, in_dim = expected_shape

    # canonical shapes: A: (r, in), B: (out, r) => delta = B @ A
    if A.ndim == 2 and B.ndim == 2:
        if A.shape[1] == in_dim and B.shape[0] == out_dim and A.shape[0] == B.shape[1]:
            return B @ A
        # alternative common layout: A: (in, r), B: (r, out) => delta = (B.T @ A.T).T => B @ A
        if A.shape[0] == in_dim and B.shape[1] == out_dim and A.shape[1] == B.shape[0]:
            # transform to canonical (r, in) and (out, r)
            A_c = A.T  # (r, in)
            B_c = B.T  # (out, r)
            return B_c @ A_c
        # older style: sometimes adapter stores A as (in, r) and B as (out, r) (inconsistent) -> try B @ A.T
        if B.shape[0] == out_dim and A.shape[0] == in_dim:
            try:
                return B @ A.T
            except Exception:
                pass
    return None


def verify(
    base_sd: Dict[str, torch.Tensor],
    ft_sd: Dict[str, torch.Tensor],
    adapter: Dict[str, Any],
    num_samples: int = 10,
    tol: float = 1e-4,
) -> int:
    grouped = group_adapter(adapter)
    keys = list(grouped.keys())
    if not keys:
        print("No LoRA layers found in adapter.")
        return 2

    sample_keys = keys[:num_samples]
    results = []
    max_err = 0.0
    problematic = []

    for base_name in sample_keys:
        weight_key = base_name + ".weight"
        if weight_key not in base_sd or weight_key not in ft_sd:
            # skip if missing in either state dict
            continue

        A = grouped[base_name].get("A")
        B = grouped[base_name].get("B")
        if A is None or B is None:
            continue

        base_w = base_sd[weight_key].to(torch.float32)
        ft_w = ft_sd[weight_key].to(torch.float32)
        expected_shape = tuple(base_w.shape)

        delta = compute_delta_from_AB(A, B, expected_shape)
        if delta is None:
            problematic.append((base_name, "unable to compute delta (shape mismatch)"))
            continue

        recon = base_w + delta
        abs_err = (recon - ft_w).abs()
        max_a = float(abs_err.max().item())
        mean_a = float(abs_err.mean().item())
        mean_ft = float(ft_w.abs().mean().item()) + 1e-12
        rel = mean_a / mean_ft

        results.append((base_name, expected_shape, max_a, mean_a, rel))
        if max_a > max_err:
            max_err = max_a

    # Print summary
    if not results:
        print("No comparable layers were found between base, ft and adapter.")
        return 3

    print(f"Verified {len(results)} layers (sample). Max absolute error: {max_err:.6e}")
    for name, shape, max_a, mean_a, rel in results:
        print(f"{name}: shape={shape} max_err={max_a:.6e} mean_err={mean_a:.6e} rel_err={rel*100:.4f}%")

    if problematic:
        print("\nSome layers were skipped due to format/shape issues:")
        for p in problematic:
            print(f" - {p[0]}: {p[1]}")

    if max_err <= tol:
        print(f"\nSUCCESS: all checked layers below tolerance {tol:.0e}")
        return 0
    else:
        print(f"\nWARNING: max error {max_err:.6e} exceeds tolerance {tol:.0e}")
        return 1


def parse_args():
    p = argparse.ArgumentParser(description="Verify LoRA extraction by reconstructing FT weights.")
    p.add_argument("--base", required=True)
    p.add_argument("--ft", required=True)
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--tol", type=float, default=1e-4)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        adapter = load_adapter(args.adapter)
    except Exception as e:
        print(f"Failed to load adapter: {e}")
        sys.exit(2)

    base_sd = load_transformer_state_dict_from_model(args.base)
    ft_sd = load_transformer_state_dict_from_model(args.ft)
    rc = verify(base_sd, ft_sd, adapter, num_samples=args.samples, tol=args.tol)
    sys.exit(rc)


if __name__ == "__main__":
    main()
