#!/usr/bin/env python3
"""Scratch GB200 gate for the exact LTX-2 video self-attention shape.

Compares the production FA4 CuTe path against FA2 and forced SDPA
flash/cuDNN backends at (B, 4290, 32, 128) fwd+bwd, mirroring the text
attention gate's A/X/B-per-candidate protocol. Diagnostics feed a trainer
gate; these numbers are never MFU evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

TensorFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def make_sdpa(backend: SDPBackend | None) -> TensorFn:

    def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if backend is None:
            out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=None, dropout_p=0.0, is_causal=False)
        else:
            with sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=None, dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2)

    return _sdpa


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = actual_f - expected_f
    expected_norm = torch.linalg.vector_norm(expected_f)
    return {
        "max_abs": float(diff.abs().max()),
        "rmse": float(torch.sqrt(torch.mean(diff.square()))),
        "relative_l2": float(torch.linalg.vector_norm(diff) / expected_norm.clamp_min(1e-20)),
        "cosine": float(F.cosine_similarity(actual_f.flatten(), expected_f.flatten(), dim=0)),
    }


def run_once(
    fn: TensorFn,
    bases: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = (tensor.detach().clone().requires_grad_(True) for tensor in bases)
    out = fn(q, k, v)
    out.backward(grad_out)
    assert q.grad is not None and k.grad is not None and v.grad is not None
    return out.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


def time_segment(
    fn: TensorFn,
    bases: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_out: torch.Tensor,
    warmup: int,
    repeats: int,
) -> dict[str, float | list[float]]:
    q, k, v = (tensor.detach().clone().requires_grad_(True) for tensor in bases)
    for _ in range(warmup):
        q.grad = k.grad = v.grad = None
        fn(q, k, v).backward(grad_out)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        q.grad = k.grad = v.grad = None
        start.record()
        fn(q, k, v).backward(grad_out)
        end.record()
    torch.cuda.synchronize()
    values = [float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)]
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=31)
    args = parser.parse_args()

    torch.manual_seed(20260722 + args.batch)
    torch.cuda.manual_seed_all(20260722 + args.batch)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    shape = (args.batch, 11 * 15 * 26, 32, 128)
    bases = (
        torch.randn(shape, device=device, dtype=dtype),
        torch.randn(shape, device=device, dtype=dtype),
        torch.randn(shape, device=device, dtype=dtype),
    )
    grad_out = torch.randn(shape, device=device, dtype=dtype)

    from flash_attn import flash_attn_func as fa2_func
    from fastvideo.attention.utils.flash_attn_cute import flash_attn_func as current_fa4

    def fa2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return fa2_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, deterministic=False)

    functions: dict[str, TensorFn] = {
        "fa4_current": current_fa4,
        "fa2": fa2,
        "sdpa_flash": make_sdpa(SDPBackend.FLASH_ATTENTION),
        "sdpa_cudnn": make_sdpa(SDPBackend.CUDNN_ATTENTION),
    }

    header = {
        "batch": args.batch,
        "shape": shape,
        "dtype": str(dtype),
        "causal": False,
        "scale": 1.0 / math.sqrt(shape[-1]),
        "device": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
    }
    print("GATE_HEADER " + json.dumps(header, sort_keys=True), flush=True)

    reference = run_once(make_sdpa(None), bases, grad_out)
    parity: dict[str, object] = {}
    for name, fn in functions.items():
        try:
            result = run_once(fn, bases, grad_out)
            values = {
                label: metrics(actual, expected)
                for label, actual, expected in zip(("out", "dq", "dk", "dv"), result, reference, strict=True)
            }
            finite = all(torch.isfinite(tensor).all().item() for tensor in result)
            parity[name] = {"finite": bool(finite), "vs_sdpa_default": values}
            del result
        except Exception as exc:
            torch.cuda.synchronize()
            parity[name] = {"error": repr(exc)}
        print("GATE_PARITY " + json.dumps({"batch": args.batch, "backend": name, **parity[name]}, sort_keys=True),
              flush=True)

    timing: dict[str, object] = {}
    for candidate in ("fa2", "sdpa_flash", "sdpa_cudnn"):
        if "error" in parity.get(candidate, {}):
            continue
        try:
            a = time_segment(current_fa4, bases, grad_out, args.warmup, args.repeats)
            x = time_segment(functions[candidate], bases, grad_out, args.warmup, args.repeats)
            b = time_segment(current_fa4, bases, grad_out, args.warmup, args.repeats)
            midpoint_ms = (float(a["median_ms"]) + float(b["median_ms"])) / 2.0
            candidate_ms = float(x["median_ms"])
            row = {
                "batch": args.batch,
                "candidate": candidate,
                "a_current_median_ms": a["median_ms"],
                "x_candidate_median_ms": candidate_ms,
                "b_current_median_ms": b["median_ms"],
                "control_midpoint_ms": midpoint_ms,
                "control_drift_pct": 100.0 * abs(float(b["median_ms"]) - float(a["median_ms"])) / midpoint_ms,
                "delta_ms_per_call": candidate_ms - midpoint_ms,
                "delta_pct": 100.0 * (candidate_ms - midpoint_ms) / midpoint_ms,
                "projected_48_block_delta_ms": 48.0 * (candidate_ms - midpoint_ms),
            }
            timing[candidate] = row
        except Exception as exc:
            torch.cuda.synchronize()
            timing[candidate] = {"error": repr(exc)}
        print("GATE_TIMING " + json.dumps({"batch": args.batch, "candidate": candidate, **timing[candidate]},
              sort_keys=True), flush=True)

    print("GATE_RESULT " + json.dumps({"header": header, "parity": parity, "timing": timing}, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
