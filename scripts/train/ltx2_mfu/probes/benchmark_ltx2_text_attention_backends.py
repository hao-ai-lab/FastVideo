#!/usr/bin/env python3
"""Scratch GB200 gate for the exact LTX-2 overfit text-attention shape."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Callable

import torch
import torch.nn.functional as F


TensorFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    ).transpose(1, 2)


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
        "samples_ms": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=(1, 3), required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=31)
    args = parser.parse_args()

    torch.manual_seed(20260722 + args.batch)
    torch.cuda.manual_seed_all(20260722 + args.batch)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    q_shape = (args.batch, 11 * 15 * 26, 32, 128)
    kv_shape = (args.batch, 1024, 32, 128)
    bases = (
        torch.randn(q_shape, device=device, dtype=dtype),
        torch.randn(kv_shape, device=device, dtype=dtype),
        torch.randn(kv_shape, device=device, dtype=dtype),
    )
    grad_out = torch.randn(q_shape, device=device, dtype=dtype)

    from flash_attn import flash_attn_func as fa2_func
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
    from fastvideo.attention.utils.flash_attn_cute import flash_attn_func as current_fa4

    def fa2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return fa2_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, deterministic=False)

    def make_fa4_split(num_splits: int) -> TensorFn:
        class _FA4Split(torch.autograd.Function):

            @staticmethod
            def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                out, lse = _flash_attn_fwd(
                    q,
                    k,
                    v,
                    softmax_scale=None,
                    causal=False,
                    window_size_left=None,
                    window_size_right=None,
                    softcap=0.0,
                    num_splits=num_splits,
                    pack_gqa=None,
                    return_lse=True,
                )[:2]
                ctx.save_for_backward(q, k, v, out, lse)
                return out

            @staticmethod
            def backward(ctx, grad_out: torch.Tensor):
                q, k, v, out, lse = ctx.saved_tensors
                return _flash_attn_bwd(
                    q,
                    k,
                    v,
                    out,
                    grad_out,
                    lse,
                    softmax_scale=None,
                    causal=False,
                    softcap=0.0,
                    window_size_left=None,
                    window_size_right=None,
                    deterministic=False,
                )

        return _FA4Split.apply

    functions: dict[str, TensorFn] = {
        "fa4_current_splits1": current_fa4,
        "fa2": fa2,
        "sdpa": sdpa,
        **{f"fa4_splits{splits}": make_fa4_split(splits) for splits in (1, 2, 4, 8)},
    }

    header = {
        "batch": args.batch,
        "q_shape": q_shape,
        "k_shape": kv_shape,
        "v_shape": kv_shape,
        "dtype": str(dtype),
        "causal": False,
        "mask": None,
        "dropout_p": 0.0,
        "scale": 1.0 / math.sqrt(q_shape[-1]),
        "device": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    print("GATE_HEADER " + json.dumps(header, sort_keys=True), flush=True)

    reference = run_once(sdpa, bases, grad_out)
    parity: dict[str, object] = {}
    for name, fn in functions.items():
        try:
            result = run_once(fn, bases, grad_out)
            values = {
                label: metrics(actual, expected)
                for label, actual, expected in zip(("out", "dq", "dk", "dv"), result, reference, strict=True)
            }
            finite = all(torch.isfinite(tensor).all().item() for tensor in result)
            parity[name] = {"finite": bool(finite), "vs_sdpa": values}
            print("GATE_PARITY " + json.dumps({"batch": args.batch, "backend": name, **parity[name]}, sort_keys=True),
                  flush=True)
            del result
        except Exception as exc:
            torch.cuda.synchronize()
            parity[name] = {"error": repr(exc)}
            print("GATE_PARITY " + json.dumps({"batch": args.batch, "backend": name, **parity[name]}, sort_keys=True),
                  flush=True)

    candidates = ("fa2", "sdpa", "fa4_splits1", "fa4_splits2", "fa4_splits4", "fa4_splits8")
    timing: dict[str, object] = {}
    for candidate in candidates:
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
                "a_current": a,
                "x_candidate": x,
                "b_current": b,
                "control_midpoint_ms": midpoint_ms,
                "control_drift_pct": 100.0 * abs(float(b["median_ms"]) - float(a["median_ms"])) / midpoint_ms,
                "delta_ms_per_call": candidate_ms - midpoint_ms,
                "delta_pct": 100.0 * (candidate_ms - midpoint_ms) / midpoint_ms,
                "projected_48_block_delta_ms": 48.0 * (candidate_ms - midpoint_ms),
            }
            timing[candidate] = row
            print("GATE_TIMING " + json.dumps(row, sort_keys=True), flush=True)
        except Exception as exc:
            torch.cuda.synchronize()
            timing[candidate] = {"error": repr(exc)}
            print("GATE_TIMING " + json.dumps({"batch": args.batch, "candidate": candidate, "error": repr(exc)},
                  sort_keys=True), flush=True)

    print("GATE_RESULT " + json.dumps({"header": header, "parity": parity, "timing": timing}, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
