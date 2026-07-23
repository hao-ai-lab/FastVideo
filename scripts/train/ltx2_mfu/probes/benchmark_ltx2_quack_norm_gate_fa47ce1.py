#!/usr/bin/env python3
"""Disposable GB200 gate for fused LTX-2 AdaLN and joint Q/K RMSNorm."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
import math
import statistics
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_BATCH = 1
DEFAULT_TOKENS = 11 * (480 // 32) * (832 // 32)  # 4290
DEFAULT_DIM = 32 * 128  # 4096
DEFAULT_BLOCKS = 48
DEFAULT_TRACE_NORM_MS = 27.723
DEFAULT_STEP_MS = 427.432493
EPS = 1e-6

_quack_rmsnorm: Callable[..., torch.Tensor] | None = None


def torch_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        normalized = F.rms_norm(x, (x.shape[-1], ), eps=EPS).to(x.dtype)
        return normalized * (1 + scale) + shift


def quack_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    if _quack_rmsnorm is None:
        raise RuntimeError("QuACK was not initialized")
    # QuACK's per-head mode supplies one affine row per sample. Moving B next
    # to D is a view; the kernel accepts dynamic leading strides, so this is
    # general for B>1 without copying the [B, T, D] activation.
    x_tbd = x.transpose(0, 1)
    weight = 1 + scale.squeeze(1)
    bias = shift.squeeze(1)
    return _quack_rmsnorm(x_tbd, weight=weight, bias=bias, eps=EPS).transpose(0, 1)


def torch_joint_qk(
    packed_qkv: torch.Tensor,
    packed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k = packed_qkv[:, :, :2, :].unbind(dim=2)
    q_weight, k_weight = packed_weight.unbind(dim=0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        q = F.rms_norm(q, (q.shape[-1], ), weight=q_weight, eps=EPS).to(q.dtype)
        k = F.rms_norm(k, (k.shape[-1], ), weight=k_weight, eps=EPS).to(k.dtype)
    return q, k


def quack_joint_qk(
    packed_qkv: torch.Tensor,
    packed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _quack_rmsnorm is None:
        raise RuntimeError("QuACK was not initialized")
    # Preserve the packed projection's stride: [B, T, 3, D] -> [B, T, 2, D]
    # is a view. QuACK treats the two slots as its per-head dimension.
    qk = _quack_rmsnorm(
        packed_qkv[:, :, :2, :],
        weight=packed_weight,
        eps=EPS,
    )
    return qk.unbind(dim=2)


def _outputs(value: torch.Tensor | Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return (value, ) if isinstance(value, torch.Tensor) else tuple(value)


def _clone_leaves(values: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().clone().requires_grad_(True) for value in values)


def _evaluate(
    function: Callable[..., Any],
    values: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    leaves = _clone_leaves(values)
    outputs = _outputs(function(*leaves))
    torch.autograd.backward(outputs, tuple(grad_outputs))
    gradients = tuple(value.grad.detach().clone() for value in leaves)
    return tuple(output.detach().clone() for output in outputs), gradients


def _comparison(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    reduction_gradient: bool,
) -> dict[str, Any]:
    if reduction_gradient:
        atol = max(
            0.1,
            float(2 * torch.finfo(torch.bfloat16).eps * expected.float().abs().max()),
        )
    else:
        atol = 0.1
    rtol = 1e-3
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    absolute = (actual.float() - expected.float()).abs()
    relative = absolute / expected.float().abs().clamp_min(1e-6)
    return {
        "name": name,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "atol": atol,
        "rtol": rtol,
        "max_abs": float(absolute.max()),
        "max_rel": float(relative.max()),
        "exact_fraction": float((actual == expected).float().mean()),
    }


def _check_parity(
    name: str,
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    values: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
) -> list[dict[str, Any]]:
    reference_outputs, reference_gradients = _evaluate(reference, values, grad_outputs)
    candidate_outputs, candidate_gradients = _evaluate(candidate, values, grad_outputs)
    if len(reference_outputs) != len(candidate_outputs):
        raise RuntimeError(f"{name}: output arity changed")
    rows = [
        _comparison(
            f"{name}.output.{index}",
            candidate_output,
            reference_output,
            reduction_gradient=False,
        )
        for index, (candidate_output, reference_output) in enumerate(
            zip(candidate_outputs, reference_outputs, strict=True)
        )
    ]
    for index, (candidate_gradient, reference_gradient) in enumerate(
        zip(candidate_gradients, reference_gradients, strict=True)
    ):
        rows.append(
            _comparison(
                f"{name}.gradient.{index}",
                candidate_gradient,
                reference_gradient,
                reduction_gradient=index > 0,
            )
        )
    return rows


def _iteration(
    function: Callable[..., Any],
    values: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
) -> None:
    for value in values:
        value.grad = None
    torch.autograd.backward(_outputs(function(*values)), tuple(grad_outputs))


def _time_ms(
    function: Callable[..., Any],
    values: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        _iteration(function, values, grad_outputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        _iteration(function, values, grad_outputs)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / repeats)


def _profile_once(
    function: Callable[..., Any],
    values: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
) -> dict[str, Any]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        _iteration(function, values, grad_outputs)
        torch.cuda.synchronize()
    kernels: dict[str, list[float]] = {}
    for event in profiler.events():
        if not str(getattr(event, "device_type", "")).endswith("CUDA"):
            continue
        row = kernels.setdefault(event.name, [0.0, 0.0])
        row[0] += 1
        row[1] += float(
            getattr(event, "self_device_time_total", None)
            or getattr(event, "self_cuda_time_total", None)
            or 0.0
        )
    top = sorted(kernels.items(), key=lambda item: item[1][1], reverse=True)[:8]
    return {
        "launches": int(sum(row[0] for row in kernels.values())),
        "summed_cuda_ms": sum(row[1] for row in kernels.values()) / 1000.0,
        "top_kernels": [{
            "name": kernel_name,
            "calls": int(row[0]),
            "cuda_ms": row[1] / 1000.0,
        } for kernel_name, row in top],
    }


def _benchmark(
    name: str,
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    values: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    *,
    calls_per_step: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    reference_values = _clone_leaves(values)
    candidate_values = _clone_leaves(values)
    reference_a = _time_ms(
        reference, reference_values, grad_outputs, warmup=warmup, repeats=repeats)
    candidate_ms = _time_ms(
        candidate, candidate_values, grad_outputs, warmup=warmup, repeats=repeats)
    reference_b = _time_ms(
        reference, reference_values, grad_outputs, warmup=0, repeats=repeats)
    midpoint = statistics.mean((reference_a, reference_b))
    delta = midpoint - candidate_ms
    return {
        "name": name,
        "calls_per_training_step": calls_per_step,
        "control_a_ms_per_call": reference_a,
        "candidate_ms_per_call": candidate_ms,
        "control_b_ms_per_call": reference_b,
        "control_midpoint_ms_per_call": midpoint,
        "control_drift_percent": abs(reference_b - reference_a) / midpoint * 100,
        "candidate_speedup_percent": delta / midpoint * 100,
        "projected_step_saving_ms": delta * calls_per_step,
        "profiles": {
            "control": _profile_once(reference, reference_values, grad_outputs),
            "candidate": _profile_once(candidate, candidate_values, grad_outputs),
        },
    }


def main() -> None:
    global _quack_rmsnorm

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--trace-norm-ms", type=float, default=DEFAULT_TRACE_NORM_MS)
    parser.add_argument("--step-ms", type=float, default=DEFAULT_STEP_MS)
    args = parser.parse_args()
    if min(args.batch_size, args.tokens, args.dim, args.blocks, args.warmup, args.repeats) < 1:
        parser.error("shape, block, warmup, and repeat values must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this gate requires CUDA")

    import quack
    from quack import rmsnorm

    _quack_rmsnorm = rmsnorm
    device = torch.device("cuda", int(torch.cuda.current_device()))
    generator = torch.Generator(device=device).manual_seed(20260722)
    shape = (args.batch_size, args.tokens, args.dim)
    x = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    scale = torch.randn(
        (args.batch_size, 1, args.dim),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) * 0.1
    shift = torch.randn(
        (args.batch_size, 1, args.dim),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) * 0.1
    adaln_grad = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)

    packed_qkv = torch.randn(
        (args.batch_size, args.tokens, 3, args.dim),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    packed_weight = torch.randn(
        (2, args.dim),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    q_grad = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    k_grad = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)

    parity = []
    parity.extend(
        _check_parity(
            "adaln",
            torch_adaln,
            quack_adaln,
            (x, scale, shift),
            (adaln_grad, ),
        )
    )
    parity.extend(
        _check_parity(
            "joint_qk",
            torch_joint_qk,
            quack_joint_qk,
            (packed_qkv, packed_weight),
            (q_grad, k_grad),
        )
    )

    compiled_torch_adaln = torch.compile(torch_adaln, fullgraph=True, dynamic=False)
    compiled_quack_adaln = torch.compile(quack_adaln, fullgraph=True, dynamic=False)
    compiled_torch_joint_qk = torch.compile(torch_joint_qk, fullgraph=True, dynamic=False)
    compiled_quack_joint_qk = torch.compile(quack_joint_qk, fullgraph=True, dynamic=False)

    benchmarks = [
        _benchmark(
            "adaln",
            compiled_torch_adaln,
            compiled_quack_adaln,
            (x, scale, shift),
            (adaln_grad, ),
            calls_per_step=2 * args.blocks,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        _benchmark(
            "joint_qk",
            compiled_torch_joint_qk,
            compiled_quack_joint_qk,
            (packed_qkv, packed_weight),
            (q_grad, k_grad),
            calls_per_step=args.blocks,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    ]
    projected_raw = sum(row["projected_step_saving_ms"] for row in benchmarks)
    projected_capped = min(max(projected_raw, 0.0), args.trace_norm_ms)
    payload = {
        "environment": {
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "quack": getattr(quack, "__version__", "unknown"),
        },
        "shape_contract": {
            "video_latent": [args.batch_size, 128, 11, 15, 26],
            "video_tokens": list(shape),
            "adaln_scale_shift": [args.batch_size, 1, args.dim],
            "packed_qkv": [args.batch_size, args.tokens, 3 * args.dim],
            "joint_qk_view": [args.batch_size, args.tokens, 2, args.dim],
            "joint_qk_weight": [2, args.dim],
            "blocks": args.blocks,
        },
        "parity": parity,
        "benchmarks": benchmarks,
        "projection": {
            "trace_norm_adaln_hard_ceiling_ms": args.trace_norm_ms,
            "raw_candidate_saving_ms": projected_raw,
            "capped_candidate_saving_ms": projected_capped,
            "baseline_step_ms": args.step_ms,
            "projected_step_ms": args.step_ms - projected_capped,
            "projected_step_speedup_percent": projected_capped / args.step_ms * 100,
            "material_gate_ms": 2.0,
            "passes_material_gate": projected_capped >= 2.0,
        },
    }
    if not math.isfinite(projected_raw):
        raise RuntimeError("non-finite timing projection")
    print("LTX2_QUACK_NORM_GATE " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
