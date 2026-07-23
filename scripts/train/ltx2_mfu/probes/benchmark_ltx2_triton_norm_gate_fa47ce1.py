#!/usr/bin/env python3
"""Direct Triton training gate for LTX-2 RMSNorm/AdaLN hot paths."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
import math
import statistics
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


BATCH = 1
TOKENS = 11 * (480 // 32) * (832 // 32)  # 4290
DIM = 32 * 128  # 4096
BLOCKS = 48
EPS = 1e-6
TRACE_NORM_MS = 27.723
STEP_MS = 427.432493


@triton.jit
def _rms_affine_fwd(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    rstd_ptr,
    stride_x_m,
    stride_x_h,
    stride_x_n,
    eps,
    H: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ADD_ONE: tl.constexpr,
):
    row = tl.program_id(0)
    m = row // H
    h = row - m * H
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(
        x_ptr + m * stride_x_m + h * stride_x_h + cols * stride_x_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    weight = tl.load(weight_ptr + h * N + cols, mask=mask, other=0.0).to(tl.float32)
    if ADD_ONE:
        weight += 1.0
    out = x * rstd * weight
    if HAS_BIAS:
        bias = tl.load(bias_ptr + h * N + cols, mask=mask, other=0.0).to(tl.float32)
        out += bias
    tl.store(out_ptr + row * N + cols, out, mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _rms_affine_dx(
    x_ptr,
    weight_ptr,
    dout_ptr,
    rstd_ptr,
    dx_ptr,
    stride_x_m,
    stride_x_h,
    stride_x_n,
    H: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ADD_ONE: tl.constexpr,
):
    row = tl.program_id(0)
    m = row // H
    h = row - m * H
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(
        x_ptr + m * stride_x_m + h * stride_x_h + cols * stride_x_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    dout = tl.load(dout_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + h * N + cols, mask=mask, other=0.0).to(tl.float32)
    if ADD_ONE:
        weight += 1.0
    rstd = tl.load(rstd_ptr + row).to(tl.float32)
    x_hat = x * rstd
    weight_dout = weight * dout
    correction = tl.sum(x_hat * weight_dout, axis=0) / N
    dx = (weight_dout - x_hat * correction) * rstd
    tl.store(dx_ptr + row * N + cols, dx, mask=mask)


@triton.jit
def _rms_affine_param_grad(
    x_ptr,
    dout_ptr,
    rstd_ptr,
    dweight_ptr,
    dbias_ptr,
    M,
    stride_x_m,
    stride_x_h,
    stride_x_n,
    H: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    h = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    dweight = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    dbias = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    for row_start in tl.range(0, M, BLOCK_M):
        rows = row_start + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(
            x_ptr
            + rows[:, None] * stride_x_m
            + h * stride_x_h
            + cols[None, :] * stride_x_n,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dout = tl.load(
            dout_ptr + (rows[:, None] * H + h) * N + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        rstd = tl.load(
            rstd_ptr + rows * H + h,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        dweight += tl.sum(dout * x * rstd[:, None], axis=0)
        if HAS_BIAS:
            dbias += tl.sum(dout, axis=0)
    tl.store(dweight_ptr + h * N + cols, dweight, mask=col_mask)
    if HAS_BIAS:
        tl.store(dbias_ptr + h * N + cols, dbias, mask=col_mask)


class _FusedRMSAffine(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        add_one: bool,
    ) -> torch.Tensor:
        if x.ndim != 3 or weight.ndim != 2:
            raise ValueError(f"expected x[M,H,N], weight[H,N], got {x.shape}, {weight.shape}")
        m, h, n = x.shape
        if weight.shape != (h, n):
            raise ValueError(f"weight {weight.shape} does not match x {x.shape}")
        if bias is not None and bias.shape != weight.shape:
            raise ValueError(f"bias {bias.shape} does not match weight {weight.shape}")
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise TypeError("scratch kernel is specialized to BF16 activations and affine tensors")
        if bias is not None and bias.dtype != torch.bfloat16:
            raise TypeError("scratch kernel is specialized to BF16 bias")
        if x.stride(-1) != 1 or not weight.is_contiguous() or (bias is not None and not bias.is_contiguous()):
            raise ValueError("last input dimension and affine tensors must be contiguous")
        block_n = triton.next_power_of_2(n)
        if block_n > 65536:
            raise ValueError(f"unsupported hidden dimension: {n}")
        out = torch.empty((m, h, n), device=x.device, dtype=x.dtype)
        rstd = torch.empty((m, h), device=x.device, dtype=torch.float32)
        _rms_affine_fwd[(m * h, )](
            x,
            weight,
            weight if bias is None else bias,
            out,
            rstd,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            EPS,
            H=h,
            N=n,
            BLOCK_N=block_n,
            HAS_BIAS=bias is not None,
            ADD_ONE=bool(add_one),
            num_warps=8,
        )
        ctx.save_for_backward(x, weight, rstd)
        ctx.has_bias = bias is not None
        ctx.add_one = bool(add_one)
        return out

    @staticmethod
    def backward(
        ctx: Any,
        dout: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None]:
        x, weight, rstd = ctx.saved_tensors
        m, h, n = x.shape
        dout = dout.contiguous()
        dx = torch.empty((m, h, n), device=x.device, dtype=x.dtype)
        dweight = torch.empty_like(weight)
        dbias = torch.empty_like(weight) if ctx.has_bias else None
        block_n = triton.next_power_of_2(n)
        _rms_affine_dx[(m * h, )](
            x,
            weight,
            dout,
            rstd,
            dx,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            H=h,
            N=n,
            BLOCK_N=block_n,
            ADD_ONE=ctx.add_one,
            num_warps=8,
        )
        grad_block_n = 32
        _rms_affine_param_grad[(h, triton.cdiv(n, grad_block_n))](
            x,
            dout,
            rstd,
            dweight,
            dweight if dbias is None else dbias,
            m,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            H=h,
            N=n,
            BLOCK_M=32,
            BLOCK_N=grad_block_n,
            HAS_BIAS=ctx.has_bias,
            num_warps=4,
        )
        return dx, dweight, dbias, None


def triton_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    x_tbd = x.transpose(0, 1)
    return _FusedRMSAffine.apply(
        x_tbd,
        scale.squeeze(1),
        shift.squeeze(1),
        True,
    ).transpose(0, 1)


def torch_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        normalized = F.rms_norm(x, (x.shape[-1], ), eps=EPS).to(x.dtype)
        return normalized * (1 + scale) + shift


def triton_joint_qk(
    packed_qkv: torch.Tensor,
    packed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, _, dim = packed_qkv.shape
    qk = packed_qkv[:, :, :2, :].reshape(batch * tokens, 2, dim)
    qk = _FusedRMSAffine.apply(qk, packed_weight, None, False)
    return qk.reshape(batch, tokens, 2, dim).unbind(dim=2)


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


def _as_tuple(value: torch.Tensor | Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return (value, ) if isinstance(value, torch.Tensor) else tuple(value)


def _clone_leaves(values: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().clone().requires_grad_(True) for value in values)


def _run(
    function: Callable[..., Any],
    values: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    leaves = _clone_leaves(values)
    outputs = _as_tuple(function(*leaves))
    torch.autograd.backward(outputs, tuple(grad_outputs))
    return (
        tuple(output.detach().clone() for output in outputs),
        tuple(value.grad.detach().clone() for value in leaves),
    )


def _compare(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    reduction_gradient: bool,
) -> dict[str, Any]:
    atol = 0.1
    if reduction_gradient:
        atol = max(
            atol,
            float(2 * torch.finfo(torch.bfloat16).eps * expected.float().abs().max()),
        )
    rtol = 1e-3
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    absolute = (actual.float() - expected.float()).abs()
    return {
        "name": name,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "atol": atol,
        "rtol": rtol,
        "max_abs": float(absolute.max()),
        "exact_fraction": float((actual == expected).float().mean()),
    }


def _parity(
    name: str,
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    values: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
) -> list[dict[str, Any]]:
    expected_outputs, expected_gradients = _run(reference, values, grad_outputs)
    actual_outputs, actual_gradients = _run(candidate, values, grad_outputs)
    rows = []
    for index, (actual, expected) in enumerate(zip(actual_outputs, expected_outputs, strict=True)):
        rows.append(_compare(f"{name}.output.{index}", actual, expected, reduction_gradient=False))
    for index, (actual, expected) in enumerate(zip(actual_gradients, expected_gradients, strict=True)):
        rows.append(_compare(f"{name}.gradient.{index}", actual, expected, reduction_gradient=index > 0))
    return rows


def _iteration(
    function: Callable[..., Any],
    values: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
) -> None:
    for value in values:
        value.grad = None
    torch.autograd.backward(_as_tuple(function(*values)), tuple(grad_outputs))


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


def _profile(
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
    return {
        "launches": int(sum(row[0] for row in kernels.values())),
        "summed_cuda_ms": sum(row[1] for row in kernels.values()) / 1000.0,
        "top_kernels": [{
            "name": name,
            "calls": int(row[0]),
            "cuda_ms": row[1] / 1000.0,
        } for name, row in sorted(kernels.items(), key=lambda item: item[1][1], reverse=True)[:8]],
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
    control_a = _time_ms(reference, reference_values, grad_outputs, warmup=warmup, repeats=repeats)
    candidate_ms = _time_ms(candidate, candidate_values, grad_outputs, warmup=warmup, repeats=repeats)
    control_b = _time_ms(reference, reference_values, grad_outputs, warmup=0, repeats=repeats)
    midpoint = statistics.mean((control_a, control_b))
    delta = midpoint - candidate_ms
    return {
        "name": name,
        "calls_per_step": calls_per_step,
        "control_a_ms_per_call": control_a,
        "candidate_ms_per_call": candidate_ms,
        "control_b_ms_per_call": control_b,
        "control_midpoint_ms_per_call": midpoint,
        "control_drift_percent": abs(control_a - control_b) / midpoint * 100,
        "candidate_speedup_percent": delta / midpoint * 100,
        "projected_step_saving_ms": delta * calls_per_step,
        "profiles": {
            "control": _profile(reference, reference_values, grad_outputs),
            "candidate": _profile(candidate, candidate_values, grad_outputs),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("--warmup and --repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(20260722)
    shape = (BATCH, TOKENS, DIM)
    x = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    scale = torch.randn((BATCH, 1, DIM), device=device, dtype=torch.bfloat16, generator=generator) * 0.1
    shift = torch.randn((BATCH, 1, DIM), device=device, dtype=torch.bfloat16, generator=generator) * 0.1
    adaln_grad = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    packed_qkv = torch.randn(
        (BATCH, TOKENS, 3, DIM),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    packed_weight = torch.randn((2, DIM), device=device, dtype=torch.bfloat16, generator=generator)
    q_grad = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    k_grad = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)

    parity = []
    parity.extend(_parity("adaln", torch_adaln, triton_adaln, (x, scale, shift), (adaln_grad, )))
    parity.extend(
        _parity(
            "joint_qk",
            torch_joint_qk,
            triton_joint_qk,
            (packed_qkv, packed_weight),
            (q_grad, k_grad),
        )
    )

    compiled_adaln = torch.compile(torch_adaln, fullgraph=True, dynamic=False)
    compiled_joint_qk = torch.compile(torch_joint_qk, fullgraph=True, dynamic=False)
    benchmarks = [
        _benchmark(
            "adaln",
            compiled_adaln,
            triton_adaln,
            (x, scale, shift),
            (adaln_grad, ),
            calls_per_step=2 * BLOCKS,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        _benchmark(
            "joint_qk",
            compiled_joint_qk,
            triton_joint_qk,
            (packed_qkv, packed_weight),
            (q_grad, k_grad),
            calls_per_step=BLOCKS,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    ]
    raw_saving = sum(row["projected_step_saving_ms"] for row in benchmarks)
    capped_saving = min(max(raw_saving, 0.0), TRACE_NORM_MS)
    if not math.isfinite(raw_saving):
        raise RuntimeError("non-finite timing projection")
    payload = {
        "environment": {
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "triton": triton.__version__,
        },
        "shape_contract": {
            "latent": [BATCH, 128, 11, 15, 26],
            "hidden": list(shape),
            "adaln_scale_shift": [BATCH, 1, DIM],
            "packed_qkv_storage": [BATCH, TOKENS, 3 * DIM],
            "joint_qk_view": [BATCH, TOKENS, 2, DIM],
            "blocks": BLOCKS,
        },
        "accumulation": "FP32 reductions and arithmetic; BF16 outputs and returned gradients",
        "parity": parity,
        "benchmarks": benchmarks,
        "projection": {
            "raw_saving_ms_per_step": raw_saving,
            "trace_capped_saving_ms_per_step": capped_saving,
            "trace_family_ceiling_ms_per_step": TRACE_NORM_MS,
            "baseline_step_ms": STEP_MS,
            "projected_step_ms": STEP_MS - capped_saving,
            "projected_latency_reduction_percent": capped_saving / STEP_MS * 100,
            "material_threshold_ms": 2.0,
            "passes_material_gate": capped_saving >= 2.0,
        },
    }
    print("LTX2_TRITON_NORM_GATE " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
