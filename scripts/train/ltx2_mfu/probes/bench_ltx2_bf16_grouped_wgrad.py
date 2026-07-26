#!/usr/bin/env python3
"""Reject-fast GB200 gate for deferred/grouped LTX-2 BF16 weight gradients."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from typing import Callable

import torch


DTYPE = torch.bfloat16
MASTER_DTYPE = torch.float32
BLOCKS = 48
VIDEO_TOKENS = 4290
TEXT_TOKENS = 1024
HIDDEN = 4096
FFN = 16384
RTOL = 1.6e-2
ATOL = 1.0e-2


@dataclass(frozen=True)
class Case:
    name: str
    tokens: int
    in_features: int
    out_features: int
    roles_per_block: int = 1


CASES = (
    Case("self_qkv", VIDEO_TOKENS, HIDDEN, 3 * HIDDEN),
    # Packed LTX has three separate 4096 -> 4096 video-token roles:
    # self-attention output, text-cross query, and text-cross output.
    Case("video_4096", VIDEO_TOKENS, HIDDEN, HIDDEN, roles_per_block=3),
    Case("text_kv", TEXT_TOKENS, HIDDEN, 2 * HIDDEN),
    Case("ffn_up", VIDEO_TOKENS, HIDDEN, FFN),
    Case("ffn_down", VIDEO_TOKENS, FFN, HIDDEN),
)


def emit(kind: str, **payload: object) -> None:
    print(json.dumps({"kind": kind, **payload}, sort_keys=True), flush=True)


def elapsed_ms(fn: Callable[[], None], inner: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(inner):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / inner


def paired_times(
    serial: Callable[[], None],
    prepacked: Callable[[], None],
    staged: Callable[[], None],
    staged_scatter: Callable[[], None],
    *,
    warmup: int,
    samples: int,
    inner: int,
) -> dict[str, object]:
    methods = (serial, prepacked, staged, staged_scatter)
    for _ in range(warmup):
        for method in methods:
            method()
    torch.cuda.synchronize()

    series = {name: [] for name in ("serial_a", "prepacked", "staged", "staged_scatter", "serial_b")}
    for _ in range(samples):
        series["serial_a"].append(elapsed_ms(serial, inner))
        series["prepacked"].append(elapsed_ms(prepacked, inner))
        series["staged"].append(elapsed_ms(staged, inner))
        series["staged_scatter"].append(elapsed_ms(staged_scatter, inner))
        series["serial_b"].append(elapsed_ms(serial, inner))

    medians = {name: statistics.median(values) for name, values in series.items()}
    paired_serial_midpoint = [
        (serial_a + serial_b) / 2
        for serial_a, serial_b in zip(series["serial_a"], series["serial_b"], strict=True)
    ]
    paired_savings = {
        mode: [midpoint - candidate for midpoint, candidate in zip(paired_serial_midpoint, series[mode], strict=True)]
        for mode in ("prepacked", "staged", "staged_scatter")
    }
    medians["serial_midpoint"] = statistics.median(paired_serial_midpoint)
    medians["serial_control_drift_percent"] = (
        abs(medians["serial_b"] - medians["serial_a"]) / medians["serial_midpoint"] * 100
    )
    return {
        "medians_ms": medians,
        "samples_ms": series,
        "paired_serial_midpoint_samples_ms": paired_serial_midpoint,
        "paired_savings_median_ms": {
            mode: statistics.median(values) for mode, values in paired_savings.items()
        },
        "paired_savings_samples_ms": paired_savings,
    }


def parity_metrics(
    x_sources: list[torch.Tensor],
    dy_sources: list[torch.Tensor],
    packed_dw: torch.Tensor,
    serial_dw: torch.Tensor,
) -> dict[str, object]:
    max_abs = 0.0
    error_sq = 0.0
    reference_sq = 0.0
    close = True
    sample_sha256 = hashlib.sha256()
    for batch_index, (x, dy) in enumerate(zip(x_sources, dy_sources, strict=True)):
        torch.mm(dy.transpose(0, 1), x, out=serial_dw)
        # Chunking keeps the parity probe well below the FFN gradient's size.
        for row in range(0, serial_dw.shape[0], 256):
            ref = serial_dw[row:row + 256].float()
            candidate = packed_dw[batch_index, row:row + 256].float()
            error = candidate - ref
            max_abs = max(max_abs, float(error.abs().max()))
            error_sq += float(torch.sum(error * error, dtype=torch.float64))
            reference_sq += float(torch.sum(ref * ref, dtype=torch.float64))
            close = close and bool(torch.all(error.abs() <= ATOL + RTOL * ref.abs()))
        sample = packed_dw[batch_index, ::max(1, packed_dw.shape[1] // 7),
                           ::max(1, packed_dw.shape[2] // 7)]
        sample_sha256.update(sample.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    relative_l2 = math.sqrt(error_sq / reference_sq) if reference_sq else 0.0
    if not close:
        raise RuntimeError(
            f"grouped wgrad parity failed: max_abs={max_abs} relative_l2={relative_l2} "
            f"rtol={RTOL} atol={ATOL}"
        )
    return {
        "allclose": close,
        "rtol": RTOL,
        "atol": ATOL,
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "sample_sha256": sample_sha256.hexdigest(),
    }


def phase_context(case: Case, group: int, warmup: int, samples: int, inner: int) -> dict[str, float]:
    x = torch.empty((case.tokens, case.in_features), device="cuda", dtype=DTYPE).normal_(std=0.02)
    dy = torch.empty((case.tokens, case.out_features), device="cuda", dtype=DTYPE).normal_(std=0.02)
    weight = torch.empty((case.out_features, case.in_features), device="cuda", dtype=DTYPE).normal_(std=0.02)
    fwd_out = torch.empty((case.tokens, case.out_features), device="cuda", dtype=DTYPE)
    dgrad_out = torch.empty((case.tokens, case.in_features), device="cuda", dtype=DTYPE)

    def fwd() -> None:
        for _ in range(group):
            torch.mm(x, weight.transpose(0, 1), out=fwd_out)

    def dgrad() -> None:
        for _ in range(group):
            torch.mm(dy, weight, out=dgrad_out)

    for _ in range(warmup):
        fwd()
        dgrad()
    torch.cuda.synchronize()
    fwd_ms = statistics.median(elapsed_ms(fwd, inner) for _ in range(samples))
    dgrad_ms = statistics.median(elapsed_ms(dgrad, inner) for _ in range(samples))
    del x, dy, weight, fwd_out, dgrad_out
    return {"forward_ms": fwd_ms, "recompute_ms": fwd_ms, "dgrad_ms": dgrad_ms}


def bench_case(case: Case, group: int, args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(20260721 + group)
    x_sources = [
        torch.empty((case.tokens, case.in_features), device="cuda", dtype=DTYPE).normal_(std=0.02)
        for _ in range(group)
    ]
    dy_sources = [
        torch.empty((case.tokens, case.out_features), device="cuda", dtype=DTYPE).normal_(std=0.02)
        for _ in range(group)
    ]
    x_packed = torch.empty((group, case.tokens, case.in_features), device="cuda", dtype=DTYPE)
    dy_packed = torch.empty((group, case.tokens, case.out_features), device="cuda", dtype=DTYPE)
    packed_dw = torch.empty((group, case.out_features, case.in_features), device="cuda", dtype=DTYPE)
    serial_dw = torch.empty((case.out_features, case.in_features), device="cuda", dtype=DTYPE)
    scattered_dw = [torch.empty_like(serial_dw) for _ in range(group)]

    def pack() -> None:
        for index in range(group):
            x_packed[index].copy_(x_sources[index])
            dy_packed[index].copy_(dy_sources[index])

    def serial() -> None:
        for x, dy in zip(x_sources, dy_sources, strict=True):
            torch.mm(dy.transpose(0, 1), x, out=serial_dw)

    def prepacked() -> None:
        torch.bmm(dy_packed.transpose(1, 2), x_packed, out=packed_dw)

    def staged() -> None:
        pack()
        prepacked()

    def staged_scatter() -> None:
        staged()
        for index in range(group):
            scattered_dw[index].copy_(packed_dw[index])

    pack()
    prepacked()
    torch.cuda.synchronize()
    parity = parity_metrics(x_sources, dy_sources, packed_dw, serial_dw)
    timings = paired_times(
        serial,
        prepacked,
        staged,
        staged_scatter,
        warmup=args.warmup,
        samples=args.samples,
        inner=args.inner,
    )
    context = phase_context(case, group, args.warmup, args.samples, args.inner)
    groups_per_step = BLOCKS // group
    multiplier = groups_per_step * case.roles_per_block
    projections = {
        mode: timings["paired_savings_median_ms"][mode] * multiplier
        for mode in ("prepacked", "staged", "staged_scatter")
    }
    result = {
        "case": case.name,
        "logical_shape_mkn": [case.tokens, case.in_features, case.out_features],
        "group": group,
        "blocks": BLOCKS,
        "roles_per_block": case.roles_per_block,
        "groups_per_step": groups_per_step,
        "working_dtype": str(DTYPE),
        "gradient_dtype": str(packed_dw.dtype),
        "master_weight_dtype": str(MASTER_DTYPE),
        "parity": parity,
        "phase_context_ms_per_group": context,
        **timings,
        "projected_kernel_local_step_savings_ms": projections,
    }
    del x_sources, dy_sources, x_packed, dy_packed, packed_dw, serial_dw, scattered_dw
    gc.collect()
    torch.cuda.empty_cache()
    return result


def self_test() -> None:
    assert BLOCKS % 2 == 0 and BLOCKS % 4 == 0
    assert sum(case.roles_per_block for case in CASES) == 7
    assert DTYPE == torch.bfloat16 and MASTER_DTYPE == torch.float32
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--inner", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(group not in (2, 4) or BLOCKS % group for group in args.groups):
        parser.error("groups must be 2 or 4 and divide 48")
    if min(args.warmup, args.samples, args.inner) < 1:
        parser.error("warmup, samples, and inner must be positive")

    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    emit(
        "environment",
        gpu=device_name,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        blocks=BLOCKS,
        checkpoint_forward_multiplicity=2,
        optimizer_contract={
            "working_weights": str(DTYPE),
            "weight_gradients": str(DTYPE),
            "resident_master_weights": str(MASTER_DTYPE),
            "resident_moments": str(MASTER_DTYPE),
        },
        caveat=(
            "kernel-local gate only; deferred wgrad can reduce FSDP reduce-scatter overlap, "
            "so a passing result still requires an end-to-end trainer A/B"
        ),
    )
    results = []
    for group in args.groups:
        for case in CASES:
            result = bench_case(case, group, args)
            results.append(result)
            emit("case", **result)

        group_results = [result for result in results if result["group"] == group]
        aggregate = {
            mode: sum(result["projected_kernel_local_step_savings_ms"][mode] for result in group_results)
            for mode in ("prepacked", "staged", "staged_scatter")
        }
        emit(
            "aggregate",
            group=group,
            projected_kernel_local_step_savings_ms=aggregate,
            conservative_gate_mode="staged_scatter",
            threshold_ms=10.0,
            passes_10ms_gate=aggregate["staged_scatter"] >= 10.0,
            interpretation=(
                "prepacked is the arithmetic ceiling; staged includes copies into contiguous bmm inputs; "
                "staged_scatter additionally copies each result to a separate parameter-grad buffer"
            ),
        )


if __name__ == "__main__":
    main()
