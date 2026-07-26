#!/usr/bin/env python3
"""Exact B=1 LTX-2 VideoOnly raw FlashInfer NVFP4 linear benchmark.

This intentionally imports only torch and flashinfer.  It compares:
  * native BF16 GEMMs in their training orientations;
  * prequantized NVFP4 mm_fp4 (primitive ceiling);
  * static-calibrated NVFP4 qmm, including the transpose/pad work needed by
    dgrad and wgrad with FlashInfer's row-major A / column-major B contract.

The four layer classes and multiplicities cover the 48 transformer blocks.
"""

import argparse
import gc
import json
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import flashinfer
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LayerShape:
    name: str
    count: int
    m: int
    k: int
    n: int


LAYERS = (
    LayerShape("video_dd", 288, 4290, 4096, 4096),
    LayerShape("text_dd", 96, 1024, 4096, 4096),
    LayerShape("ffn_up", 48, 4290, 4096, 16384),
    LayerShape("ffn_down", 48, 4290, 16384, 4096),
)
PHASES = ("fwd", "dgrad", "wgrad")
BACKENDS = ("auto", "cutlass", "cudnn")
FP4_MAX_TIMES_FP8_MAX = 6.0 * 448.0


def emit(kind: str, **values: object) -> None:
    print(json.dumps({"kind": kind, **values}, sort_keys=True), flush=True)


def calibrated_static_sf(tensor: torch.Tensor) -> torch.Tensor:
    # Calibration is outside all timed regions.  Production would retain one
    # such scalar per tensor role/layer and refresh it only deliberately.
    amax = tensor.abs().amax().float().clamp_min_(1e-12)
    return torch.as_tensor(FP4_MAX_TIMES_FP8_MAX, device=tensor.device) / amax


def quantize(tensor: torch.Tensor, global_sf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return flashinfer.nvfp4_quantize(
        tensor,
        global_sf,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False,
    )


def mm_fp4(
    lhs_fp4: torch.Tensor,
    rhs_fp4: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    return flashinfer.mm_fp4(
        lhs_fp4,
        rhs_fp4.T,
        lhs_scale,
        rhs_scale.T,
        alpha,
        torch.bfloat16,
        out,
        block_size=16,
        use_8x4_sf_layout=False,
        backend=backend,
        use_nvfp4=True,
    )


def qmm(
    lhs: torch.Tensor,
    rhs_rows: torch.Tensor,
    lhs_sf: torch.Tensor,
    rhs_sf: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    lhs_fp4, lhs_scale = quantize(lhs, lhs_sf)
    rhs_fp4, rhs_scale = quantize(rhs_rows, rhs_sf)
    return mm_fp4(lhs_fp4, rhs_fp4, lhs_scale, rhs_scale, alpha, out, backend)


def benchmark(fn: Callable[[], torch.Tensor], warmup: int, samples: int, inner: int) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        values = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(inner):
                fn()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) / inner)
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def numerical_error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    actual_f = actual.float()
    reference_f = reference.float()
    diff = actual_f - reference_f
    reference_rms = reference_f.square().mean().sqrt()
    return {
        "relative_rms": (diff.square().mean().sqrt() / reference_rms.clamp_min(1e-30)).item(),
        "mean_abs": diff.abs().mean().item(),
        "max_abs": diff.abs().amax().item(),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def make_case(
    layer: LayerShape,
    phase: str,
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[], tuple[torch.Tensor, torch.Tensor]],
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int],
    int,
]:
    """Return native BF16 op, FP4-orientation builder, bases, (m,k,n), FLOPs."""
    m, k, n = layer.m, layer.k, layer.n
    if phase == "fwd":
        x = torch.empty((m, k), device="cuda", dtype=torch.bfloat16).normal_(0.0, 1.0)
        weight = torch.empty((n, k), device="cuda", dtype=torch.bfloat16).normal_(0.0, 0.02)

        def bf16(out: torch.Tensor) -> torch.Tensor:
            return torch.mm(x, weight.T, out=out)

        def orient() -> tuple[torch.Tensor, torch.Tensor]:
            return x, weight

        return bf16, orient, x, weight, (m, k, n), 2 * m * k * n

    if phase == "dgrad":
        dy = torch.empty((m, n), device="cuda", dtype=torch.bfloat16).normal_(0.0, 0.01)
        weight = torch.empty((n, k), device="cuda", dtype=torch.bfloat16).normal_(0.0, 0.02)

        def bf16(out: torch.Tensor) -> torch.Tensor:
            return torch.mm(dy, weight, out=out)

        def orient() -> tuple[torch.Tensor, torch.Tensor]:
            return dy, weight.T.contiguous()

        return bf16, orient, dy, weight, (m, n, k), 2 * m * n * k

    if phase == "wgrad":
        dy = torch.empty((m, n), device="cuda", dtype=torch.bfloat16).normal_(0.0, 0.01)
        x = torch.empty((m, k), device="cuda", dtype=torch.bfloat16).normal_(0.0, 1.0)
        # FlashInfer's quantizer accepts K % 16 == 0, but its CUTLASS mm_fp4
        # path additionally requires logical K % 32 == 0.
        padded_m = (m + 31) // 32 * 32

        def bf16(out: torch.Tensor) -> torch.Tensor:
            return torch.mm(dy.T, x, out=out)

        def orient() -> tuple[torch.Tensor, torch.Tensor]:
            pad_rows = padded_m - m
            if pad_rows:
                dy_for_fp4 = F.pad(dy, (0, 0, 0, pad_rows))
                x_for_fp4 = F.pad(x, (0, 0, 0, pad_rows))
            else:
                dy_for_fp4 = dy
                x_for_fp4 = x
            return dy_for_fp4.T.contiguous(), x_for_fp4.T.contiguous()

        return bf16, orient, dy, x, (n, padded_m, k), 2 * m * n * k

    raise ValueError(phase)


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.set_device(0)
    torch.manual_seed(20260721)
    torch.backends.cuda.matmul.allow_tf32 = False
    emit(
        "environment",
        torch=torch.__version__,
        flashinfer=getattr(flashinfer, "__version__", "unknown"),
        gpu=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0),
        warmup=args.warmup,
        samples=args.samples,
        inner=args.inner,
        backends=args.backends,
    )

    aggregate: dict[str, dict[str, dict[str, float]]] = {
        "bf16": {phase: {"latency_ms": 0.0, "logical_flops": 0.0, "cases": 0.0} for phase in PHASES}
    }
    for backend in args.backends:
        aggregate[f"fp4_mm_{backend}"] = {
            phase: {"latency_ms": 0.0, "logical_flops": 0.0, "cases": 0.0} for phase in PHASES
        }
        aggregate[f"fp4_qmm_{backend}"] = {
            phase: {"latency_ms": 0.0, "logical_flops": 0.0, "cases": 0.0} for phase in PHASES
        }

    for phase in PHASES:
        for layer_index, layer in enumerate(LAYERS):
            gc.collect()
            torch.cuda.empty_cache()
            torch.manual_seed(20260721 + 100 * layer_index + PHASES.index(phase))
            bf16_op, orient, base_a, base_b, qshape, logical_flops = make_case(layer, phase)
            qm, qk, qn = qshape
            out_bf16 = torch.empty((qm, qn), device="cuda", dtype=torch.bfloat16)

            bf16_timing = benchmark(
                lambda: bf16_op(out_bf16),
                args.warmup,
                args.samples,
                args.inner,
            )
            aggregate["bf16"][phase]["latency_ms"] += bf16_timing["median_ms"] * layer.count
            aggregate["bf16"][phase]["logical_flops"] += logical_flops * layer.count
            aggregate["bf16"][phase]["cases"] += layer.count
            emit(
                "case",
                tier="bf16",
                phase=phase,
                layer=layer.name,
                count=layer.count,
                qmm_shape=qshape,
                logical_shape=(layer.m, layer.k, layer.n),
                logical_flops=logical_flops,
                **bf16_timing,
            )

            # Build each operand once for static calibration and primitive-only
            # timing.  Calibration and this orientation work are outside the
            # primitive tier, but the latter is repeated inside deployable qmm.
            lhs, rhs_rows = orient()
            lhs_sf = calibrated_static_sf(lhs)
            rhs_sf = calibrated_static_sf(rhs_rows)
            alpha = (1.0 / (lhs_sf * rhs_sf)).float()
            lhs_fp4, lhs_scale = quantize(lhs, lhs_sf)
            rhs_fp4, rhs_scale = quantize(rhs_rows, rhs_sf)
            torch.cuda.synchronize()

            for backend in args.backends:
                out_fp4 = torch.empty((qm, qn), device="cuda", dtype=torch.bfloat16)
                try:
                    mm_timing = benchmark(
                        lambda backend=backend: mm_fp4(
                            lhs_fp4,
                            rhs_fp4,
                            lhs_scale,
                            rhs_scale,
                            alpha,
                            out_fp4,
                            backend,
                        ),
                        args.warmup,
                        args.samples,
                        args.inner,
                    )
                    mm_fp4(lhs_fp4, rhs_fp4, lhs_scale, rhs_scale, alpha, out_fp4, backend)
                    torch.cuda.synchronize()
                    mm_error = numerical_error(out_fp4, out_bf16)
                    key = f"fp4_mm_{backend}"
                    aggregate[key][phase]["latency_ms"] += mm_timing["median_ms"] * layer.count
                    aggregate[key][phase]["logical_flops"] += logical_flops * layer.count
                    aggregate[key][phase]["cases"] += layer.count
                    emit(
                        "case",
                        tier="fp4_mm",
                        backend=backend,
                        phase=phase,
                        layer=layer.name,
                        count=layer.count,
                        qmm_shape=qshape,
                        logical_shape=(layer.m, layer.k, layer.n),
                        logical_flops=logical_flops,
                        error=mm_error,
                        **mm_timing,
                    )
                except Exception as exc:
                    torch.cuda.synchronize()
                    emit(
                        "failure",
                        tier="fp4_mm",
                        backend=backend,
                        phase=phase,
                        layer=layer.name,
                        qmm_shape=qshape,
                        exception=repr(exc),
                    )
                    continue

                try:
                    def deployable(backend: str = backend) -> torch.Tensor:
                        deploy_lhs, deploy_rhs = orient()
                        return qmm(deploy_lhs, deploy_rhs, lhs_sf, rhs_sf, alpha, out_fp4, backend)

                    qmm_timing = benchmark(
                        deployable,
                        args.warmup,
                        args.samples,
                        args.inner,
                    )
                    deployable()
                    torch.cuda.synchronize()
                    qmm_error = numerical_error(out_fp4, out_bf16)
                    key = f"fp4_qmm_{backend}"
                    aggregate[key][phase]["latency_ms"] += qmm_timing["median_ms"] * layer.count
                    aggregate[key][phase]["logical_flops"] += logical_flops * layer.count
                    aggregate[key][phase]["cases"] += layer.count
                    emit(
                        "case",
                        tier="fp4_qmm",
                        backend=backend,
                        phase=phase,
                        layer=layer.name,
                        count=layer.count,
                        qmm_shape=qshape,
                        logical_shape=(layer.m, layer.k, layer.n),
                        logical_flops=logical_flops,
                        error=qmm_error,
                        **qmm_timing,
                    )
                except Exception as exc:
                    torch.cuda.synchronize()
                    emit(
                        "failure",
                        tier="fp4_qmm",
                        backend=backend,
                        phase=phase,
                        layer=layer.name,
                        qmm_shape=qshape,
                        exception=repr(exc),
                    )

            del lhs, rhs_rows, lhs_fp4, lhs_scale, rhs_fp4, rhs_scale
            del base_a, base_b, out_bf16
            gc.collect()
            torch.cuda.empty_cache()

    for tier, phases in aggregate.items():
        complete = all(values["cases"] == sum(layer.count for layer in LAYERS) for values in phases.values())
        total_ms = sum(values["latency_ms"] for values in phases.values())
        total_flops = sum(values["logical_flops"] for values in phases.values())
        phase_results = {}
        for phase, values in phases.items():
            latency_ms = values["latency_ms"]
            phase_results[phase] = {
                **values,
                "effective_tflops": values["logical_flops"] / latency_ms / 1e9 if latency_ms else None,
            }
        emit(
            "aggregate",
            tier=tier,
            complete=complete,
            phases=phase_results,
            total_latency_ms=total_ms,
            total_logical_flops=total_flops,
            effective_tflops=total_flops / total_ms / 1e9 if total_ms else None,
        )

    emit("done", elapsed_wall_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
