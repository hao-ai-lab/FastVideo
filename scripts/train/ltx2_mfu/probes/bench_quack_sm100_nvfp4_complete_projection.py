#!/usr/bin/env python3
"""Count a complete QuACK NVFP4 projection step on the exact LTX-2 packs.

This is a scratch gate, not a FastVideo integration.  It keeps FP32 master
weights resident, derives BF16 working weights once outside the timed region,
then counts exact-current quantization, both weight-cache orientations,
forward/dgrad/wgrad, output postscales, forward bias, and bias gradient.
"""

import argparse
import gc
import importlib.metadata
import json
import statistics


# name, occurrences in 48 blocks, logical M, K, N
PACKS = (
    ("self_qkv", 48, 4290, 4096, 12288),
    ("video_dd", 144, 4290, 4096, 4096),
    ("text_kv", 48, 1024, 4096, 8192),
    ("ffn_up", 48, 4290, 4096, 16384),
    ("ffn_down", 48, 4290, 16384, 4096),
)
CONFIGS = (
    ((128, 64), (1, 1)),
    ((128, 128), (1, 1)),
    ((128, 192), (1, 1)),
    ((128, 256), (1, 1)),
    ((256, 64), (2, 1)),
    ((256, 128), (2, 1)),
    ((256, 192), (2, 1)),
    ((256, 256), (2, 1)),
)
# Best config for each exact shape in the paired QuACK 0.5 config 5/6/7 gate.
BEST_CONFIG = {
    (1024, 4096, 8192): 7,
    (1024, 8192, 4096): 5,
    (4096, 4320, 4096): 7,
    (4096, 4320, 16384): 7,
    (4290, 4096, 4096): 6,
    (4290, 4096, 12288): 6,
    (4290, 4096, 16384): 6,
    (4290, 12288, 4096): 6,
    (4290, 16384, 4096): 6,
    (8192, 1024, 4096): 7,
    (12288, 4320, 4096): 7,
    (16384, 4320, 4096): 7,
}
HISTORICAL_BF16_MS = 196.431
GATE_MS = 63.463
MARGIN_MS = 59.524


def emit(kind, **fields):
    print(json.dumps({"kind": kind, **fields}, sort_keys=True), flush=True)


def phase_shape(pack, phase):
    _, _, m, k, n = pack
    if phase == "fwd":
        return m, k, n
    if phase == "dgrad":
        return m, n, k
    if phase == "wgrad":
        return n, (m + 31) // 32 * 32, k
    raise ValueError(phase)


def self_check():
    assert sum(pack[1] for pack in PACKS) == 7 * 48
    assert sum(2 * m * k * n * count for _, count, m, k, n in PACKS) == 100031935807488
    shapes = {phase_shape(pack, phase) for pack in PACKS for phase in ("fwd", "dgrad", "wgrad")}
    assert shapes == set(BEST_CONFIG)
    assert all(shape[1] % 16 == 0 for shape in shapes)


def timed(torch, fn, warmup, samples, inner):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) / inner)
    return statistics.median(values)


def sampled_metrics(torch, actual, reference, limit=1 << 20):
    actual = actual.flatten()[:limit].float()
    reference = reference.flatten()[:limit].float()
    denom = reference.square().mean().sqrt().clamp_min(1e-12)
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "relative_rms": float((actual - reference).square().mean().sqrt() / denom),
        "cosine": float(torch.nn.functional.cosine_similarity(actual, reference, dim=0)),
        "norm_ratio": float(actual.norm() / reference.norm().clamp_min(1e-12)),
        "sampled_values": actual.numel(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", choices=[pack[0] for pack in PACKS], action="append")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--inner", type=int, default=1)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    self_check()
    selected = [pack for pack in PACKS if not args.pack or pack[0] in args.pack]
    if args.dry_run:
        emit(
            "contract",
            packs=selected,
            best_config=[{"shape": shape, "config": config} for shape, config in BEST_CONFIG.items()],
            precision={
                "master_weights": "fp32",
                "working_weights_activations_gradients": "bf16",
                "optimizer_moments": "fp32 (outside projection timing)",
                "gemm_accumulator": "fp32",
                "outputs": "bf16",
            },
            counted=(
                "x/dY exact amax and row/transpose NVFP4 quantization; BF16 transpose materialization; "
                "one weight amax plus row/column cache quantization; fwd+dgrad+wgrad; "
                "global postscale; forward bias; bias gradient"
            ),
        )
        return

    import cutlass
    import cutlass.cute as cute
    import torch

    # QuACK 0.5 targets the older location of these two CUTLASS DSL types.
    cute.core.ThrMma = cute.ThrMma
    cute.core.ThrCopy = cute.ThrCopy
    import quack.blockscaled_gemm_utils as blockscaled

    assert torch.cuda.get_device_capability() == (10, 0)
    torch.manual_seed(20260721)

    def exact_per_tensor_scale(x):
        return x.float().abs().nan_to_num().amax().clamp_min(1e-12) / (448.0 * 6.0)

    def quantize_and_pack(x, per_tensor_scale):
        q_u8, scale_2d, _ = blockscaled.to_nvfp4_compiled(x, 16, per_tensor_scale)
        return q_u8, blockscaled.pack_scale_2d_to_blocked_contig(scale_2d)

    def forward_epilogue(out, alpha, bias):
        out.mul_(alpha)
        return out.add_(bias)

    def scale_epilogue(out, alpha):
        return out.mul_(alpha)

    exact_per_tensor_scale = torch.compile(exact_per_tensor_scale, dynamic=True)
    forward_epilogue = torch.compile(forward_epilogue, dynamic=True)
    scale_epilogue = torch.compile(scale_epilogue, dynamic=True)

    def quantized_operand(x, per_tensor_scale):
        q_u8, scales = quantize_and_pack(x, per_tensor_scale)
        rows, packed_k = q_u8.shape
        operand = q_u8.view(1, rows, packed_k).permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
        return operand, scales

    def compile_gemm(shape, a, b, out, sfa, sfb):
        config = BEST_CONFIG[shape]
        tile, cluster = CONFIGS[config]
        run = blockscaled.compile_blockscaled_gemm_tvm_ffi(
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            16,
            cutlass.BFloat16,
            tile,
            cluster,
            a,
            b,
            out,
            sfa,
            sfb,
        )
        return run, config

    emit(
        "environment",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        torch_source=torch.__file__,
        cutlass_source=cutlass.__file__,
        quack=importlib.metadata.version("quack-kernels"),
        quack_source=blockscaled.__file__,
        source_commit="7f139e2b28610063d2f30526ba8f0ccae5d88944",
    )

    rows = []
    for pack in selected:
        name, count, m, k, n = pack
        padded_m = (m + 31) // 32 * 32

        # The masters remain resident and untouched.  Optimizer moments are
        # deliberately outside this projection-only gate.
        weight_master = torch.randn((n, k), device="cuda", dtype=torch.float32) * 0.02
        bias_master = torch.randn((n,), device="cuda", dtype=torch.float32) * 0.02
        weight = weight_master.to(torch.bfloat16)
        bias = bias_master.to(torch.bfloat16)
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        dy = torch.randn((m, n), device="cuda", dtype=torch.bfloat16) * 0.01

        fp4_fwd_store = torch.empty((1, m, n), device="cuda", dtype=torch.bfloat16)
        fp4_dx_store = torch.empty((1, m, k), device="cuda", dtype=torch.bfloat16)
        fp4_dw_store = torch.empty((1, n, k), device="cuda", dtype=torch.bfloat16)
        fp4_dbias = torch.empty((n,), device="cuda", dtype=torch.bfloat16)
        bf16_fwd = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        bf16_dx = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
        bf16_dw = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
        bf16_dbias = torch.empty((n,), device="cuda", dtype=torch.bfloat16)

        # Materialize one set only to compile the three exact-shape QMMs.
        weight_scale = exact_per_tensor_scale(weight)
        qw_row, sw_row = quantized_operand(weight, weight_scale)
        qw_col, sw_col = quantized_operand(weight.T.contiguous(), weight_scale)
        x_scale = exact_per_tensor_scale(x)
        qx, sx = quantized_operand(x, x_scale)
        dy_scale = exact_per_tensor_scale(dy)
        qdy, sdy = quantized_operand(dy, dy_scale)
        x_t = torch.nn.functional.pad(x.T, (0, padded_m - m)).contiguous()
        dy_t = torch.nn.functional.pad(dy.T, (0, padded_m - m)).contiguous()
        qx_t, sx_t = quantized_operand(x_t, x_scale)
        qdy_t, sdy_t = quantized_operand(dy_t, dy_scale)

        run_fwd, fwd_config = compile_gemm(
            phase_shape(pack, "fwd"), qx, qw_row, fp4_fwd_store.permute(1, 2, 0), sx, sw_row)
        run_dgrad, dgrad_config = compile_gemm(
            phase_shape(pack, "dgrad"), qdy, qw_col, fp4_dx_store.permute(1, 2, 0), sdy, sw_col)
        run_wgrad, wgrad_config = compile_gemm(
            phase_shape(pack, "wgrad"), qdy_t, qx_t, fp4_dw_store.permute(1, 2, 0), sdy_t, sx_t)
        del qw_row, sw_row, qw_col, sw_col, qx, sx, qdy, sdy, qx_t, sx_t, qdy_t, sdy_t, x_t, dy_t

        def bf16_gemms():
            torch.mm(x, weight.T, out=bf16_fwd)
            torch.mm(dy, weight, out=bf16_dx)
            torch.mm(dy.T, x, out=bf16_dw)

        def bf16_complete():
            bf16_gemms()
            bf16_fwd.add_(bias)
            torch.sum(dy, dim=0, out=bf16_dbias)

        def fp4_complete():
            # One exact scale reduction is reused for both orientations of
            # each underlying BF16 tensor; the quantized layouts differ.
            w_scale = exact_per_tensor_scale(weight)
            w_row, w_row_sf = quantized_operand(weight, w_scale)
            w_col, w_col_sf = quantized_operand(weight.T.contiguous(), w_scale)

            current_x_scale = exact_per_tensor_scale(x)
            x_row, x_row_sf = quantized_operand(x, current_x_scale)
            run_fwd(x_row, w_row, fp4_fwd_store.permute(1, 2, 0), x_row_sf, w_row_sf)
            forward_epilogue(fp4_fwd_store, current_x_scale * w_scale, bias)

            current_dy_scale = exact_per_tensor_scale(dy)
            dy_row, dy_row_sf = quantized_operand(dy, current_dy_scale)
            run_dgrad(dy_row, w_col, fp4_dx_store.permute(1, 2, 0), dy_row_sf, w_col_sf)
            scale_epilogue(fp4_dx_store, current_dy_scale * w_scale)

            x_transposed = torch.nn.functional.pad(x.T, (0, padded_m - m)).contiguous()
            dy_transposed = torch.nn.functional.pad(dy.T, (0, padded_m - m)).contiguous()
            x_col, x_col_sf = quantized_operand(x_transposed, current_x_scale)
            dy_col, dy_col_sf = quantized_operand(dy_transposed, current_dy_scale)
            run_wgrad(dy_col, x_col, fp4_dw_store.permute(1, 2, 0), dy_col_sf, x_col_sf)
            scale_epilogue(fp4_dw_store, current_dy_scale * current_x_scale)
            torch.sum(dy, dim=0, out=fp4_dbias)

        def x_both_orientations():
            shared_scale = exact_per_tensor_scale(x)
            quantized_operand(x, shared_scale)
            x_transposed = torch.nn.functional.pad(x.T, (0, padded_m - m)).contiguous()
            quantized_operand(x_transposed, shared_scale)

        if args.smoke_only:
            fp4_complete()
            bf16_complete()
            torch.cuda.synchronize()
            metrics = {
                "fwd": sampled_metrics(torch, fp4_fwd_store[0], bf16_fwd),
                "dgrad": sampled_metrics(torch, fp4_dx_store[0], bf16_dx),
                "wgrad": sampled_metrics(torch, fp4_dw_store[0], bf16_dw),
                "dbias": sampled_metrics(torch, fp4_dbias, bf16_dbias),
            }
            if not all(metric["finite"] for metric in metrics.values()):
                raise RuntimeError(f"non-finite complete projection output: {metrics}")
            emit(
                "smoke",
                name=name,
                configs={"fwd": fwd_config, "dgrad": dgrad_config, "wgrad": wgrad_config},
                metrics=metrics,
            )
            break

        bf16_a_ms = timed(torch, bf16_gemms, args.warmup, args.samples, args.inner)
        bf16_complete_ms = timed(torch, bf16_complete, args.warmup, args.samples, args.inner)
        torch.cuda.reset_peak_memory_stats()
        fp4_ms = timed(torch, fp4_complete, args.warmup, args.samples, args.inner)
        peak_bytes = torch.cuda.max_memory_allocated()
        # Text context is identical across all 48 blocks, so its row/transposed
        # packs can live across the step and are charged once, not 48 times.
        shared_x_ms = (timed(torch, x_both_orientations, args.warmup, args.samples, args.inner)
                       if name == "text_kv" else 0.0)
        bf16_b_ms = timed(torch, bf16_gemms, args.warmup, args.samples, args.inner)
        bf16_ms = (bf16_a_ms + bf16_b_ms) / 2

        fp4_complete()
        bf16_complete()
        torch.cuda.synchronize()
        metrics = {
            "fwd": sampled_metrics(torch, fp4_fwd_store[0], bf16_fwd),
            "dgrad": sampled_metrics(torch, fp4_dx_store[0], bf16_dx),
            "wgrad": sampled_metrics(torch, fp4_dw_store[0], bf16_dw),
            "dbias": sampled_metrics(torch, fp4_dbias, bf16_dbias),
        }
        if not all(metric["finite"] for metric in metrics.values()):
            raise RuntimeError(f"non-finite complete projection output: {metrics}")
        row = {
            "name": name,
            "count": count,
            "logical_shape": [m, k, n],
            "configs": {"fwd": fwd_config, "dgrad": dgrad_config, "wgrad": wgrad_config},
            "bf16_gemm_a_ms": bf16_a_ms,
            "bf16_gemm_b_ms": bf16_b_ms,
            "bf16_gemm_midpoint_ms": bf16_ms,
            "bf16_complete_ms": bf16_complete_ms,
            "fp4_complete_ms": fp4_ms,
            "shareable_x_quant_ms": shared_x_ms,
            "speedup_vs_bf16_gemms": bf16_ms / fp4_ms,
            "speedup_vs_bf16_complete": bf16_complete_ms / fp4_ms,
            "peak_allocated_gib": peak_bytes / 2**30,
            "metrics": metrics,
        }
        rows.append(row)
        emit("case", **row)

        del weight_master, bias_master, weight, bias, x, dy
        del fp4_fwd_store, fp4_dx_store, fp4_dw_store, fp4_dbias
        del bf16_fwd, bf16_dx, bf16_dw, bf16_dbias
        del run_fwd, run_dgrad, run_wgrad
        gc.collect()
        torch.cuda.empty_cache()

    if rows:
        bf16_gemm_ms = sum(row["bf16_gemm_midpoint_ms"] * row["count"] for row in rows)
        bf16_complete_ms = sum(row["bf16_complete_ms"] * row["count"] for row in rows)
        fp4_unamortized_ms = sum(row["fp4_complete_ms"] * row["count"] for row in rows)
        fp4_ms = sum(row["fp4_complete_ms"] * row["count"]
                     - row["shareable_x_quant_ms"] * (row["count"] - 1) for row in rows)
        speedup = bf16_gemm_ms / fp4_ms
        normalized_ms = HISTORICAL_BF16_MS / speedup
        emit(
            "aggregate",
            packs=[row["name"] for row in rows],
            bf16_gemm_ms=bf16_gemm_ms,
            bf16_complete_ms=bf16_complete_ms,
            fp4_unamortized_ms=fp4_unamortized_ms,
            fp4_complete_ms=fp4_ms,
            speedup_vs_bf16_gemms=speedup,
            speedup_vs_bf16_complete=bf16_complete_ms / fp4_ms,
            historical_bf16_ms=HISTORICAL_BF16_MS,
            ratio_normalized_complete_ms=normalized_ms,
            gate_ms=GATE_MS,
            margin_ms=MARGIN_MS,
            passes_break_even=normalized_ms <= GATE_MS,
            passes_margin=normalized_ms <= MARGIN_MS,
        )


if __name__ == "__main__":
    main()
