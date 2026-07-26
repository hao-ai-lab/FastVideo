#!/usr/bin/env python3
"""Gate QuACK's existing SM100 NVFP4 kernel on the exact LTX-2 packs.

The optimized total keeps forward and dgrad sequential, but batches each
same-shaped wgrad across all 48 blocks.  It is an arithmetic lower bound:
operands are prequantized and bias/global postscale/cache work is excluded.
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
GATE_MS = 63.463
MARGIN_MS = 59.524
BREAK_EVEN_SPEEDUP = 3.095
MARGIN_SPEEDUP = 3.3


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
    one_pass_flops = sum(2 * m * k * n * count for _, count, m, k, n in PACKS)
    assert one_pass_flops == 100031935807488
    assert len({phase_shape(pack, phase) for pack in PACKS for phase in ("fwd", "dgrad", "wgrad")}) == 12


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


def tensors(torch, m, k, n, batch):
    # Physical FP4 storage is K/2; permuting contiguous (L,M,K/2) makes K
    # unit-stride while retaining the logical batch mode expected by CuTe.
    a_store = torch.empty((batch, m, k // 2), device="cuda", dtype=torch.float4_e2m1fn_x2)
    b_store = torch.empty((batch, n, k // 2), device="cuda", dtype=torch.float4_e2m1fn_x2)
    a_store.view(torch.uint8).random_(0, 256)
    b_store.view(torch.uint8).random_(0, 256)
    a = a_store.permute(1, 2, 0)
    b = b_store.permute(1, 2, 0)
    out_store = torch.empty((batch, m, n), device="cuda", dtype=torch.bfloat16)
    out = out_store.permute(1, 2, 0)
    sf_k = k // 16
    sfa = torch.ones((batch, (m + 127) // 128, (sf_k + 3) // 4, 512),
                     device="cuda", dtype=torch.float8_e4m3fn)
    sfb = torch.ones((batch, (n + 127) // 128, (sf_k + 3) // 4, 512),
                     device="cuda", dtype=torch.float8_e4m3fn)
    return a, b, out, out_store, sfa, sfb


def bench_bf16_shape(torch, shape, args):
    m, k, n = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    latency = timed(torch, lambda: torch.mm(a, b.T, out=out), args.warmup,
                    args.samples, args.inner)
    emit("bf16", shape=shape, median_ms=latency)
    del a, b, out
    gc.collect()
    torch.cuda.empty_cache()
    return latency


def bench_shape(torch, cutlass, compile_gemm, shape, batch, configs, args):
    m, k, n = shape
    a, b, out, out_store, sfa, sfb = tensors(torch, m, k, n, batch)
    rows = []
    for index in configs:
        tile, cluster = CONFIGS[index]
        try:
            run = compile_gemm(
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
            latency = timed(torch, lambda: run(a, b, out, sfa, sfb), args.warmup,
                            args.samples, args.inner if batch == 1 else 1)
            finite = bool(torch.isfinite(out_store.flatten()[:1 << 20]).all())
            if not finite:
                raise RuntimeError("non-finite output sample")
            rows.append((latency, index))
            emit("case", shape=shape, batch=batch, config=index, tile=tile,
                 cluster=cluster, median_ms=latency, finite_sample=finite)
        except Exception as exc:
            torch.cuda.synchronize()
            emit("failure", shape=shape, batch=batch, config=index, exception=repr(exc))
    if not rows:
        raise RuntimeError(f"all configurations failed for shape={shape}, batch={batch}")
    latency, index = min(rows)
    emit("best", shape=shape, batch=batch, config=index, median_ms=latency,
         finite_sample=True)
    del a, b, out, out_store, sfa, sfb
    gc.collect()
    torch.cuda.empty_cache()
    return latency


def correctness_smoke(torch, cutlass, compile_gemm):
    """Prove packed FP4, BF16 output, and L>1 agree on a tiny exact case."""
    batch = 2
    m = n = k = 256
    a_store = torch.empty((batch, m, k // 2), device="cuda",
                          dtype=torch.float4_e2m1fn_x2)
    b_store = torch.empty((batch, n, k // 2), device="cuda",
                          dtype=torch.float4_e2m1fn_x2)
    # E2M1 code 0b0010 is exactly 1.0; each byte stores two values.
    a_store.view(torch.uint8).fill_(0x22)
    b_store.view(torch.uint8).fill_(0x22)
    a = a_store.permute(1, 2, 0)
    b = b_store.permute(1, 2, 0)
    out_store = torch.empty((batch, m, n), device="cuda", dtype=torch.bfloat16)
    out = out_store.permute(1, 2, 0)
    scales = torch.ones((batch, 2, 4, 512), device="cuda",
                       dtype=torch.float8_e4m3fn)
    run = compile_gemm(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        16,
        cutlass.BFloat16,
        (256, 128),
        (2, 1),
        a,
        b,
        out,
        scales,
        scales,
    )
    run(a, b, out, scales, scales)
    torch.cuda.synchronize()
    max_abs = float((out_store.float() - k).abs().max())
    if max_abs != 0.0:
        raise RuntimeError(f"packed NVFP4 batched smoke failed: max_abs={max_abs}")
    emit("correctness", shape=(m, k, n), batch=batch, expected=float(k),
         max_abs=max_abs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=int, nargs="+", default=list(range(len(CONFIGS))))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--inner", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.dry_run:
        emit("contract", packs=PACKS, configs=[CONFIGS[i] for i in args.configs],
             gate_ms=GATE_MS, margin_ms=MARGIN_MS)
        return

    import cutlass
    import cutlass.cute as cute
    import torch
    # QuACK 0.5 uses pre-4.6 annotation aliases only; the runtime classes are
    # unchanged. Restore those names before importing its Python modules.
    if not hasattr(cute.core, "ThrMma"):
        cute.core.ThrMma = cute.ThrMma
    if not hasattr(cute.core, "ThrCopy"):
        cute.core.ThrCopy = cute.ThrCopy
    try:
        import quack.blockscaled_gemm_utils as blockscaled_gemm
        compile_blockscaled_gemm_tvm_ffi = blockscaled_gemm.compile_blockscaled_gemm_tvm_ffi
        quack_source = blockscaled_gemm.__file__
        quack_api = "blockscaled_gemm_utils"
    except ModuleNotFoundError:
        from functools import partial

        import cutlass.cute as cute
        from quack.compile_utils import make_fake_tensor as fake_tensor
        from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
        from quack.gemm_sm100 import GemmSm100
        from quack.gemm_tvm_ffi_utils import div_for_dtype, make_scheduler_args
        from quack.rounding import RoundingMode
        from quack.varlen_utils import VarlenArguments

        if args.configs != [7]:
            raise RuntimeError(
                "QuACK >=0.6 large-shape default is config 7; "
                "rerun with --configs 7")

        def leading_dim(tensor):
            return next(i for i, stride in enumerate(tensor.stride()) if stride == 1)

        def fake_compact(tensor, dtype):
            logical_shape = list(tensor.shape)
            ld = leading_dim(tensor)
            if dtype == cutlass.Float4E2M1FN:
                logical_shape[ld] *= 2
            return fake_tensor(
                dtype,
                tuple(logical_shape),
                leading_dim=ld,
                divisibility=div_for_dtype(dtype),
            )

        def compile_tensor_like(tensor, dtype):
            compile_tensor = cute.runtime.from_dlpack(tensor)
            compile_tensor.element_type = dtype
            marked = compile_tensor.mark_layout_dynamic(leading_dim=leading_dim(tensor))
            return compile_tensor if marked is None else marked

        def compile_blockscaled_gemm_tvm_ffi(
            ab_dtype, sf_dtype, sf_vec_size, d_dtype, tile, cluster,
            a, b, out, sfa, sfb):
            if get_device_capacity(a.device)[0] != 10:
                raise RuntimeError("SM100 NVFP4 benchmark requires SM100")

            # QuACK 0.6.1's wheel omits blockscaled/utils.py, while importing
            # its public interface pulls in an SM90 epilogue that is
            # incompatible with the installed CUTLASS.  Compile the same
            # native GemmSm100 kernel directly with the base identity
            # epilogue; no package or environment mutation is required.
            gemm = partial(
                GemmSm100,
                sf_vec_size=sf_vec_size,
                use_clc_persistence=True,
            )(cutlass.Float32, ab_dtype, tile, (*cluster, 1))
            gemm.rounding_mode = RoundingMode.RN
            compile_epi_args = gemm.EpilogueArguments()
            scheduler_args = make_scheduler_args(
                get_max_active_clusters(cluster[0] * cluster[1]),
                max_swizzle_size=8,
                tile_count_semaphore=None,
                batch_idx_permute=None,
            )
            stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

            @cute.jit
            def runner(a_, b_, out_, sfa_, sfb_, varlen_args_, stream_):
                gemm(
                    a_, b_, out_, None, compile_epi_args, scheduler_args,
                    varlen_args_, stream_, sfa_, sfb_)

            compiled = cute.compile(
                runner,
                fake_compact(a, ab_dtype),
                fake_compact(b, ab_dtype),
                fake_compact(out, d_dtype),
                compile_tensor_like(sfa, sf_dtype),
                compile_tensor_like(sfb, sf_dtype),
                VarlenArguments(),
                stream,
                options="--enable-tvm-ffi",
            )

            def run(a_, b_, out_, sfa_, sfb_):
                compiled(a_, b_, out_, sfa_, sfb_, VarlenArguments())

            return run

        quack_source = importlib.import_module("quack.gemm_sm100").__file__
        quack_api = "gemm_sm100_direct_identity_epilogue"

    assert torch.cuda.get_device_capability() == (10, 0)
    assert all(0 <= index < len(CONFIGS) for index in args.configs)
    torch.manual_seed(20260721)
    emit("environment", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         torch_source=torch.__file__, cutlass_source=cutlass.__file__,
         quack=importlib.metadata.version("quack-kernels"),
         quack_source=quack_source, quack_api=quack_api, configs=args.configs)
    correctness_smoke(torch, cutlass, compile_blockscaled_gemm_tvm_ffi)

    single = {}
    bf16 = {}
    for pack in PACKS:
        for phase in ("fwd", "dgrad", "wgrad"):
            shape = phase_shape(pack, phase)
            if shape not in single:
                bf16[shape] = bench_bf16_shape(torch, shape, args)
                single[shape] = bench_shape(torch, cutlass, compile_blockscaled_gemm_tvm_ffi,
                                            shape, 1, args.configs, args)

    batched_wgrad = {}
    for pack in PACKS:
        name, count, *_ = pack
        batched_wgrad[name] = bench_shape(
            torch, cutlass, compile_blockscaled_gemm_tvm_ffi,
            phase_shape(pack, "wgrad"), count, args.configs, args)

    phases = {phase: sum(single[phase_shape(pack, phase)] * pack[1] for pack in PACKS)
              for phase in ("fwd", "dgrad", "wgrad")}
    bf16_phases = {phase: sum(bf16[phase_shape(pack, phase)] * pack[1] for pack in PACKS)
                   for phase in ("fwd", "dgrad", "wgrad")}
    bf16_ms = sum(bf16_phases.values())
    unbatched_ms = sum(phases.values())
    deferred_wgrad_ms = sum(batched_wgrad.values())
    deferred_optimized_ms = phases["fwd"] + phases["dgrad"] + deferred_wgrad_ms
    emit("aggregate", bf16_ms=bf16_ms, bf16_phases_ms=bf16_phases,
         unbatched_ms=unbatched_ms, phases_ms=phases,
         deferred_batched_wgrad_ms=deferred_wgrad_ms,
         deferred_optimized_ms=deferred_optimized_ms,
         gate_ms=GATE_MS, margin_ms=MARGIN_MS,
         break_even_speedup=BREAK_EVEN_SPEEDUP,
         margin_speedup=MARGIN_SPEEDUP,
         unbatched_speedup=bf16_ms / unbatched_ms,
         deferred_speedup=bf16_ms / deferred_optimized_ms,
         unbatched_ratio_break_even=bf16_ms / unbatched_ms >= BREAK_EVEN_SPEEDUP,
         unbatched_ratio_margin=bf16_ms / unbatched_ms >= MARGIN_SPEEDUP,
         deferred_ratio_break_even=bf16_ms / deferred_optimized_ms >= BREAK_EVEN_SPEEDUP,
         deferred_ratio_margin=bf16_ms / deferred_optimized_ms >= MARGIN_SPEEDUP,
         unbatched_passes_break_even=unbatched_ms <= GATE_MS,
         unbatched_passes_margin=unbatched_ms <= MARGIN_MS,
         deferred_passes_break_even=deferred_optimized_ms <= GATE_MS,
         deferred_passes_margin=deferred_optimized_ms <= MARGIN_MS,
         unbatched_effective_tflops=300095807422464 / unbatched_ms / 1e9,
         deferred_effective_tflops=300095807422464 / deferred_optimized_ms / 1e9)


if __name__ == "__main__":
    main()
