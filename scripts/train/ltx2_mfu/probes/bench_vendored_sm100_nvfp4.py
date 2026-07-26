#!/usr/bin/env python3
"""Tiny exact-shape gate for PyTorch's vendored SM100 NVFP4 CuTe GEMM."""

import argparse
import importlib
import json
import statistics
import time

import cutlass
import cutlass.cute as cute
import flashinfer
import torch
from cutlass.cute.runtime import from_dlpack


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


# PyTorch 2.12's vendored template uses the pre-4.6 enum spelling; the
# installed 4.6 DSL accepts the same values as strings.
if not hasattr(cute.arch, "ProxyKind"):
    cute.arch.ProxyKind = type("ProxyKind", (), {"async_shared": "async.shared"})
    cute.arch.SharedSpace = type("SharedSpace", (), {"shared_cta": "cta"})


def emit(kind, **fields):
    print(json.dumps({"kind": kind, **fields}, sort_keys=True), flush=True)


def quantize(x):
    one = torch.ones((), device=x.device, dtype=torch.float32)
    q, s = flashinfer.nvfp4_quantize(
        x,
        one,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False,
    )
    return q.view(torch.float4_e2m1fn_x2), s.view(torch.float8_e4m3fn)


def as_cute(x):
    return from_dlpack(x.detach(), assumed_align=16, enable_tvm_ffi=True)


def compile_gemm(kernel_cls, a, b, sfa, sfb, out, tile, cluster):
    kernel = kernel_cls(16, tile, cluster)
    max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        as_cute(a),
        as_cute(b),
        as_cute(sfa),
        as_cute(sfb),
        as_cute(out),
        max_clusters,
        stream,
        options="--enable-tvm-ffi",
    )


def measure(fn, warmup, samples, inner):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / inner)
    return statistics.median(timings), min(timings), max(timings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4290)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--config", type=int, nargs="*")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(20260721)
    module = importlib.import_module(
        "torch._inductor.kernel.vendored_templates.cutedsl.dense_blockscaled_gemm_persistent"
    )
    kernel_cls = module.Sm100BlockScaledPersistentDenseGemmKernel
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16) * 0.02
    aq, asf = quantize(x)
    bq, bsf = quantize(w)
    bq_t = bq.T
    out = torch.empty(args.m, args.n, device="cuda", dtype=torch.bfloat16)
    ref = x @ w.T
    emit(
        "environment",
        torch=torch.__version__,
        cutlass=cutlass.__version__,
        flashinfer=flashinfer.__version__,
        gpu=torch.cuda.get_device_name(),
        shape=(args.m, args.k, args.n),
        a=(tuple(aq.shape), tuple(aq.stride()), str(aq.dtype)),
        b=(tuple(bq_t.shape), tuple(bq_t.stride()), str(bq_t.dtype)),
        sfa=(tuple(asf.shape), asf.numel()),
        sfb=(tuple(bsf.shape), bsf.numel()),
    )

    selected = range(len(CONFIGS)) if args.config is None else args.config
    for index in selected:
        tile, cluster = CONFIGS[index]
        started = time.monotonic()
        try:
            compiled = compile_gemm(kernel_cls, aq, bq_t, asf, bsf, out, tile, cluster)
            compile_s = time.monotonic() - started
            call = lambda: compiled(aq, bq_t, asf, bsf, out)
            call()
            torch.cuda.synchronize()
            diff = out.float() - ref.float()
            rel_rms = (diff.square().mean().sqrt() / ref.float().square().mean().sqrt()).item()
            median, minimum, maximum = measure(call, args.warmup, args.samples, args.inner)
            emit(
                "result",
                index=index,
                tile=tile,
                cluster=cluster,
                compile_seconds=compile_s,
                median_ms=median,
                min_ms=minimum,
                max_ms=maximum,
                relative_rms=rel_rms,
                finite=bool(torch.isfinite(out).all()),
                effective_tflops=2 * args.m * args.k * args.n / median / 1e9,
            )
        except Exception as exc:
            torch.cuda.synchronize()
            emit(
                "failure",
                index=index,
                tile=tile,
                cluster=cluster,
                elapsed_seconds=time.monotonic() - started,
                exception=repr(exc),
            )


if __name__ == "__main__":
    main()
