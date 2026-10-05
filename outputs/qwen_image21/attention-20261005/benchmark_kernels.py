"""Compare unchanged SDPA and FastVideo's FA4 wrapper on Qwen target Q/K/V.

Run with FASTVIDEO_FA4=1. No weights or model arithmetic are modified.
Reported latency excludes imports/JIT/warmup and uses CUDA events.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from fastvideo.attention.backends.flash_attn import FlashAttentionImpl, fa_version


def main():
    out = Path(__file__).resolve().parent
    report = {
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "flash_attention_version": fa_version, "dtype": "bfloat16",
        "heads": 32, "head_dim": 128, "warmup": 3, "iterations": 12, "cases": [],
    }
    assert fa_version == "4"
    impl = FlashAttentionImpl(num_heads=32, head_size=128, causal=False, softmax_scale=128**-0.5)
    torch.manual_seed(42)
    for name, target, prefix in [("t2i_1024", 4096, 26), ("i2i_1024", 4096, 4200),
                                 ("t2i_2048", 16384, 26)]:
        q = torch.randn(1, target, 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, target + prefix, 32, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)

        def sdpa():
            return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                  v.transpose(1, 2), dropout_p=0).transpose(1, 2)

        def fa4():
            return impl.forward(q, k, v, None)

        def cudnn():
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return sdpa()

        timings = {}
        errors = {}
        with torch.inference_mode():
            baseline = sdpa()
            actual = fa4()
            error = (actual.float() - baseline.float()).abs()
            for backend, fn in [("sdpa", sdpa), ("fa4", fa4), ("cudnn", cudnn)]:
                try:
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(12):
                        fn()
                    end.record()
                    end.synchronize()
                    timings[backend] = start.elapsed_time(end) / 12
                    if backend == "cudnn":
                        delta = (fn().float() - baseline.float()).abs()
                        report.setdefault("cudnn_errors", {})[name] = {
                            "max_abs": delta.max().item(), "mean_abs": delta.mean().item()}
                except RuntimeError as exc:
                    errors[backend] = str(exc)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA]) as profile:
                sdpa()
                torch.cuda.synchronize()
            operators = [event.key for event in profile.key_averages()
                         if "scaled_dot_product" in event.key or "flash" in event.key]
        case = {"name": name, "query_tokens": target, "kv_tokens": target + prefix,
                "milliseconds": timings, "speedup": timings["sdpa"] / timings["fa4"],
                "max_abs": error.max().item(), "mean_abs": error.mean().item(),
                "finite": bool(torch.isfinite(actual).all()), "sdpa_operators": operators,
                "unavailable_backends": errors}
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
        (out / "kernel-metrics-cudnn.json").write_text(json.dumps(report, indent=2) + "\n")
        del q, k, v, baseline, actual, error
    print("KERNEL_BENCHMARK " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
