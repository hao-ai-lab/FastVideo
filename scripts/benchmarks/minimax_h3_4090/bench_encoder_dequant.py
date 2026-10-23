"""Compare serialized NVFP4 weight expansion on an idle consumer GPU."""
import argparse
import json
import os
import pathlib
import time

import torch

from fastvideo.layers.quantization.nvfp4_dequant import dequantize_nvfp4_cuda
from fastvideo.models.encoders.minimax_h3_checkpoint_nvfp4 import dequantize_serialized_nvfp4


def measure(fn, packed, scales, global_scale):
    for _ in range(3):
        fn(packed, scales, global_scale)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    for _ in range(10):
        fn(packed, scales, global_scale)
    stop.record()
    stop.synchronize()
    return {"gpu_ms": start.elapsed_time(stop) / 10,
            "wall_ms": (time.perf_counter() - wall) * 100,
            "peak_extra_gib": (torch.cuda.max_memory_allocated() - baseline) / 2**30}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    rows = []
    global_scale = float(torch.tensor(317.224, dtype=torch.float32))
    torch.manual_seed(63)
    for n, k in [(8192, 8192), (25600, 8192), (8192, 25600)]:
        packed = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
        scales = torch.randint(0, 127, (n, k // 16), device="cuda", dtype=torch.uint8)
        reference = dequantize_serialized_nvfp4(packed, scales, global_scale)
        fused = dequantize_nvfp4_cuda(packed, scales, global_scale)
        torch.testing.assert_close(fused, reference, rtol=0, atol=0)
        del reference, fused
        row = {"n": n, "k": k, "bf16_exact": True,
               "torch": measure(dequantize_serialized_nvfp4, packed, scales, global_scale),
               "fused": measure(dequantize_nvfp4_cuda, packed, scales, global_scale)}
        row["speedup"] = row["torch"]["gpu_ms"] / row["fused"]["gpu_ms"]
        print(json.dumps(row), flush=True)
        rows.append(row)
        del packed, scales
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                                     "source_commit": os.environ.get("FASTVIDEO_SOURCE_COMMIT"),
                                     "global_scale": global_scale, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
