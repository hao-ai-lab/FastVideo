import argparse
import gc
import json
import statistics

import torch
from torchao.float8 import Float8LinearConfig, convert_to_float8_training


LAYOUTS = {
    "separate": {
        "h_h": (4290, 4096, 4096, 6),
        "text_kv": (1024, 4096, 4096, 2),
        "ffn_up": (4290, 4096, 16384, 1),
        "ffn_down": (4290, 16384, 4096, 1),
    },
    "packed": {
        "self_qkv": (4290, 4096, 12288, 1),
        "h_h": (4290, 4096, 4096, 3),
        "text_kv": (1024, 4096, 8192, 1),
        "ffn_up": (4290, 4096, 16384, 1),
        "ffn_down": (4290, 16384, 4096, 1),
    },
}


def measure(precision, layout, shape, warmup, repeat, compile_module):
    m, k, n, count = LAYOUTS[layout][shape]
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    layer = torch.nn.Linear(k, n, bias=True, device="cuda", dtype=torch.bfloat16)
    if precision == "fp8":
        layer = convert_to_float8_training(
            layer,
            config=Float8LinearConfig(pad_inner_dim=True),
        )
    elif precision != "bf16":
        raise ValueError(precision)
    if compile_module:
        layer = torch.compile(layer, dynamic=False)

    def step():
        layer.zero_grad(set_to_none=True)
        x.grad = None
        layer(x).backward(dy)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    result = {
        "precision": precision,
        "layout": layout,
        "shape": shape,
        "m": m,
        "k": k,
        "n": n,
        "count_per_block": count,
        "compiled": compile_module,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
    }
    del layer, x, dy
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--layouts", nargs="+", choices=tuple(LAYOUTS), default=list(LAYOUTS))
    parser.add_argument("--shapes", nargs="+")
    parser.add_argument("--precisions", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"])
    args = parser.parse_args()

    print(json.dumps({
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "compiled": args.compile,
    }), flush=True)
    results = []
    for layout in args.layouts:
        shapes = LAYOUTS[layout]
        for precision in args.precisions:
            for shape in shapes:
                if args.shapes is not None and shape not in args.shapes:
                    continue
                row = measure(
                    precision,
                    layout,
                    shape,
                    args.warmup,
                    args.repeat,
                    args.compile,
                )
                results.append(row)
                print(json.dumps(row), flush=True)

    totals = {
        layout: {
            precision: sum(
                row["median_ms"] * row["count_per_block"]
                for row in results
                if row["layout"] == layout and row["precision"] == precision
            )
            for precision in args.precisions
        }
        for layout in args.layouts
    }
    print(json.dumps({"totals": totals}), flush=True)


if __name__ == "__main__":
    main()
