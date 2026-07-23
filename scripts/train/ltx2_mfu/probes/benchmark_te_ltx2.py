import argparse
import gc
import json
import statistics

import torch
import torch.nn.functional as F
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import (
    DelayedScaling,
    Format,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
)


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

BASELINE_STEP_S = 0.479309913
BASELINE_BLOCK_GEMM_S = 0.216001
MFU_PERCENT_SECONDS = 14.444115


def _recipe(name):
    if name == "bf16":
        return None
    if name == "fp8_delayed":
        return DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=16,
            amax_compute_algo="max",
        )
    if name == "mxfp8":
        return MXFP8BlockScaling(fp8_format=Format.E4M3)
    if name in ("nvfp4", "nvfp4_primary"):
        return NVFP4BlockScaling()
    if name == "nvfp4_1d_weight":
        return NVFP4BlockScaling(disable_2d_quantization=True)
    raise ValueError(name)


def _measure(name, layout_name, shape_name, warmup, repeat, bias, alignment, m_multiplier):
    base_m, k, n, count = LAYOUTS[layout_name][shape_name]
    m = base_m * m_multiplier
    recipe = _recipe(name)
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    if recipe is None:
        layer = torch.nn.Linear(k, n, bias=bias, device="cuda", dtype=torch.bfloat16)
        padded_m = m
    else:
        if name == "nvfp4_primary":
            with te.quantized_model_init(
                enabled=True,
                recipe=recipe,
                preserve_high_precision_init_val=False,
            ):
                layer = te.Linear(
                    k,
                    n,
                    bias=bias,
                    device="cuda",
                    params_dtype=torch.bfloat16,
                )
        else:
            layer = te.Linear(k, n, bias=bias, device="cuda", params_dtype=torch.bfloat16)
        # TE block recipes require the flattened row count to be aligned.
        padded_m = ((m + alignment - 1) // alignment) * alignment

    def step():
        layer.zero_grad(set_to_none=True)
        x.grad = None
        if recipe is None:
            y = layer(x)
            grad = dy
        else:
            x_in = F.pad(x, (0, 0, 0, padded_m - m)) if padded_m != m else x
            with te.autocast(enabled=True, recipe=recipe):
                y_padded = layer(x_in, is_first_microbatch=None)
            y = y_padded[:m]
            grad = dy
        y.backward(grad)

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
        "precision": name,
        "layout": layout_name,
        "shape": shape_name,
        "base_m": base_m,
        "m": m,
        "m_multiplier": m_multiplier,
        "padded_m": padded_m,
        "k": k,
        "n": n,
        "count_per_block": count,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
    }
    del layer, x, dy
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--alignment", type=int, default=16)
    parser.add_argument("--m-multiplier", type=int, default=1)
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["bf16", "fp8_delayed", "mxfp8", "nvfp4", "nvfp4_1d_weight"],
    )
    args = parser.parse_args()

    print(json.dumps({
        "torch": torch.__version__,
        "transformer_engine": transformer_engine.__version__,
        "device": torch.cuda.get_device_name(),
        "nvfp4": te.is_nvfp4_available(return_reason=True),
        "mxfp8": te.is_mxfp8_available(return_reason=True),
    }, default=str), flush=True)
    results = []
    for layout_name, shapes in LAYOUTS.items():
        for precision in args.precisions:
            for shape_name in shapes:
                result = _measure(
                    precision,
                    layout_name,
                    shape_name,
                    args.warmup,
                    args.repeat,
                    not args.no_bias,
                    args.alignment,
                    args.m_multiplier,
                )
                results.append(result)
                print(json.dumps(result), flush=True)

    totals = {
        layout_name: {
            precision: sum(
                row["median_ms"] * row["count_per_block"]
                for row in results
                if row["layout"] == layout_name and row["precision"] == precision
            )
            for precision in args.precisions
        }
        for layout_name in LAYOUTS
    }
    current_bf16 = totals["separate"]["bf16"]
    summary = {}
    for layout_name, layout_totals in totals.items():
        layout_bf16 = layout_totals["bf16"]
        summary[layout_name] = {}
        for precision, weighted_ms in layout_totals.items():
            linear_ratio = weighted_ms / current_bf16
            projected_step_s = (
                BASELINE_STEP_S - BASELINE_BLOCK_GEMM_S + BASELINE_BLOCK_GEMM_S * linear_ratio
            )
            summary[layout_name][precision] = {
                "weighted_ms_per_block": weighted_ms,
                "weighted_speedup_vs_layout_bf16": layout_bf16 / weighted_ms,
                "weighted_speedup_vs_current_bf16": current_bf16 / weighted_ms,
                "projected_step_s": projected_step_s,
                "projected_step_speedup": BASELINE_STEP_S / projected_step_s,
                "projected_mfu_percent": MFU_PERCENT_SECONDS / projected_step_s,
            }
    print(json.dumps({
        "projection_baseline": {
            "layout": "separate",
            "precision": "bf16",
            "step_s": BASELINE_STEP_S,
            "block_gemm_s": BASELINE_BLOCK_GEMM_S,
            "m_multiplier": args.m_multiplier,
        },
        "summary": summary,
    }), flush=True)


if __name__ == "__main__":
    main()
