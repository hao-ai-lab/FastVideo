#!/usr/bin/env python3
"""cuBLASLt heuristic-algo sweep for the packed LTX-2 GEMM band (plan item 2).

Enumerates ``cublasLtMatmulAlgoGetHeuristic`` candidates per exact shape and
training orientation (fwd, dgrad, wgrad), times each against torch's own
dispatch in the same process, and reports the best-algo margin. Kill
criterion from ``reports/bf16_kernel_research_plan.md``: if the best
heuristic algo is within 2% of torch/nvjet per weighted band, native algo
pinning is closed. Bare GEMMs (no bias epilogue) on both sides; BF16 inputs,
FP32 compute/accumulate. Parity-checked against torch per case.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics

import torch

CUDA_R_16BF = 14
CUDA_R_32F = 0
CUBLAS_COMPUTE_32F = 68
OP_N, OP_T = 0, 1
DESC_TRANSA, DESC_TRANSB = 3, 4
PREF_MAX_WORKSPACE = 1
HIDDEN = 4096
FFN = 16384
VIDEO_TOKENS = 11 * 15 * 26
TEXT_TOKENS = 1024
WORKSPACE_BYTES = 128 * 1024 * 1024
MAX_ALGOS = 48


class HeuristicResult(ctypes.Structure):
    _fields_ = [
        ("algo", ctypes.c_uint64 * 8),
        ("workspaceSize", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("wavesCount", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


def load_lt() -> ctypes.CDLL:
    torch.cuda.init()
    for name in ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError("libcublasLt not found")


def check(status: int, what: str) -> None:
    if status != 0:
        raise RuntimeError(f"{what} failed with cublas status {status}")


def build_cases(batch: int) -> list[dict]:
    m_video = VIDEO_TOKENS * batch
    m_text = TEXT_TOKENS * batch
    shapes = [
        ("self_qkv", m_video, HIDDEN, 3 * HIDDEN, 48),
        ("video_dd", m_video, HIDDEN, HIDDEN, 144),
        ("text_kv", m_text, HIDDEN, 2 * HIDDEN, 48),
        ("ffn_up", m_video, HIDDEN, FFN, 48),
        ("ffn_down", m_video, FFN, HIDDEN, 48),
    ]
    cases = []
    for name, rows, in_features, out_features, occurrences in shapes:
        # Row-major training GEMMs expressed as column-major cuBLASLt calls
        # (documented mapping: fwd y=xW^T, dgrad dx=dyW, wgrad dW=dy^T x).
        cases.append({
            "case": f"{name}:fwd", "occurrences": occurrences,
            "m": out_features, "n": rows, "k": in_features,
            "transa": OP_T, "transb": OP_N,
            "a_shape": (out_features, in_features), "b_shape": (rows, in_features),
            "d_shape": (rows, out_features),
            "torch_fn": lambda a, b: b @ a.t(),
        })
        cases.append({
            "case": f"{name}:dgrad", "occurrences": occurrences,
            "m": in_features, "n": rows, "k": out_features,
            "transa": OP_N, "transb": OP_N,
            "a_shape": (out_features, in_features), "b_shape": (rows, out_features),
            "d_shape": (rows, in_features),
            "torch_fn": lambda a, b: b @ a,
        })
        cases.append({
            "case": f"{name}:wgrad", "occurrences": occurrences,
            "m": in_features, "n": out_features, "k": rows,
            "transa": OP_N, "transb": OP_T,
            "a_shape": (rows, in_features), "b_shape": (rows, out_features),
            "d_shape": (out_features, in_features),
            "torch_fn": lambda a, b: b.t() @ a,
        })
    return cases


def time_fn(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeats)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in events)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()

    torch.manual_seed(20260722)
    device = torch.device("cuda:0")
    lt = load_lt()
    handle = ctypes.c_void_p()
    check(lt.cublasLtCreate(ctypes.byref(handle)), "ltCreate")
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)

    summary = []
    for case in build_cases(args.batch):
        a_rm = torch.randn(case["a_shape"], device=device, dtype=torch.bfloat16)
        b_rm = torch.randn(case["b_shape"], device=device, dtype=torch.bfloat16)
        d = torch.empty(case["d_shape"], device=device, dtype=torch.bfloat16)
        reference = case["torch_fn"](a_rm, b_rm)

        # Column-major operand descriptors. Buffers: A is the weight-like
        # tensor, B the activation-like tensor, both row-major contiguous.
        m, n, k = case["m"], case["n"], case["k"]
        op_desc = ctypes.c_void_p()
        check(lt.cublasLtMatmulDescCreate(ctypes.byref(op_desc), CUBLAS_COMPUTE_32F, CUDA_R_32F), "descCreate")
        for attr, value in ((DESC_TRANSA, case["transa"]), (DESC_TRANSB, case["transb"])):
            v = ctypes.c_int32(value)
            check(lt.cublasLtMatmulDescSetAttribute(op_desc, attr, ctypes.byref(v), 4), "descSet")

        def layout(rows_cm: int, cols_cm: int, ld: int) -> ctypes.c_void_p:
            handle_ = ctypes.c_void_p()
            check(lt.cublasLtMatrixLayoutCreate(ctypes.byref(handle_), CUDA_R_16BF, rows_cm, cols_cm,
                                                ctypes.c_int64(ld)), "layoutCreate")
            return handle_

        lda = case["a_shape"][1]
        ldb = case["b_shape"][1]
        a_rows_cm = lda if case["transa"] == OP_T else m
        a_cols_cm = m if case["transa"] == OP_T else k
        b_rows_cm = ldb if case["transb"] == OP_T else k
        b_cols_cm = k if case["transb"] == OP_T else n
        layout_a = layout(a_rows_cm, a_cols_cm, lda)
        layout_b = layout(b_rows_cm, b_cols_cm, ldb)
        layout_d = layout(m, n, m)

        pref = ctypes.c_void_p()
        check(lt.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "prefCreate")
        ws = ctypes.c_size_t(WORKSPACE_BYTES)
        check(lt.cublasLtMatmulPreferenceSetAttribute(pref, PREF_MAX_WORKSPACE, ctypes.byref(ws), 8), "prefSet")

        results = (HeuristicResult * MAX_ALGOS)()
        found = ctypes.c_int(0)
        status = lt.cublasLtMatmulAlgoGetHeuristic(handle, op_desc, layout_a, layout_b, layout_d, layout_d,
                                                   pref, MAX_ALGOS, results, ctypes.byref(found))
        check(status, "algoGetHeuristic")

        def run_lt(algo_ref) -> None:
            check(
                lt.cublasLtMatmul(handle, op_desc, ctypes.byref(alpha),
                                  ctypes.c_void_p(a_rm.data_ptr()), layout_a,
                                  ctypes.c_void_p(b_rm.data_ptr()), layout_b,
                                  ctypes.byref(beta),
                                  ctypes.c_void_p(d.data_ptr()), layout_d,
                                  ctypes.c_void_p(d.data_ptr()), layout_d,
                                  algo_ref,
                                  ctypes.c_void_p(workspace.data_ptr()), ctypes.c_size_t(WORKSPACE_BYTES),
                                  stream), "ltMatmul")

        torch_ms = time_fn(lambda: case["torch_fn"](a_rm, b_rm), args.warmup, args.repeats)

        best = None
        parity_max_abs = None
        for index in range(found.value):
            if results[index].state != 0:
                continue
            algo_ref = ctypes.byref(results[index], 0)
            try:
                run_lt(algo_ref)
            except RuntimeError:
                continue
            torch.cuda.synchronize()
            diff = (d.float() - reference.float()).abs().max().item()
            if diff > 0.25:
                continue
            lt_ms = time_fn(lambda: run_lt(algo_ref), args.warmup, args.repeats)
            if best is None or lt_ms < best[0]:
                best = (lt_ms, index)
                parity_max_abs = diff

        row = {
            "case": case["case"],
            "mnk": [m, n, k],
            "occurrences": case["occurrences"],
            "algos_returned": found.value,
            "torch_ms": round(torch_ms, 4),
            "best_lt_ms": round(best[0], 4) if best else None,
            "best_algo_index": best[1] if best else None,
            "parity_max_abs": parity_max_abs,
            "lt_vs_torch_pct": round(100.0 * (best[0] - torch_ms) / torch_ms, 2) if best else None,
        }
        summary.append(row)
        print("LT_SWEEP_CASE " + json.dumps(row, sort_keys=True), flush=True)
        for obj, destroy in ((pref, lt.cublasLtMatmulPreferenceDestroy), (layout_a, lt.cublasLtMatrixLayoutDestroy),
                             (layout_b, lt.cublasLtMatrixLayoutDestroy), (layout_d, lt.cublasLtMatrixLayoutDestroy),
                             (op_desc, lt.cublasLtMatmulDescDestroy)):
            destroy(obj)

    weighted_torch = sum(r["torch_ms"] * r["occurrences"] for r in summary)
    weighted_lt = sum((r["best_lt_ms"] or r["torch_ms"]) * r["occurrences"] for r in summary)
    print(
        "LT_SWEEP_RESULT " + json.dumps({
            "batch": args.batch,
            "weighted_torch_band_ms": round(weighted_torch, 2),
            "weighted_best_lt_band_ms": round(weighted_lt, 2),
            "band_delta_pct": round(100.0 * (weighted_lt - weighted_torch) / weighted_torch, 3),
            "cases": summary,
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
