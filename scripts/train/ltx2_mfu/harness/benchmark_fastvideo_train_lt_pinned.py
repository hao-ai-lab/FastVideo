#!/usr/bin/env python3
"""Scratch gate: pinned cuBLASLt dgrad/wgrad algos for the packed projections.

Follows research-plan item 2 after the algo sweep found an 8-9% weighted GEMM
band win with bit-exact parity. This driver builds the ``lt_pinned_ops`` C++
extension, tunes the eight conservative backward cases (dgrad/wgrad where the
sweep beat torch by >2%; forward stays stock nvjet with fused bias), wraps the
matching block-level ``nn.Linear`` modules in an autograd function whose
backward dispatches to the pinned algos, and runs the frozen packed benchmark
harness. Bias gradients fall back to an eager ``sum(0)`` (integration would
fuse BGRADB later).
"""

from __future__ import annotations

import ctypes
import os
import statistics

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import benchmark_fastvideo_train_pack_d016 as benchmark
import fastvideo.train.trainer as trainer_module

CUDA_R_16BF = 14
CUDA_R_32F = 0
CUBLAS_COMPUTE_32F = 68
OP_N, OP_T = 0, 1
DESC_TRANSA, DESC_TRANSB = 3, 4
PREF_MAX_WORKSPACE = 1
WORKSPACE_BYTES = 128 * 1024 * 1024
MAX_ALGOS = 48
HIDDEN, FFN = 4096, 16384
VIDEO_TOKENS, TEXT_TOKENS = 11 * 15 * 26, 1024
LOCAL_BATCH = int(os.environ.get("FASTVIDEO_BENCH_LOCAL_BATCH_SIZE", "1"))
MIN_WIN_PCT = 2.0

_ext = load(
    name="lt_pinned_ops",
    sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probes", "lt_pinned_ops.cpp")],
    extra_ldflags=["-lcublasLt"],
    with_cuda=True,
    verbose=False,
)


class HeuristicResult(ctypes.Structure):
    _fields_ = [
        ("algo", ctypes.c_uint64 * 8),
        ("workspaceSize", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("wavesCount", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


def _tune_case(lt, handle, workspace, name, m, n, k, transa, transb, a_shape, b_shape, d_shape, torch_fn):
    device = torch.device("cuda")
    a = torch.randn(a_shape, device=device, dtype=torch.bfloat16)
    b = torch.randn(b_shape, device=device, dtype=torch.bfloat16)
    d = torch.empty(d_shape, device=device, dtype=torch.bfloat16)
    reference = torch_fn(a, b)

    def check(status, what):
        if status != 0:
            raise RuntimeError(f"{what} failed: {status}")

    op_desc = ctypes.c_void_p()
    check(lt.cublasLtMatmulDescCreate(ctypes.byref(op_desc), CUBLAS_COMPUTE_32F, CUDA_R_32F), "descCreate")
    for attr, value in ((DESC_TRANSA, transa), (DESC_TRANSB, transb)):
        v = ctypes.c_int32(value)
        check(lt.cublasLtMatmulDescSetAttribute(op_desc, attr, ctypes.byref(v), 4), "descSet")
    lda, ldb = a_shape[1], b_shape[1]

    def layout(rows, cols, ld):
        h = ctypes.c_void_p()
        check(lt.cublasLtMatrixLayoutCreate(ctypes.byref(h), CUDA_R_16BF, rows, cols, ctypes.c_int64(ld)), "layout")
        return h

    layout_a = layout(lda if transa == OP_T else m, m if transa == OP_T else k, lda)
    layout_b = layout(ldb if transb == OP_T else k, k if transb == OP_T else n, ldb)
    layout_d = layout(m, n, m)
    pref = ctypes.c_void_p()
    check(lt.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "prefCreate")
    ws = ctypes.c_size_t(WORKSPACE_BYTES)
    check(lt.cublasLtMatmulPreferenceSetAttribute(pref, PREF_MAX_WORKSPACE, ctypes.byref(ws), 8), "prefSet")
    results = (HeuristicResult * MAX_ALGOS)()
    found = ctypes.c_int(0)
    check(lt.cublasLtMatmulAlgoGetHeuristic(handle, op_desc, layout_a, layout_b, layout_d, layout_d,
                                            pref, MAX_ALGOS, results, ctypes.byref(found)), "heuristic")
    alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def run_lt(index):
        check(lt.cublasLtMatmul(handle, op_desc, ctypes.byref(alpha),
                                ctypes.c_void_p(a.data_ptr()), layout_a,
                                ctypes.c_void_p(b.data_ptr()), layout_b,
                                ctypes.byref(beta),
                                ctypes.c_void_p(d.data_ptr()), layout_d,
                                ctypes.c_void_p(d.data_ptr()), layout_d,
                                ctypes.byref(results[index], 0),
                                ctypes.c_void_p(workspace.data_ptr()), ctypes.c_size_t(WORKSPACE_BYTES),
                                stream), "ltMatmul")

    def time_fn(fn):
        for _ in range(6):
            fn()
        torch.cuda.synchronize()
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(15)]
        for start, end in events:
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        return statistics.median(start.elapsed_time(end) for start, end in events)

    torch_ms = time_fn(lambda: torch_fn(a, b))
    best = None
    for index in range(found.value):
        if results[index].state != 0:
            continue
        try:
            run_lt(index)
        except RuntimeError:
            continue
        torch.cuda.synchronize()
        if (d.float() - reference.float()).abs().max().item() > 0.0:
            continue
        lt_ms = time_fn(lambda: run_lt(index))
        if best is None or lt_ms < best[0]:
            best = (lt_ms, index)
    win_pct = 100.0 * (torch_ms - best[0]) / torch_ms if best else -1.0
    spec = None
    if best and win_pct >= MIN_WIN_PCT:
        spec = {
            "m": m, "n": n, "k": k, "transa": transa == OP_T, "transb": transb == OP_T,
            "lda": lda, "ldb": ldb, "d_rows": d_shape[0], "d_cols": d_shape[1],
            "algo": bytes(results[best[1]].algo),
        }
    print(f"LT_PIN_TUNE {name} torch_ms={torch_ms:.4f} best_lt_ms={best[0] if best else -1:.4f} "
          f"win_pct={win_pct:.2f} pinned={spec is not None}", flush=True)
    return spec


def _tune_all_blobs() -> dict[tuple[int, int], dict]:
    """Rank-0 tuning: returns per-shape dgrad/wgrad win metadata + algo blobs."""
    lt = None
    for so in ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"):
        try:
            lt = ctypes.CDLL(so)
            break
        except OSError:
            continue
    if lt is None:
        raise OSError("libcublasLt not found")
    handle = ctypes.c_void_p()
    if lt.cublasLtCreate(ctypes.byref(handle)) != 0:
        raise RuntimeError("cublasLtCreate failed")
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    mapping: dict[tuple[int, int], dict] = {}
    shapes = [
        ("self_qkv", VIDEO_TOKENS * LOCAL_BATCH, HIDDEN, 3 * HIDDEN),
        ("video_dd", VIDEO_TOKENS * LOCAL_BATCH, HIDDEN, HIDDEN),
        ("text_kv", TEXT_TOKENS * LOCAL_BATCH, HIDDEN, 2 * HIDDEN),
        ("ffn_up", VIDEO_TOKENS * LOCAL_BATCH, HIDDEN, FFN),
        ("ffn_down", VIDEO_TOKENS * LOCAL_BATCH, FFN, HIDDEN),
    ]
    for name, rows, in_features, out_features in shapes:
        dgrad_spec = _tune_case(
            lt, handle, workspace, f"{name}:dgrad",
            in_features, rows, out_features, OP_N, OP_N,
            (out_features, in_features), (rows, out_features), (rows, in_features),
            lambda a, b: b @ a)
        wgrad_spec = _tune_case(
            lt, handle, workspace, f"{name}:wgrad",
            in_features, out_features, rows, OP_N, OP_T,
            (rows, in_features), (rows, out_features), (out_features, in_features),
            lambda a, b: b.t() @ a)
        mapping[(in_features, out_features)] = {"rows": rows, "dgrad": dgrad_spec, "wgrad": wgrad_spec}
    return mapping


def _register_from_spec(spec: dict | None) -> int:
    if spec is None:
        return -1
    blob = torch.frombuffer(bytearray(spec["algo"]), dtype=torch.uint8).clone()
    return _ext.register_case(spec["m"], spec["n"], spec["k"], spec["transa"], spec["transb"],
                              spec["lda"], spec["ldb"], spec["d_rows"], spec["d_cols"], blob)


def _tune_and_broadcast() -> dict[tuple[int, int], tuple[int, int, int]]:
    import torch.distributed as dist
    payload = [None]
    if not dist.is_initialized() or dist.get_rank() == 0:
        payload = [_tune_all_blobs()]
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    mapping: dict[tuple[int, int], tuple[int, int, int]] = {}
    for key, entry in payload[0].items():
        mapping[key] = (entry["rows"], _register_from_spec(entry["dgrad"]), _register_from_spec(entry["wgrad"]))
    return mapping


@torch.library.custom_op("fastvideo::lt_pinned_dgrad", mutates_args=())
def lt_pinned_dgrad(grad_out: torch.Tensor, weight: torch.Tensor, case_id: int) -> torch.Tensor:
    return _ext.lt_mm(weight.contiguous(), grad_out.contiguous(), case_id)


@lt_pinned_dgrad.register_fake
def _(grad_out, weight, case_id):
    return grad_out.new_empty((grad_out.shape[0], weight.shape[1]))


@torch.library.custom_op("fastvideo::lt_pinned_wgrad", mutates_args=())
def lt_pinned_wgrad(grad_out: torch.Tensor, saved_input: torch.Tensor, case_id: int) -> torch.Tensor:
    return _ext.lt_mm(saved_input.contiguous(), grad_out.contiguous(), case_id)


@lt_pinned_wgrad.register_fake
def _(grad_out, saved_input, case_id):
    return grad_out.new_empty((grad_out.shape[1], saved_input.shape[1]))


@torch.library.custom_op("fastvideo::lt_pinned_linear", mutates_args=())
def lt_pinned_linear(x2d: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                     rows: int, dgrad_id: int, wgrad_id: int) -> torch.Tensor:
    return F.linear(x2d, weight, bias)


@lt_pinned_linear.register_fake
def _(x2d, weight, bias, rows, dgrad_id, wgrad_id):
    return x2d.new_empty((x2d.shape[0], weight.shape[0]))


def _lt_linear_setup(ctx, inputs, output):
    x2d, weight, bias, rows, dgrad_id, wgrad_id = inputs
    ctx.save_for_backward(x2d, weight)
    ctx.meta = (rows, dgrad_id, wgrad_id, bias is not None)


def _lt_linear_backward(ctx, grad_out):
    x2d, weight = ctx.saved_tensors
    rows, dgrad_id, wgrad_id, has_bias = ctx.meta
    grad_out = grad_out.contiguous()
    if grad_out.shape[0] == rows and dgrad_id >= 0:
        grad_input = torch.ops.fastvideo.lt_pinned_dgrad(grad_out, weight, dgrad_id)
    else:
        grad_input = grad_out @ weight
    if grad_out.shape[0] == rows and wgrad_id >= 0:
        grad_weight = torch.ops.fastvideo.lt_pinned_wgrad(grad_out, x2d, wgrad_id)
    else:
        grad_weight = grad_out.t() @ x2d
    grad_bias = grad_out.sum(0) if has_bias else None
    return grad_input, grad_weight, grad_bias, None, None, None


lt_pinned_linear.register_autograd(_lt_linear_backward, setup_context=_lt_linear_setup)


def _patched_forward(module, rows, dgrad_id, wgrad_id):

    def forward(x):
        bias = module.bias if not module.skip_bias_add else None
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        out = torch.ops.fastvideo.lt_pinned_linear(x2d, module.weight, bias, rows, dgrad_id, wgrad_id)
        out = out.reshape(*shape[:-1], out.shape[-1])
        output_bias = module.bias if module.skip_bias_add else None
        return out, output_bias

    return forward


def _install(method) -> None:
    from fastvideo.layers.linear import ReplicatedLinear

    mapping = _tune_and_broadcast()
    patched = 0
    for name, module in method.student.transformer.named_modules():
        if not isinstance(module, ReplicatedLinear) or "transformer_blocks" not in name:
            continue
        key = (module.input_size, module.output_size)
        if key not in mapping:
            continue
        rows, dgrad_id, wgrad_id = mapping[key]
        if dgrad_id < 0 and wgrad_id < 0:
            continue
        module.forward = _patched_forward(module, rows, dgrad_id, wgrad_id)
        patched += 1
    print(f"LT_PIN_INSTALLED modules={patched}", flush=True)


_original_run = trainer_module.Trainer.run


def _run_with_pins(self, method, **kwargs):
    _install(method)
    return _original_run(self, method, **kwargs)


trainer_module.Trainer.run = _run_with_pins

if __name__ == "__main__":
    benchmark.main()
