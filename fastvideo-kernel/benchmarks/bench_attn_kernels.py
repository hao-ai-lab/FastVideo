#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the four attention kernels: speed, accuracy, TFLOPS.

Lanes (each probes availability + arch gate and skips cleanly, with a
printed reason, where it cannot run):

  bf16_flash_attn / bf16_sdpa
                    High-precision baseline: flash-attn (flash_attn_func)
                    when importable, else torch SDPA — the lane name says
                    which one is running. (Prefix follows --dtype.)
  fp8_sageattn      ``sageattention.sageattn`` — the attention op activated by
                    examples/inference/optimizations/fp8_wan2_1_1_3b.py
                    (PR #1496, FASTVIDEO_ATTENTION_BACKEND=SAGE_ATTN →
                    fastvideo/attention/backends/sage_attn.py). sageattn
                    auto-dispatches per-arch; on sm89+ it selects the
                    qk-int8 / pv-fp8 CUDA kernels.
  nvfp4_local       fastvideo-kernel/attn_qat_infer (local SageAttention3
                    port, ``sageattn_blackwell``). Built for sm_120a only.
  nvfp4_flashinfer  flashinfer.nvfp4_attention_sm120_quantize_qkv +
                    nvfp4_attention_sm120_fwd. Compute capability 12.0/12.1,
                    head_dim 64/128.

Accuracy for every lane is measured against an fp32 torch-SDPA ground truth
(chunked over query length so the score matrix never materializes whole)
computed on the same fixed-seed inputs. Reported: cosine similarity,
max abs err, mean rel err.

Inputs are BIASED by default: q and k get a nonzero mean shift (--bias, v
stays zero-mean so a constant output offset doesn't inflate cosine
similarity). Zero-mean randn is blind to mean-dependent quantization errors —
the flashinfer per-block-mean qk-correction head-broadcast bug produces
cos ~0.99 on zero-mean inputs but cos <0.5 at bias 1.0. Use
--include-zeromean to also run the zero-mean rows for contrast.

All lanes run at the [B, H, L, D] interface.

Usage:
  python bench_attn_kernels.py                          # default shapes
  python bench_attn_kernels.py --shapes 1x12x32760x128,1x24x4096x64
  python bench_attn_kernels.py --causal --json out.json
  python bench_attn_kernels.py --shapes-jsonl /tmp/attn_shapes.jsonl.pid1234
      # shapes collected via FASTVIDEO_ATTN_SHAPE_LOG (fastvideo/attention/
      # shape_logger.py); recorded causal/dtype/sm_scale are honored.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Pure helpers (CPU-unit-tested in fastvideo/tests/attention/
# test_attn_kernels_bench.py) — keep them torch-free where possible.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Shape:
    batch: int
    num_q_heads: int
    num_kv_heads: int
    seq_len_q: int
    seq_len_kv: int
    head_dim: int
    causal: bool
    dtype: str = "torch.bfloat16"
    sm_scale: float | None = None

    def label(self) -> str:
        base = f"{self.batch}x{self.num_q_heads}x{self.seq_len_q}x{self.head_dim}"
        if self.seq_len_kv != self.seq_len_q or self.num_kv_heads != self.num_q_heads:
            base += f" (kv {self.num_kv_heads}x{self.seq_len_kv})"
        if self.causal:
            base += " causal"
        return base


def attn_tflops(shape: Shape, ms: float) -> float:
    """TFLOPS of one dense attention forward.

    FLOPs = 4 * B * H * Lq * Lk * D:
      QK^T is B*H*Lq*Lk*D multiply-adds = 2*B*H*Lq*Lk*D FLOPs, and P@V is
      another 2*B*H*Lq*Lk*D. Causal masking computes ~half the score matrix,
      so FLOPs are halved. TFLOPS = FLOPs / (ms * 1e9).
    """
    flops = 4.0 * shape.batch * shape.num_q_heads * shape.seq_len_q * shape.seq_len_kv * shape.head_dim
    if shape.causal:
        flops *= 0.5
    return flops / (ms * 1e9)


def parse_shapes_cli(spec: str, causal: bool, dtype: str) -> list[Shape]:
    """Parse --shapes 'BxHxLxD,BxHxLxD,...' (self-attention shapes)."""
    shapes = []
    for part in spec.split(","):
        try:
            b, h, length, d = (int(x) for x in part.lower().split("x"))
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"bad shape {part!r}, expected BxHxLxD") from e
        shapes.append(Shape(batch=b, num_q_heads=h, num_kv_heads=h, seq_len_q=length,
                            seq_len_kv=length, head_dim=d, causal=causal, dtype=dtype))
    return shapes


def parse_shapes_jsonl(paths: list[str], default_dtype: str) -> list[Shape]:
    """Parse fastvideo/attention/shape_logger.py output (one JSON object per
    line; keys: batch, num_q_heads, num_kv_heads, seq_len_q, seq_len_kv,
    head_dim, dtype, causal, sm_scale, count, ...). Dedupes across files,
    preserving first-seen order."""
    shapes: list[Shape] = []
    seen: set[Shape] = set()
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                shape = Shape(
                    batch=int(rec["batch"]),
                    num_q_heads=int(rec["num_q_heads"]),
                    num_kv_heads=int(rec.get("num_kv_heads", rec["num_q_heads"])),
                    seq_len_q=int(rec["seq_len_q"]),
                    seq_len_kv=int(rec.get("seq_len_kv", rec["seq_len_q"])),
                    head_dim=int(rec["head_dim"]),
                    causal=bool(rec.get("causal", False)),
                    dtype=str(rec.get("dtype") or default_dtype),
                    sm_scale=rec.get("sm_scale"),
                )
                if shape not in seen:
                    seen.add(shape)
                    shapes.append(shape)
    return shapes


@dataclasses.dataclass(frozen=True)
class LaneSpec:
    name: str
    exact_cap: tuple[int, int] | None = None  # kernel runs on exactly this SM
    caps: tuple[tuple[int, int], ...] | None = None  # kernel runs only on these SMs
    min_cap: tuple[int, int] | None = None
    head_dims: tuple[int, ...] | None = None  # None = any
    same_shape: bool = False  # requires q/k/v equal shapes (self-attention)
    same_heads: bool = False  # requires num_q_heads == num_kv_heads


def skip_reason(spec: LaneSpec, shape: Shape, cap: tuple[int, int] | None,
                unavailable: str | None) -> str | None:
    """Why this lane cannot run this shape (None = it can)."""
    if unavailable:
        return unavailable
    if cap is None:
        return "no CUDA device"
    if spec.exact_cap is not None and cap != spec.exact_cap:
        return (f"requires SM {spec.exact_cap[0]}.{spec.exact_cap[1]}, "
                f"device is SM {cap[0]}.{cap[1]}")
    if spec.caps is not None and cap not in spec.caps:
        allowed = "/".join(f"{a}.{b}" for a, b in spec.caps)
        return f"requires SM {allowed}, device is SM {cap[0]}.{cap[1]}"
    if spec.min_cap is not None and cap < spec.min_cap:
        return (f"requires SM >= {spec.min_cap[0]}.{spec.min_cap[1]}, "
                f"device is SM {cap[0]}.{cap[1]}")
    if spec.head_dims is not None and shape.head_dim not in spec.head_dims:
        return f"head_dim {shape.head_dim} not in {list(spec.head_dims)}"
    if spec.same_shape and (shape.seq_len_q != shape.seq_len_kv or shape.num_q_heads != shape.num_kv_heads):
        return "requires equal q/k/v shapes (self-attention)"
    if spec.same_heads and shape.num_q_heads != shape.num_kv_heads:
        return "requires num_q_heads == num_kv_heads"
    return None


def shape_workspace_bytes(shape: Shape, dtype_bytes: int, ref_chunk: int) -> int:
    """Rough peak device memory to bench one shape: q/k/v in the bench dtype,
    their fp32 copies + fp32 reference output for the ground truth, and the
    fp32 score-matrix chunk (x2 for the math-backend softmax temporary)."""
    numel_q = shape.batch * shape.num_q_heads * shape.seq_len_q * shape.head_dim
    numel_kv = shape.batch * shape.num_kv_heads * shape.seq_len_kv * shape.head_dim
    lq = shape.seq_len_q if shape.causal else min(ref_chunk, shape.seq_len_q)
    scores = shape.batch * shape.num_q_heads * lq * shape.seq_len_kv * 4
    return (numel_q + 2 * numel_kv) * (dtype_bytes + 4) + numel_q * 4 + 2 * scores


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Aligned plain-text table (left-justified columns)."""
    cols = list(zip(headers, *rows)) if rows else [(h,) for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]
    lines = [
        "  ".join(str(h).ljust(w) for h, w in zip(headers, widths)),
        "  ".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lanes (GPU)
# ---------------------------------------------------------------------------

AttnFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool, float], torch.Tensor]

_REF_CHUNK = 2048


@dataclasses.dataclass
class Lane:
    spec: LaneSpec
    fn: AttnFn | None  # None when unavailable
    unavailable: str | None = None


def _run_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float) -> torch.Tensor:
    kwargs: dict = {"is_causal": causal, "scale": scale}
    if q.shape[1] != k.shape[1]:
        kwargs["enable_gqa"] = True
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, **kwargs)


def _ref_sdpa_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float) -> torch.Tensor:
    """fp32 SDPA ground truth, chunked over query length.

    Where flash/mem-efficient SDPA reject fp32 (e.g. SM121), torch falls back
    to the math backend, which materializes the full [B, H, Lq, Lkv] score
    matrix -- ~48 GiB for the 32760-token default shape (host OOM-kill on
    unified-memory devices like GB10). Non-causal attention is exact under
    query chunking; causal shapes stay unchunked (per-chunk is_causal would
    mis-align the mask), so they keep the old memory ceiling.
    """
    if causal or q.shape[2] <= _REF_CHUNK:
        return _run_sdpa(q, k, v, causal, scale)
    return torch.cat([_run_sdpa(q[:, :, i:i + _REF_CHUNK], k, v, False, scale)
                      for i in range(0, q.shape[2], _REF_CHUNK)], dim=2)


def _load_baseline(dtype_tag: str) -> Lane:
    """High-precision baseline: flash-attn when importable, else torch SDPA.

    The lane name records which implementation is actually timed.
    """
    try:
        from flash_attn import flash_attn_func  # noqa: PLC0415
    except ImportError:
        return Lane(LaneSpec(f"{dtype_tag}_sdpa"), _run_sdpa)

    def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float) -> torch.Tensor:
        # flash_attn_func takes [B, L, H, D]; only the head dim must be
        # contiguous, so plain transposes are accepted without a copy.
        out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                              softmax_scale=scale, causal=causal)
        return out.transpose(1, 2)

    # flash-attn's fp16/bf16 kernels need sm80+.
    return Lane(LaneSpec(f"{dtype_tag}_flash_attn", min_cap=(8, 0)), fn)


def _load_fp8_sageattn() -> Lane:
    # sageattn's fp8 (qk-int8 / pv-fp8) kernels need sm89+; the package
    # dispatches per-arch internally.
    spec = LaneSpec("fp8_sageattn", min_cap=(8, 9), same_heads=True)
    try:
        from sageattention import sageattn  # noqa: PLC0415
    except ImportError as e:
        return Lane(spec, None, f"sageattention not installed ({e})")

    def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float) -> torch.Tensor:
        return sageattn(q, k, v, tensor_layout="HND", is_causal=causal, sm_scale=scale)

    return Lane(spec, fn)


def _load_nvfp4_local(single_level_p_quant: bool) -> Lane:
    spec = LaneSpec("nvfp4_local", exact_cap=(12, 0), head_dims=(64, 128), same_shape=True)
    for p in (_REPO_ROOT, _REPO_ROOT / "fastvideo-kernel", _REPO_ROOT / "fastvideo-kernel" / "python"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from attn_qat_infer import sageattn_blackwell  # noqa: PLC0415
    except ImportError as e:
        return Lane(spec, None, f"attn_qat_infer import failed ({e}); build fastvideo-kernel with sm_120a")

    def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float) -> torch.Tensor:
        # sageattn_blackwell pads L to a multiple of 128 internally and trims.
        return sageattn_blackwell(q, k, v, is_causal=causal,
                                  single_level_p_quant=single_level_p_quant, sm_scale=scale)

    return Lane(spec, fn)


def _load_nvfp4_flashinfer() -> Lane:
    # SM121 (GB10 / DGX Spark) is enabled by flashinfer#3897.
    spec = LaneSpec("nvfp4_flashinfer", caps=((12, 0), (12, 1)), head_dims=(64, 128), same_shape=True)
    try:
        from flashinfer import (  # noqa: PLC0415
            nvfp4_attention_sm120_fwd, nvfp4_attention_sm120_quantize_qkv)
    except ImportError as e:
        return Lane(spec, None, f"flashinfer nvfp4_attention_sm120 not installed ({e})")

    quant_kwargs = {}
    if "per_block_mean" in inspect.signature(nvfp4_attention_sm120_quantize_qkv).parameters:
        quant_kwargs["per_block_mean"] = True

    def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float) -> torch.Tensor:
        seq_len = q.shape[2]
        # flashinfer's per-block-mean quantize (flashinfer#3897 _preprocess_qkv)
        # materializes an fp32 [B, H, Lpad, Lpad] qk-correction matrix (48 GiB at
        # L=32760, 274 GiB at L=75600). Preflight it so an oversized shape becomes
        # a "fail" row instead of the host OOM-killer SIGKILLing the whole bench
        # (observed on GB10 unified memory).
        lpad = -(-seq_len // 128) * 128
        need = 4 * q.shape[0] * q.shape[1] * lpad * lpad
        free = torch.cuda.mem_get_info(q.device)[0]
        if need > free:
            raise RuntimeError(f"qk_correction workspace {need / 2**30:.0f} GiB exceeds "
                               f"free device memory {free / 2**30:.0f} GiB")
        packed = nvfp4_attention_sm120_quantize_qkv(q, k, v, **quant_kwargs)
        out, _lse = nvfp4_attention_sm120_fwd(*packed, sm_scale=scale, causal=causal, out_dtype=q.dtype)
        # flashinfer pads seq_len to a multiple of 128; trim it back off.
        return out[:, :, :seq_len, :]

    return Lane(spec, fn)


def build_lanes(args: argparse.Namespace) -> list[Lane]:
    return [
        _load_baseline(args.dtype),
        _load_fp8_sageattn(),
        _load_nvfp4_local(single_level_p_quant=args.p_quant == "single"),
        _load_nvfp4_flashinfer(),
    ]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

_DTYPES = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16,
           "bf16": torch.bfloat16, "fp16": torch.float16}


def make_qkv(shape: Shape, seed: int, bias: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed-seed q/k/v; q and k get a +bias mean shift. v stays zero-mean so
    a constant output offset doesn't inflate the cosine-similarity column."""
    gen = torch.Generator(device="cuda").manual_seed(seed)  # fixed seed per shape
    dtype = _DTYPES.get(shape.dtype, torch.bfloat16)

    def rand(heads: int, seq: int, shift: float) -> torch.Tensor:
        t = torch.randn(shape.batch, heads, seq, shape.head_dim,
                        generator=gen, device="cuda", dtype=dtype)
        return t + shift if shift else t

    return (rand(shape.num_q_heads, shape.seq_len_q, bias),
            rand(shape.num_kv_heads, shape.seq_len_kv, bias),
            rand(shape.num_kv_heads, shape.seq_len_kv, 0.0))


def accuracy_stats(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    out = out.float()
    diff = (out - ref).abs()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    return {
        "cos_sim": cos.item(),
        "max_abs_err": diff.max().item(),
        # mean relative error: mean |out-ref| normalized by the reference's
        # mean magnitude (elementwise |out-ref|/|ref| blows up near ref~0).
        "mean_rel_err": (diff.mean() / ref.abs().mean().clamp_min(1e-12)).item(),
    }


def time_fn(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> dict[str, float]:
    """CUDA-event latency stats (ms) over `iters` timed calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {"mean_ms": statistics.fmean(times), "std_ms": statistics.pstdev(times),
            "median_ms": statistics.median(times)}


def bench_shape(shape: Shape, bias: float, lanes: list[Lane], cap: tuple[int, int] | None,
                args: argparse.Namespace) -> list[dict]:
    dist = f"biased(+{bias:g})" if bias else "zeromean"
    results = []
    runnable = [(lane, skip_reason(lane.spec, shape, cap, lane.unavailable)) for lane in lanes]

    oom_reason = None
    if any(reason is None for _, reason in runnable):
        dtype_bytes = 2  # bf16/fp16
        need = shape_workspace_bytes(shape, dtype_bytes, _REF_CHUNK)
        free = torch.cuda.mem_get_info()[0]
        if need > 0.9 * free:
            oom_reason = (f"would OOM: needs ~{need / 2**30:.1f} GiB workspace, "
                          f"{free / 2**30:.1f} GiB free")
            print(f"  SKIP shape: {oom_reason}")

    q = k = v = ref = None
    if oom_reason is None and any(reason is None for _, reason in runnable):
        q, k, v = make_qkv(shape, args.seed, bias)
        scale = shape.sm_scale or shape.head_dim**-0.5
        # fp32 SDPA ground truth (accuracy baseline only; never timed).
        ref = _ref_sdpa_fp32(q.float(), k.float(), v.float(), shape.causal, scale)

    for lane, reason in runnable:
        rec: dict = {"shape": shape.label(), "dist": dist, "lane": lane.spec.name,
                     **dataclasses.asdict(shape)}
        reason = reason or oom_reason
        if reason is not None:
            rec.update(status="skip", reason=reason)
            results.append(rec)
            continue
        scale = shape.sm_scale or shape.head_dim**-0.5
        try:
            out = lane.fn(q, k, v, shape.causal, scale)
            rec.update(accuracy_stats(out, ref))
            del out
            rec.update(status="ok", **time_fn(lambda: lane.fn(q, k, v, shape.causal, scale),
                                              args.warmup, args.iters))
            rec["tflops"] = attn_tflops(shape, rec["mean_ms"])
        except Exception as e:  # lane failed at runtime: report, keep going
            rec.update(status="fail", reason=f"{type(e).__name__}: {e}")
        results.append(rec)
    del q, k, v, ref
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# The real logged workload (FASTVIDEO_ATTN_SHAPE_LOG on 5090/GB10/GB200 runs of
# Wan2.1-1.3B 480p, 81 frames: latent grid 21x30x52 = 32760 tokens, 12 heads x
# head_dim 128, bf16, non-causal) plus a seq-len sweep at the same head
# geometry. 32760 appears once (deduped).
DEFAULT_SHAPE_DIMS = [
    (1, 12, 1024, 128),
    (1, 12, 4096, 128),
    (1, 12, 16384, 128),
    (1, 12, 32760, 128),  # Wan2.1-1.3B 480p self-attention (the logged shape)
]


def default_shapes(causal: bool, dtype: str) -> list[Shape]:
    return [
        Shape(batch=b, num_q_heads=h, num_kv_heads=h, seq_len_q=length, seq_len_kv=length,
              head_dim=d, causal=causal, dtype=dtype) for b, h, length, d in DEFAULT_SHAPE_DIMS
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--shapes", metavar="BxHxLxD,...",
                   help="comma-separated self-attention shapes; replaces the default suite")
    p.add_argument("--shapes-jsonl", nargs="+", metavar="PATH",
                   help="attention shape log(s) written via FASTVIDEO_ATTN_SHAPE_LOG "
                   "(fastvideo/attention/shape_logger.py); replaces the default shapes")
    p.add_argument("--causal", action="store_true",
                   help="run --shapes/default shapes causally (jsonl shapes keep their recorded flag)")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                   help="input dtype for default shapes / jsonl records without one")
    p.add_argument("--bias", type=float, default=1.0,
                   help="mean shift added to q and k (biased inputs expose mean-dependent "
                   "quantization bugs that zero-mean randn hides)")
    p.add_argument("--include-zeromean", action="store_true",
                   help="additionally run every shape with zero-mean inputs for contrast")
    p.add_argument("--p-quant", choices=["single", "two"], default="single",
                   help="nvfp4_local P-matrix quantization: single-level or two-level "
                   "(sageattn_blackwell single_level_p_quant knob)")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations per lane")
    p.add_argument("--iters", type=int, default=50, help="timed iterations per lane")
    p.add_argument("--seed", type=int, default=0, help="input seed (fixed per shape)")
    p.add_argument("--json", metavar="PATH", help="also dump results as JSON")
    return p.parse_args(argv)


def _fmt(rec: dict, key: str, spec: str) -> str:
    return format(rec[key], spec) if key in rec else "-"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dtype_name = f"torch.{'bfloat16' if args.dtype == 'bf16' else 'float16'}"
    if args.shapes:
        shapes = parse_shapes_cli(args.shapes, args.causal, dtype_name)
    elif args.shapes_jsonl:
        shapes = parse_shapes_jsonl(args.shapes_jsonl, dtype_name)
    else:
        shapes = default_shapes(args.causal, dtype_name)

    cap = None
    meta: dict = {"torch": torch.__version__, "warmup": args.warmup, "iters": args.iters,
                  "seed": args.seed, "bias": args.bias}
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        meta.update(device=torch.cuda.get_device_name(0), sm=f"{cap[0]}.{cap[1]}")
        print(f"Device: {meta['device']} (SM {meta['sm']}), torch {torch.__version__}")
    else:
        print("No CUDA device: all lanes will be skipped (shape/skip dry run).")

    lanes = build_lanes(args)
    biases = [args.bias] + ([0.0] if args.include_zeromean and args.bias else [])
    results = []
    for shape in shapes:
        for bias in biases:
            print(f"\n== {shape.label()}  dtype={shape.dtype}  bias={bias:g} ==")
            results.extend(bench_shape(shape, bias, lanes, cap, args))

    headers = ["shape", "dist", "lane", "status", "mean_ms", "std_ms", "TFLOPS",
               "cos_sim", "max_abs_err", "mean_rel_err"]
    rows = []
    for rec in results:
        status = rec["status"] if rec["status"] == "ok" else f"{rec['status']}: {rec['reason']}"
        rows.append([
            rec["shape"], rec["dist"], rec["lane"], status,
            _fmt(rec, "mean_ms", ".3f"),
            _fmt(rec, "std_ms", ".3f"),
            _fmt(rec, "tflops", ".1f"),
            _fmt(rec, "cos_sim", ".6f"),
            _fmt(rec, "max_abs_err", ".4e"),
            _fmt(rec, "mean_rel_err", ".4e"),
        ])
    print("\n" + format_table(headers, rows))
    print("\nAccuracy is vs chunked fp32 torch-SDPA ground truth on identical inputs "
          "(the fp32 reference itself is never timed).")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
