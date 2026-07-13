# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the pure parts of
fastvideo-kernel/benchmarks/bench_attn_kernels.py (flops formula, shape
parsing, lane-skip logic, table formatting). GPU lanes are exercised
manually on real hardware."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "fastvideo-kernel" / "benchmarks" / "bench_attn_kernels.py"

spec = importlib.util.spec_from_file_location("bench_attn_kernels", _SCRIPT)
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench  # dataclasses resolve annotations via sys.modules
spec.loader.exec_module(bench)


def _shape(**overrides):
    base = dict(batch=1, num_q_heads=12, num_kv_heads=12, seq_len_q=1024,
                seq_len_kv=1024, head_dim=128, causal=False)
    base.update(overrides)
    return bench.Shape(**base)


def test_attn_tflops_formula():
    # 4*B*H*Lq*Lk*D FLOPs at 1 ms.
    shape = _shape(batch=2, num_q_heads=3, seq_len_q=100, seq_len_kv=200, head_dim=64)
    expected = 4 * 2 * 3 * 100 * 200 * 64 / 1e9  # TFLOPS at 1 ms
    assert bench.attn_tflops(shape, 1.0) == pytest.approx(expected)
    # Causal halves the FLOPs; 2 ms halves the rate again.
    causal = _shape(batch=2, num_q_heads=3, seq_len_q=100, seq_len_kv=200, head_dim=64, causal=True)
    assert bench.attn_tflops(causal, 2.0) == pytest.approx(expected / 4)


def test_parse_shapes_jsonl_dedupe_and_defaults(tmp_path):
    rec = {"backend": "FLASHINFER_SAGE_ATTN3", "batch": 1, "num_q_heads": 12,
           "num_kv_heads": 12, "seq_len_q": 75600, "seq_len_kv": 75600,
           "head_dim": 128, "dtype": "torch.bfloat16", "causal": False,
           "sm_scale": 0.0884, "count": 1200, "torch": "2.8.0",
           "device": "NVIDIA GB10", "sm": "12.1"}
    minimal = {"batch": 2, "num_q_heads": 8, "seq_len_q": 512, "head_dim": 64}
    path = tmp_path / "shapes.jsonl"
    path.write_text(json.dumps(rec) + "\n\n" + json.dumps(rec) + "\n" + json.dumps(minimal) + "\n")

    shapes = bench.parse_shapes_jsonl([str(path)], default_dtype="torch.float16")
    assert len(shapes) == 2  # duplicate line deduped
    full, mini = shapes
    assert (full.batch, full.num_q_heads, full.seq_len_q, full.head_dim) == (1, 12, 75600, 128)
    assert full.sm_scale == pytest.approx(0.0884)
    assert full.dtype == "torch.bfloat16"
    # Missing keys fall back: kv defaults to q, dtype to the CLI default.
    assert (mini.num_kv_heads, mini.seq_len_kv) == (8, 512)
    assert mini.dtype == "torch.float16"
    assert mini.causal is False and mini.sm_scale is None


def test_skip_reason():
    sm120 = bench.LaneSpec("nvfp4", exact_cap=(12, 0), head_dims=(64, 128), same_shape=True)
    ok = _shape()
    assert bench.skip_reason(sm120, ok, (12, 0), None) is None
    assert "not installed" in bench.skip_reason(sm120, ok, (12, 0), "not installed")
    assert "no CUDA" in bench.skip_reason(sm120, ok, None, None)
    assert "SM 12.0" in bench.skip_reason(sm120, ok, (10, 0), None)  # GB200
    assert "SM 12.0" in bench.skip_reason(sm120, ok, (12, 1), None)  # Spark
    assert "head_dim 96" in bench.skip_reason(sm120, _shape(head_dim=96), (12, 0), None)
    cross = _shape(seq_len_kv=256)
    assert "equal q/k/v" in bench.skip_reason(sm120, cross, (12, 0), None)

    fp8 = bench.LaneSpec("fp8", min_cap=(8, 9), same_heads=True)
    assert bench.skip_reason(fp8, ok, (9, 0), None) is None
    assert bench.skip_reason(fp8, cross, (12, 0), None) is None  # cross-attn ok
    assert "SM >= 8.9" in bench.skip_reason(fp8, ok, (8, 6), None)
    assert "num_q_heads" in bench.skip_reason(fp8, _shape(num_kv_heads=4), (9, 0), None)

    anywhere = bench.LaneSpec("sdpa_bf16")
    assert bench.skip_reason(anywhere, cross, (7, 5), None) is None


def test_format_table_alignment():
    headers = ["shape", "lane", "TFLOPS"]
    rows = [["1x12x75600x128", "nvfp4_local", "812.3"],
            ["1x8x512x64", "sdpa_bf16", "9.1"]]
    table = bench.format_table(headers, rows)
    lines = table.splitlines()
    assert len(lines) == 4  # header, rule, 2 rows
    assert len({len(line) for line in lines}) == 1  # columns aligned
    assert "nvfp4_local" in lines[2] and "sdpa_bf16" in lines[3]
    assert bench.format_table(headers, []).splitlines()[0].startswith("shape")


def test_cli_defaults_and_shapes():
    args = bench.parse_args([])
    assert (args.warmup, args.iters, args.p_quant) == (10, 50, "single")
    assert args.bias == 1.0 and not args.include_zeromean  # biased inputs by default
    shapes = bench.default_shapes(causal=False, dtype="torch.bfloat16")
    assert bench.Shape(1, 12, 12, 32760, 32760, 128, False) in shapes  # Wan2.1 480p
    assert len(shapes) == len(set(shapes))  # sweep + logged shapes deduped
    with pytest.raises(SystemExit) as exc:
        bench.parse_args(["--help"])
    assert exc.value.code == 0


def test_parse_shapes_cli():
    shapes = bench.parse_shapes_cli("1x12x32760x128,2x24x512x64", causal=True,
                                    dtype="torch.float16")
    assert shapes == [
        bench.Shape(1, 12, 12, 32760, 32760, 128, True, "torch.float16"),
        bench.Shape(2, 24, 24, 512, 512, 64, True, "torch.float16"),
    ]
    with pytest.raises(Exception, match="BxHxLxD"):
        bench.parse_shapes_cli("1x12x128", causal=False, dtype="torch.bfloat16")


def test_shape_workspace_bytes_scales_with_seq():
    small = bench.shape_workspace_bytes(_shape(seq_len_q=1024, seq_len_kv=1024), 2, 2048)
    big = bench.shape_workspace_bytes(_shape(seq_len_q=32760, seq_len_kv=32760), 2, 2048)
    assert big > small > 0
