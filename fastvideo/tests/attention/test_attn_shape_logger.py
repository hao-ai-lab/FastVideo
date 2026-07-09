# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for fastvideo.attention.shape_logger.

The logger reads FASTVIDEO_ATTN_SHAPE_LOG once at module import, so each test
sets the env var and reloads the module to get a fresh, enabled instance.
"""

from __future__ import annotations

import importlib
import json
import os

import pytest
import torch

import fastvideo.attention.shape_logger as shape_logger_mod


@pytest.fixture(autouse=True)
def _reset_logger_module():
    """Reload the module disabled after each test so the atexit flush never
    targets a deleted tmp_path."""
    yield
    os.environ.pop("FASTVIDEO_ATTN_SHAPE_LOG", None)
    importlib.reload(shape_logger_mod)


def _fresh_logger(monkeypatch, path: str):
    monkeypatch.setenv("FASTVIDEO_ATTN_SHAPE_LOG", str(path))
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    return importlib.reload(shape_logger_mod)


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("FASTVIDEO_ATTN_SHAPE_LOG", raising=False)
    mod = importlib.reload(shape_logger_mod)
    assert mod.enabled is False
    q = torch.zeros(1, 4, 2, 8)
    mod.record("X", q, q, q, causal=False, sm_scale=None)  # no-op, no crash
    assert not mod._entries


def test_dedupe_count_and_dump_roundtrip(monkeypatch, tmp_path):
    log_base = str(tmp_path / "shapes.jsonl")
    mod = _fresh_logger(monkeypatch, log_base)
    assert mod.enabled is True

    q = torch.zeros(2, 16, 4, 64, dtype=torch.float32)  # [B, L, H, D]
    k = torch.zeros(2, 32, 4, 64, dtype=torch.float32)
    for _ in range(3):
        mod.record("FLASHINFER_SAGE_ATTN3", q, k, k, causal=False, sm_scale=0.125)
    # different causal flag -> new entry
    mod.record("FLASHINFER_SAGE_ATTN3", q, k, k, causal=True, sm_scale=0.125)
    # bhsd layout: [B, H, L, D] with same logical dims dedupes onto entry 1
    q_bhsd = q.transpose(1, 2)
    k_bhsd = k.transpose(1, 2)
    mod.record("FLASHINFER_SAGE_ATTN3", q_bhsd, k_bhsd, k_bhsd, causal=False, sm_scale=0.125, layout="bhsd")
    # different backend name -> new entry
    mod.record("SAGE_ATTN_THREE", q, k, k, causal=False, sm_scale=0.125)

    assert len(mod._entries) == 3
    mod._flush_at_exit()

    out_path = f"{log_base}.pid{os.getpid()}"
    entries = _read_jsonl(out_path)
    assert len(entries) == 3

    main = next(e for e in entries if e["backend"] == "FLASHINFER_SAGE_ATTN3" and e["causal"] is False)
    assert main["count"] == 4
    assert main["batch"] == 2
    assert main["num_q_heads"] == 4
    assert main["num_kv_heads"] == 4
    assert main["seq_len_q"] == 16
    assert main["seq_len_kv"] == 32
    assert main["head_dim"] == 64
    assert main["dtype"] == "torch.float32"
    assert main["sm_scale"] == 0.125
    assert main["torch"] == torch.__version__
    assert "device" in main and "sm" in main


def test_periodic_flush_on_new_unique_shapes(monkeypatch, tmp_path):
    log_base = str(tmp_path / "flush.jsonl")
    mod = _fresh_logger(monkeypatch, log_base)
    monkeypatch.setattr(mod, "_FLUSH_EVERY", 4)

    out_path = f"{log_base}.pid{os.getpid()}"
    for i in range(1, 5):
        q = torch.zeros(1, i, 2, 8)
        mod.record("B", q, q, q, causal=False, sm_scale=None)
    # 4th unique shape triggered a flush without any atexit involvement
    assert len(_read_jsonl(out_path)) == 4


def test_rank_suffix_when_distributed(monkeypatch, tmp_path):
    log_base = str(tmp_path / "dist.jsonl")
    mod = _fresh_logger(monkeypatch, log_base)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    q = torch.zeros(1, 4, 2, 8)
    mod.record("B", q, q, q, causal=False, sm_scale=None)
    mod._flush_at_exit()
    assert os.path.exists(f"{log_base}.rank2")
