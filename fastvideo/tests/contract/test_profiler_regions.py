# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the profiler region system.

The region system's failure mode is silent: a profiler that records nothing
(the old wait/warmup schedule bug), a trace that never exports (no shutdown
hook), or a typo'd region name that no-ops forever. Each test here runs a
child process — the profiler is process-global and its export happens at
atexit, so in-process tests can't observe the contract.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch

from fastvideo.profiler import nvtx_range

# Five-window child: ops before any region, inside a region, between regions,
# inside a second (short-named) region, after the last region. Exits without
# calling stop() — export must happen via the atexit hook. Each window is
# labeled with a record_function marker so the parent can check which windows
# the trace actually captured.
_CHILD = r"""
import torch
from fastvideo.profiler import get_or_create_profiler, profiler_region

controller = get_or_create_profiler("{trace_dir}")

def burn(tag):
    with torch.profiler.record_function(tag):
        torch.mm(torch.ones(8, 8), torch.ones(8, 8))

burn("win_pre")
with profiler_region("profiler_region_inference_denoising"):
    burn("win_region1")
burn("win_between")
with profiler_region("model_loading"):  # short name must resolve
    burn("win_region2")
for _ in range(2):  # warn-once: second use must not log again
    with profiler_region("definitely_not_a_region"):
        burn("win_typo")
burn("win_post")
# no controller.stop(): atexit owns the export
"""


def _run_child(tmp_path):
    trace_dir = str(tmp_path / "traces")
    env = os.environ.copy()
    env["FASTVIDEO_TORCH_PROFILER_DIR"] = trace_dir
    env["FASTVIDEO_TORCH_PROFILE_REGIONS"] = "inference_denoising,model_loading"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD.format(trace_dir=trace_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    return trace_dir, proc.stdout + proc.stderr


def _trace_event_names(trace_dir):
    traces = glob.glob(os.path.join(trace_dir, "**", "*.json*"), recursive=True)
    traces = [t for t in traces if "summary" not in os.path.basename(t)]
    assert traces, f"no trace exported to {trace_dir} (atexit hook missing?)"
    opener = gzip.open if traces[0].endswith(".gz") else open
    with opener(traces[0], "rt") as fh:
        events = json.load(fh).get("traceEvents", [])
    return {e.get("name", "") for e in events}


def test_regions_gate_collection_and_atexit_exports(tmp_path):
    trace_dir, output = _run_child(tmp_path)
    names = _trace_event_names(trace_dir)

    assert "win_region1" in names, "op inside an enabled region was not captured"
    assert "win_region2" in names, "short region name did not resolve/capture"
    assert "fastvideo.region::profiler_region_inference_denoising" in names

    for leaked in ("win_pre", "win_between", "win_typo", "win_post"):
        assert leaked not in names, f"op outside any region leaked into trace: {leaked}"

    # unregistered region warns exactly once across repeated uses
    assert output.count("definitely_not_a_region") >= 1
    assert output.count("is not registered") == 1

    # per-rank op summary written next to the trace
    summaries = glob.glob(os.path.join(trace_dir, "summary_rank0.*"))
    assert sorted(os.path.splitext(s)[1] for s in summaries) == [".json", ".txt"]


def test_noop_without_profiler_dir(tmp_path):
    # profiler_region must be a clean no-op when profiling is not configured
    child = (
        "from fastvideo.profiler import profiler_region\n"
        "with profiler_region('inference_denoising'):\n"
        "    x = 1\n"
        "assert x == 1\n"
    )
    env = os.environ.copy()
    env.pop("FASTVIDEO_TORCH_PROFILER_DIR", None)
    env.pop("FASTVIDEO_TORCH_PROFILE_REGIONS", None)
    proc = subprocess.run([sys.executable, "-c", child], env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr


def test_nvtx_range_disabled_is_noop(monkeypatch):
    """Keep CUDA NVTX untouched when external profiling is disabled."""
    range_push = Mock()
    range_pop = Mock()
    monkeypatch.setenv("FASTVIDEO_NVTX_PROFILE", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", range_push)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", range_pop)

    with nvtx_range("disabled"):
        body_executed = True

    assert body_executed is True
    range_push.assert_not_called()
    range_pop.assert_not_called()


def test_nvtx_range_without_cuda_is_noop(monkeypatch):
    """Keep NVTX untouched when profiling is enabled on a CPU-only process."""
    range_push = Mock()
    range_pop = Mock()
    monkeypatch.setenv("FASTVIDEO_NVTX_PROFILE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", range_push)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", range_pop)

    with nvtx_range("cpu-only"):
        body_executed = True

    assert body_executed is True
    range_push.assert_not_called()
    range_pop.assert_not_called()


def test_nvtx_range_enabled_orders_push_body_pop(monkeypatch):
    """Place the profiled body between one matching NVTX push and pop."""
    events = []
    monkeypatch.setenv("FASTVIDEO_NVTX_PROFILE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda name: events.append(("push", name)))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: events.append(("pop", None)))

    with nvtx_range("minimax_h3.test"):
        events.append(("body", None))

    assert events == [
        ("push", "minimax_h3.test"),
        ("body", None),
        ("pop", None),
    ]


def test_nvtx_range_body_exception_pops_and_propagates(monkeypatch):
    """Balance the NVTX stack while preserving a body exception."""
    events = []
    monkeypatch.setenv("FASTVIDEO_NVTX_PROFILE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda name: events.append(("push", name)))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: events.append(("pop", None)))

    with pytest.raises(RuntimeError, match="profile body failed"):
        with nvtx_range("minimax_h3.failure"):
            raise RuntimeError("profile body failed")

    assert events == [
        ("push", "minimax_h3.failure"),
        ("pop", None),
    ]
