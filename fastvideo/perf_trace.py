# SPDX-License-Identifier: Apache-2.0
"""Campaign-only torch.profiler capture and memory log around one pipeline request.

Everything here is gated by environment variables and is a no-op otherwise:

  FASTVIDEO_PERF_TRACE_DIR=<dir>       export a chrome trace per rank, `<dir>/rank<N>.json`
  FASTVIDEO_PERF_TRACE_REQUESTS=1,2    which requests (0 = the warm-up) to capture, default "2"
  FASTVIDEO_PERF_MEM=1                 log GPU allocated/reserved/peak and host RSS after every stage

Each captured request is wrapped in a `ProfilerStep#<n>` annotation so the view
tools treat one request as one step.
"""
from __future__ import annotations

import contextlib
import os
import resource
import time
from pathlib import Path

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)

_TRACE_DIR = os.environ.get("FASTVIDEO_PERF_TRACE_DIR") or None
_TRACE_REQUESTS = {int(x) for x in os.environ.get("FASTVIDEO_PERF_TRACE_REQUESTS", "2").split(",") if x.strip()}
_MEM = os.environ.get("FASTVIDEO_PERF_MEM", "0") != "0"
_request = -1


def _rank() -> int:
    try:
        from fastvideo.distributed.parallel_state import get_world_rank
        return int(get_world_rank())
    except Exception:
        return int(os.environ.get("RANK", "0") or 0)


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def stage_scope(stage_name: str, active: bool):
    """A record_function span per stage, only while a capture is active."""
    if not active:
        return contextlib.nullcontext()
    return torch.profiler.record_function(f"stage:{stage_name}")


def mem_line(stage_name: str) -> None:
    if not _MEM or not torch.cuda.is_available():
        return
    logger.info("[perfmem] req=%d stage=%s alloc=%.2fGB reserved=%.2fGB peak=%.2fGB rss_max=%.2fGB t=%.3f",
                _request, stage_name, torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9,
                torch.cuda.max_memory_allocated() / 1e9, _rss_gb(), time.perf_counter())


@contextlib.contextmanager
def request_scope():
    """Wrap one pipeline request: counts requests, captures the selected ones."""
    global _request
    _request += 1
    n = _request
    if _MEM and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_line("request_start")
    if _TRACE_DIR is None or n not in _TRACE_REQUESTS:
        yield False
        return
    rank = _rank()
    out = Path(_TRACE_DIR)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("[perftrace] capturing request %d on rank %d -> %s", n, rank, out)
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    )
    prof.start()
    try:
        with torch.profiler.record_function(f"ProfilerStep#{n}"):
            yield True
    finally:
        torch.cuda.synchronize()
        prof.stop()
        path = out / f"rank{rank}.json"
        prof.export_chrome_trace(str(path))
        logger.info("[perftrace] wrote %s", path)
