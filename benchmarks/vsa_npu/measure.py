# SPDX-License-Identifier: Apache-2.0
"""Frozen synchronized-wall/event timing and warm-stability protocol."""
import os
from pathlib import Path
import statistics
import time
import torch
from collections.abc import Callable
from typing import Any


def timed(fn: Callable[[], torch.Tensor], event: bool = False) -> tuple[torch.Tensor, float, float | None]:
    torch.npu.synchronize()
    if event:
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
    t = time.perf_counter()
    result = fn()
    if event:
        end.record()
    torch.npu.synchronize()
    wall = (time.perf_counter() - t) * 1000
    return result, wall, begin.elapsed_time(end) if event else None


def jit_count() -> int:
    return len(list(Path(os.environ['TRITON_CACHE_DIR']).rglob('*.npubin')))


def benchmark(fn: Callable[[], torch.Tensor], samples: int = 30) -> dict[str, Any]:
    warms = []
    stable = False
    counts = []
    for i in range(50):
        _, ms, _ = timed(fn)
        warms.append(ms)
        counts.append(jit_count())
        if len(warms) >= 10 and len(warms) % 5 == 0:
            old = statistics.median(warms[-10:-5])
            new = statistics.median(warms[-5:])
            if abs(new / old - 1) <= .02 and counts[-1] == counts[-10]:
                stable = True
                break
    walls = []
    events = []
    for _ in range(samples):
        _, w, e = timed(fn, event=True)
        walls.append(w)
        events.append(e)
    return {
        'warmup_ms': warms,
        'stable': stable,
        'warm_wall_ms_samples': walls,
        'warm_device_ms_samples': events,
        'jit_count_before_after': [counts[-1], jit_count()]
    }
