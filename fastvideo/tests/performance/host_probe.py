# SPDX-License-Identifier: Apache-2.0
"""Host CPU health probe for the performance lane.

The perf lane runs on shared Modal hosts. A packed host slows CPU-bound
pipeline work by ~1.6-1.8x (observed: text encoder pinned at 3.595s vs the
2.03s healthy cluster, DiT +58-80%) while GPU telemetry stays healthy, which
produces false REGRESSION verdicts against PRs. CPU reservations are a floor,
not isolation, so the comparator instead refuses to gate on measurements
taken on a degraded host (review r30, option a).

Two fixed-work, wall-clock-timed sub-probes (~1.5s total), run once per
pytest process before the first benchmark:

- single-thread: a pure-Python arithmetic loop. Captures per-core slowdown
  (CPU steal, scheduling delay, frequency/cache pressure). This is the axis
  ``host_cpu_score`` gates on: it directly mirrors the packed-host signature
  (single-stream stage wall time), and healthy lane hosts cluster tightly.
- multi-thread: torch float32 matmuls pinned to the lane's CPU reservation
  (cpu=8 in fastvideo/tests/modal/pr_test.py). Recorded as informational
  telemetry only: healthy-pool throughput spans 582-1010 GFLOP/s across host
  SKUs (1.74x), wider than the packed-host signal, so it cannot gate.

Calibration basis (2026-07-07, fastvideo-dev:latest image /opt/venv
python 3.10.19 + torch 2.9.1, five Modal L40S:2 cpu=8 memory=32GiB
containers — the exact perf-lane reservation): four hosts (cpu_count=21)
scored 19.0-20.1k kops/s single-thread, one faster host (cpu_count=24)
scored 22.9k. Reference = 19700 (common-SKU median), so healthy samples
score 0.97-1.17. The packed-host signature (r30) inflates CPU-bound wall
time 1.58-1.77x, i.e. throughput drops to 0.56-0.63x: 0.54-0.61 on the
common SKU, at most ~0.73 on the fast SKU. The comparator's default floor
of 0.75 (PERF_HOST_CPU_MIN_SCORE) sits above every packed projection and
below every healthy sample. Recalibrate the reference after an image /
python / torch bump via PERF_HOST_CPU_ST_REF_KOPS (a stale reference can
only skip the gate loudly, never fail a PR).
"""

import os
import time

_DEFAULT_ST_REF_KOPS = 19700.0

_PROBE_THREADS = 8  # matches the perf lane's cpu=8.0 Modal reservation
_ST_ITERS = 20_000_000  # ~1.0s single-threaded on a healthy lane host
_MT_N = 2048
_MT_REPS = 6


def _single_thread_kops(iters: int = _ST_ITERS) -> float:
    """Fixed pure-Python workload; returns kilo-iterations per second."""
    start = time.perf_counter()
    acc = 0
    for i in range(iters):
        acc += i * i
    elapsed = time.perf_counter() - start
    return iters / elapsed / 1e3


def _multi_thread_gflops(n: int = _MT_N, reps: int = _MT_REPS) -> float:
    """Fixed torch float32 matmul workload on the lane's CPU reservation."""
    import torch
    old_threads = torch.get_num_threads()
    torch.set_num_threads(min(_PROBE_THREADS, os.cpu_count() or _PROBE_THREADS))
    try:
        a = torch.rand(n, n, dtype=torch.float32)
        b = torch.rand(n, n, dtype=torch.float32)
        a @ b  # warmup: thread-pool spin-up and first-touch page faults
        start = time.perf_counter()
        for _ in range(reps):
            a @ b
        elapsed = time.perf_counter() - start
    finally:
        torch.set_num_threads(old_threads)
    return reps * 2 * n**3 / elapsed / 1e9


def measure_host_cpu_profile() -> dict[str, float]:
    """Run both sub-probes and return the normalized host CPU profile.

    ``host_cpu_score`` is ~1.0 on a healthy perf-lane host (single-thread
    axis); the raw sub-probe throughputs are kept alongside for auditing
    and recalibration.
    """
    st_ref = float(os.environ.get("PERF_HOST_CPU_ST_REF_KOPS", _DEFAULT_ST_REF_KOPS))
    st = _single_thread_kops()
    mt = _multi_thread_gflops()
    return {
        "host_cpu_score": round(st / st_ref, 3),
        "host_cpu_single_thread_kops": round(st, 1),
        "host_cpu_multi_thread_gflops": round(mt, 1),
    }


if __name__ == "__main__":
    # Recalibration helper: run this on a perf-lane container (Modal L40S:2,
    # cpu=8) a few times, take healthy-cluster medians, update the refs.
    print(measure_host_cpu_profile())
