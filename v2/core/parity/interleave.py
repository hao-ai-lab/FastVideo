"""Interleave parity helper for examples and smoke checks."""
from __future__ import annotations

from typing import Any

from v2.core.parity.compare import compare_outputs
from v2.core.parity.ladder import Divergence
from v2.core.request.artifacts import Output
from v2.runtime.scheduler import AdmissionInfeasible


def run_interleaved(engine: Any, requests: list[Any]) -> dict[str, Output]:
    """Drive several requests one engine tick at a time."""
    runners = [engine._make_runner(req) for req in requests]
    stuck = 0
    while not all(r.done for r in runners):
        progress = False
        for runner in runners:
            if runner.done:
                continue
            before = runner._progress
            runner.tick()
            progress = progress or runner.done or runner._progress != before
        if progress:
            stuck = 0
        else:
            stuck += 1
            if stuck > 3:
                raise AdmissionInfeasible("interleaved run made no progress for several ticks")
    return {runner.request.request_id: runner.output() for runner in runners}


def assert_interleave_parity(engine: Any, requests: list[Any]) -> list[Divergence]:
    """Run requests serially and step-interleaved, returning any bitwise output divergences."""
    serial = engine.run_serial(requests)
    interleaved = run_interleaved(engine, requests)
    return compare_outputs(serial, interleaved)
