"""The batch-of-N interleave parity gate — required, not optional.

Loop inversion's real hazard is cross-request state smearing under interleaving, which a
batch-of-1 parity gate cannot detect. So this gate runs two or more concurrent requests
interleaved at step granularity and requires bit-identical output versus running them serially.

The gate is pure and duck-typed: it takes any engine exposing ``run_serial`` and ``run_interleaved``.
"""
from __future__ import annotations

from typing import Any

from v2.request.artifacts import (
    Output, )
from v2.parity.ladder import ConsistencyLevel, Divergence, bit_identical


def _artifact_payload(art: Any) -> Any:
    for attr in ("frames", "samples", "tensor", "latent", "token_ids", "text"):
        if hasattr(art, attr) and getattr(art, attr) is not None:
            return getattr(art, attr)
    return None


def compare_outputs(serial: dict[str, Output], interleaved: dict[str, Output]) -> list[Divergence]:
    """Bitwise-compare per-request outputs from serial vs interleaved runs."""
    divs: list[Divergence] = []
    if set(serial) != set(interleaved):
        divs.append(
            Divergence("request_set", ConsistencyLevel.C1, float("inf"), float("inf"),
                       f"serial reqs {set(serial)} != interleaved {set(interleaved)}"))
        return divs
    for rid in sorted(serial):
        so, io = serial[rid], interleaved[rid]
        if set(so.artifacts) != set(io.artifacts):
            divs.append(
                Divergence(f"{rid}:artifacts", ConsistencyLevel.C1, float("inf"), float("inf"),
                           f"artifact names differ: {set(so.artifacts)} vs {set(io.artifacts)}"))
            continue
        for name in sorted(so.artifacts):
            a, b = _artifact_payload(so.artifacts[name]), _artifact_payload(io.artifacts[name])
            if a is None and b is None:
                # both empty is NOT parity — a deferred/aborted request must not pass the gate vacuously
                divs.append(
                    Divergence(
                        f"{rid}:{name}", ConsistencyLevel.C1, float("inf"), float("inf"),
                        "both serial and interleaved produced EMPTY output for a declared "
                        "artifact (deferred/aborted?) — suspicious, not parity"))
                continue
            if not bit_identical(a, b):
                divs.append(
                    Divergence(f"{rid}:{name}", ConsistencyLevel.C1, float("nan"), float("nan"),
                               "serial vs interleaved output not bit-identical "
                               "(cross-request state smearing!)"))
    return divs


def assert_interleave_parity(engine: Any, requests: list[Any]) -> list[Divergence]:
    """Drive ``engine`` both ways and return divergences (empty list == gate PASSES)."""
    serial = engine.run_serial(requests)
    interleaved = engine.run_interleaved(requests)
    return compare_outputs(serial, interleaved)
