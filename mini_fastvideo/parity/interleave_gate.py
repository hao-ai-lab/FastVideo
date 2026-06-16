"""The batch-of-N interleave parity gate (design_v3 §9.3) — non-negotiable.

> Loop inversion's real hazard is cross-request state smearing under interleaving — and a
> batch-of-1 parity gate is structurally blind to it. So v3 makes a batch-of-N interleave
> parity test a *required* gate: two (or more) concurrent requests, interleaved at step
> granularity, must be bit-identical to the same requests run serially.

This is the test the whole loop-inversion bet lives or dies on. The gate is pure and
duck-typed: it takes any engine exposing ``run_serial`` and ``run_interleaved``.
"""
from __future__ import annotations

from typing import Any

from ..request.artifacts import (
    AudioArtifact,
    LatentArtifact,
    Output,
    TensorArtifact,
    TextArtifact,
    VideoArtifact,
)
from .ladder import ConsistencyLevel, Divergence, bit_identical


def _artifact_payload(art: Any) -> Any:
    for attr in ("frames", "samples", "tensor", "latent", "token_ids", "text"):
        if hasattr(art, attr) and getattr(art, attr) is not None:
            return getattr(art, attr)
    return None


def compare_outputs(serial: dict[str, Output], interleaved: dict[str, Output]) -> list[Divergence]:
    """Bitwise-compare per-request outputs from serial vs interleaved runs."""
    divs: list[Divergence] = []
    if set(serial) != set(interleaved):
        divs.append(Divergence("request_set", ConsistencyLevel.C1, float("inf"), float("inf"),
                               f"serial reqs {set(serial)} != interleaved {set(interleaved)}"))
        return divs
    for rid in sorted(serial):
        so, io = serial[rid], interleaved[rid]
        if set(so.artifacts) != set(io.artifacts):
            divs.append(Divergence(f"{rid}:artifacts", ConsistencyLevel.C1, float("inf"), float("inf"),
                                   f"artifact names differ: {set(so.artifacts)} vs {set(io.artifacts)}"))
            continue
        for name in sorted(so.artifacts):
            a, b = _artifact_payload(so.artifacts[name]), _artifact_payload(io.artifacts[name])
            if not bit_identical(a, b):
                divs.append(Divergence(f"{rid}:{name}", ConsistencyLevel.C1, float("nan"), float("nan"),
                                       "serial vs interleaved output not bit-identical "
                                       "(cross-request state smearing!)"))
    return divs


def assert_interleave_parity(engine: Any, requests: list[Any]) -> list[Divergence]:
    """Drive ``engine`` both ways and return divergences (empty list == gate PASSES)."""
    serial = engine.run_serial(requests)
    interleaved = engine.run_interleaved(requests)
    return compare_outputs(serial, interleaved)
