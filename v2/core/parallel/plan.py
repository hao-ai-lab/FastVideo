"""ParallelPlan — parallelism as a model contract.

Parallelism is not a launch flag; it affects cache keys, scheduling, transport,
capture, and parity, so it lives on the card. Declarative, validated, compiled to a
mesh via a ParallelDims-style builder (``parallel/mesh.py``). This module is a pure
leaf (no card/runtime imports) so ``card/`` can hold plans without a cycle.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

# Canonical axis names. cfgp is <=2.
AXIS_NAMES = ("dp", "tp", "sp", "cp", "cfgp", "pp_patch", "vae", "ep", "fsdp", "role", "replica")


@dataclass
class ParallelPlan:
    axes: dict[str, int] = field(default_factory=dict)  # e.g. {"tp": 2, "sp": 4, "cfgp": 2}
    mesh_order: list[str] = field(default_factory=list)
    placement: str = "colocated"
    communication: dict[str, str] = field(default_factory=dict)
    # applicability conditions travel with axes
    applicability: dict[str, dict] = field(default_factory=dict)
    per_axis_communication: dict[str, str] = field(default_factory=dict)

    def degree(self, axis: str) -> int:
        return int(self.axes.get(axis, 1))

    def world_size(self) -> int:
        w = 1
        for v in self.axes.values():
            w *= int(v)
        return max(w, 1)

    @property
    def hash(self) -> str:
        """Stable hash — part of the CacheKey (parallel_plan_hash)."""
        payload = json.dumps({"axes": self.axes, "order": self.mesh_order}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @classmethod
    def single(cls) -> ParallelPlan:
        """The default deployment: one device, all degree-one trivial groups."""
        return cls(axes={"dp": 1}, mesh_order=["dp"])
