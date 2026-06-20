"""Parallelism plane — named axes -> validated mesh, part of the cache key."""
from __future__ import annotations

from v2.core.parallel.mesh import FakeDeviceMesh, build_mesh
from v2.core.parallel.plan import AXIS_NAMES, ParallelPlan
from v2.core.parallel.validation import ParallelValidationError, validate_plan

__all__ = ["ParallelPlan", "AXIS_NAMES", "FakeDeviceMesh", "build_mesh", "validate_plan", "ParallelValidationError"]
