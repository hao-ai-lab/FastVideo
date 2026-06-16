"""Parallelism plane (design_v3 §8) — named axes → validated mesh, part of the cache key."""
from __future__ import annotations

from .mesh import FakeDeviceMesh, build_mesh
from .plan import AXIS_NAMES, ParallelPlan
from .validation import ParallelValidationError, validate_plan

__all__ = ["ParallelPlan", "AXIS_NAMES", "FakeDeviceMesh", "build_mesh",
           "validate_plan", "ParallelValidationError"]
