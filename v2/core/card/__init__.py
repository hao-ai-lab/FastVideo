"""Model Plane — the (recipe, runtime) pair as a typed card."""
from __future__ import annotations

from v2.core.enums import Capability, ConsistencyLevel, ExecutionProfile, LoopKind, WorkUnitKind
from v2.core.card.instance import ModelInstance, load_card
from v2.core.card.specs import (
    CacheContract,
    CapabilityMatrix,
    CardValidationError,
    CheckpointManifest,
    ComponentSpec,
    DataRef,
    LoopSpec,
    ModelCard,
    ParallelismContract,
    ParitySpec,
    ParityTestSpec,
    PrecisionContract,
    RecipeSpec,
    SamplingDefaults,
)

__all__ = [
    "ModelCard",
    "ComponentSpec",
    "LoopSpec",
    "RecipeSpec",
    "ParitySpec",
    "ParityTestSpec",
    "CheckpointManifest",
    "CapabilityMatrix",
    "CacheContract",
    "SamplingDefaults",
    "ParallelismContract",
    "PrecisionContract",
    "DataRef",
    "CardValidationError",
    "ModelInstance",
    "load_card",
    "Capability",
    "ConsistencyLevel",
    "ExecutionProfile",
    "LoopKind",
    "WorkUnitKind",
]
