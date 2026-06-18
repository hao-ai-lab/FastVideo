"""Model Plane — the center (design_v3 §4). The (recipe, runtime) pair as a typed card."""
from __future__ import annotations

from v2._enums import Capability, ConsistencyLevel, ExecutionProfile, LoopKind, WorkUnitKind
from v2.card.instance import ModelInstance, load_card
from v2.card.specs import (
    CacheContract,
    CapabilityMatrix,
    CardValidationError,
    CheckpointManifest,
    ComponentSpec,
    CostModel,
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
    "ModelCard", "ComponentSpec", "LoopSpec", "RecipeSpec", "ParitySpec", "ParityTestSpec",
    "CheckpointManifest", "CapabilityMatrix", "CostModel", "CacheContract", "SamplingDefaults",
    "ParallelismContract", "PrecisionContract", "DataRef", "CardValidationError",
    "ModelInstance", "load_card",
    "Capability", "ConsistencyLevel", "ExecutionProfile", "LoopKind", "WorkUnitKind",
]
