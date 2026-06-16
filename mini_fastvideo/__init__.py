"""mini-fastvideo — a scoped, CPU-testable realization of the design_v3 model-native runtime.

> A model card is a (recipe, runtime) pair with a parity obligation.
> The model owns loop semantics; the runtime owns loop lifecycle.
> One resident instance runs many loops; one scheduler runs their steps in one currency.
> Caches are correct by key; parity is correct by test; the interleave gate is non-negotiable.
> Training records behavior on the same loops it serves.

Phase 1 supports Wan2.1-1.3B (T2V) and LTX2.3 (2-stage distilled), plus four training
methods on Wan2.1-1.3B (finetuning, DMD2, DiffusionNFT, self-forcing). The spine is
omni-ready (multi-loop ModelInstance, ar_decode/chunk_rollout loop kinds) for the phase-2
Cosmos3 + vllm-omni omni ports.

The core is numpy-only and CPU-testable; heavy Wan/LTX neural forwards become lazy torch
adapters (see ``mini_fastvideo/models/*/components.py``) that are off the test path.
"""
from __future__ import annotations

from ._enums import (
    Capability,
    ConsistencyLevel,
    ExecutionProfile,
    LoopKind,
    WorkUnitKind,
)
from .card import (
    CapabilityMatrix,
    ComponentSpec,
    CostModel,
    LoopSpec,
    ModelCard,
    ModelInstance,
    ParitySpec,
    RecipeSpec,
    load_card,
)
from .program import ComponentNode, ModelLoopNode, Program, ProgramKind, when_opt, when_task
from .request import (
    DiffusionParams,
    Output,
    Request,
    SamplingParams,
    Session,
    TaskType,
    make_request,
)
from .runtime import AsyncEngine, Engine

__version__ = "0.2.0"

__all__ = [
    "ModelCard", "ComponentSpec", "LoopSpec", "RecipeSpec", "ParitySpec", "CostModel",
    "CapabilityMatrix", "ModelInstance", "load_card",
    "Engine", "AsyncEngine", "Program", "ProgramKind", "ComponentNode", "ModelLoopNode", "when_task", "when_opt",
    "Request", "Session", "Output", "make_request", "TaskType", "SamplingParams", "DiffusionParams",
    "LoopKind", "WorkUnitKind", "ConsistencyLevel", "ExecutionProfile", "Capability",
    "__version__",
]
