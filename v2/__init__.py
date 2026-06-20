"""v2 — a scoped, CPU-testable realization of the model-native runtime (see v2/README.md).

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
adapters (see ``v2/platform/backends/``) that are off the test path.
"""
from __future__ import annotations

from v2.core.enums import (
    Capability,
    ConsistencyLevel,
    ExecutionProfile,
    LoopKind,
    WorkUnitKind,
)
from v2.core.card import (
    CapabilityMatrix,
    ComponentSpec,
    LoopSpec,
    ModelCard,
    ModelInstance,
    ParitySpec,
    RecipeSpec,
    load_card,
)
from v2.core.program import ComponentNode, ModelLoopNode, Program, ProgramKind, when_opt, when_task
from v2.core.request import (
    DiffusionParams,
    Output,
    Request,
    SamplingParams,
    Session,
    TaskType,
    make_request,
)
from v2.runtime import AsyncEngine, Engine

__version__ = "0.2.0"

__all__ = [
    "ModelCard",
    "ComponentSpec",
    "LoopSpec",
    "RecipeSpec",
    "ParitySpec",
    "CapabilityMatrix",
    "ModelInstance",
    "load_card",
    "Engine",
    "AsyncEngine",
    "Program",
    "ProgramKind",
    "ComponentNode",
    "ModelLoopNode",
    "when_task",
    "when_opt",
    "Request",
    "Session",
    "Output",
    "make_request",
    "TaskType",
    "SamplingParams",
    "DiffusionParams",
    "LoopKind",
    "WorkUnitKind",
    "ConsistencyLevel",
    "ExecutionProfile",
    "Capability",
    "VideoGenerator",
    "__version__",
]


def __getattr__(name: str):
    # Lazy: the GPU entrypoint imports torch / fastvideo, so resolve it only on access — plain
    # ``import v2`` (and the CPU-only mini) stay torch-free.
    if name == "VideoGenerator":
        from v2.video_generator import VideoGenerator
        return VideoGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
