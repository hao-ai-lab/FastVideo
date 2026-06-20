"""TaskType — declared, never inferred.

The task is part of the typed request and the program branches on it. Heuristics
may only suggest a default at the product boundary; the runtime never infers it.
"""
from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    # --- wired in phase 1 (Wan2.1 / LTX2) ---
    T2V = "t2v"  # text -> video
    I2V = "i2v"  # image -> video
    TI2V = "ti2v"  # text+image -> video
    V2V = "v2v"  # video -> video
    T2I = "t2i"  # text -> image (batchable diffusion)

    # --- declared now for omni-readiness; implemented in phase 2 ---
    T2A = "t2a"  # text -> audio (diffusion)
    T2VS = "t2vs"  # text -> video+sound (joint denoise)
    V2W = "v2w"  # video -> world rollout (chunked, causal)
    A2W = "a2w"  # action -> world rollout
    REASON = "reason"  # AR text reasoning (Cosmos3 reasoner / omni thinker)

    @property
    def is_world_rollout(self) -> bool:
        return self in (TaskType.V2W, TaskType.A2W)

    @property
    def is_autoregressive(self) -> bool:
        return self in (TaskType.REASON, )
