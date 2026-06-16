"""BehaviorRecord — captured at generation time on the serving loop (design_v3 §10).

> The rollout forward *is* the serve forward plus capture.

Sized honestly (design_v3 §10): full routing capture is GB/sample for omni, so it is an opt-in
instrument. ``logprobs`` is None for likelihood-free methods (DiffusionNFT) — the C2 split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .._enums import ConsistencyLevel


class TrajectoryStatus(str, Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class BehaviorRecord:
    request_id: str
    model_id: str
    loop_id: str
    weights_version: str = "v0"
    adapter_versions: dict[str, str] = field(default_factory=dict)
    seeds: dict[str, int] = field(default_factory=dict)
    timesteps: list[float] = field(default_factory=list)
    sigmas: list[float] = field(default_factory=list)
    latents: Any = None                       # final clean latents (or refs if too large)
    logprobs: list[float] | None = None       # None for likelihood-free methods (NFT)
    sampled_tokens: list[int] = field(default_factory=list)
    guidance: dict[str, float] = field(default_factory=dict)
    reward_inputs: dict[str, Any] = field(default_factory=dict)
    reward_outputs: dict[str, float] = field(default_factory=dict)
    precision: str = "float32"
    parallel_plan_hash: str = ""
    attention_backend: str = "dense"
    deterministic_flags: dict[str, bool] = field(default_factory=dict)
    consistency_level: ConsistencyLevel = ConsistencyLevel.C1
    status: TrajectoryStatus = TrajectoryStatus.COMPLETED
