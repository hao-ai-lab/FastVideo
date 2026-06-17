"""Request parameters: AR sampling vs diffusion params + output/capture spec.

design_v3 §12: "AR ``sampling`` vs ``diffusion`` params, an ``OutputSpec``
(requested modalities + streaming + capture flags), and per-node overrides."
design.md §6.1: ``sampling``/``diffusion`` are *defaults*; ``node_params`` override
per graph node (validated against the PipelineSpec's node-parameter schema).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class SamplingParams:
    """AR decode knobs (used by ar_decode loops: reasoner/thinker/talker — phase 2)."""
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop: tuple[str, ...] = ()
    seed: int | None = None


@dataclass(frozen=True)
class DiffusionParams:
    """Denoise knobs. ``guidance`` may be a scalar or per-modality dict (joint A/V)."""
    num_steps: int = 50
    guidance_scale: float = 5.0
    # Per-modality guidance for joint denoise (LTX-2 A/V, Cosmos3 t2vs). Empty => use scalar.
    guidance_per_modality: dict[str, float] = field(default_factory=dict)
    shift: float = 5.0                     # flow-match sigma shift
    sigmas: tuple[float, ...] | None = None  # explicit schedule (distilled few-step)
    negative_prompt: str = ""
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16
    seed: int | None = None
    # FlowGRPO-style RL rollout: use the SDE sampler (with per-step log-prob capture) instead of the
    # deterministic ODE serve sampler (PromptRL/UniRL §6). Default False ⇒ the serve path is unchanged.
    sde_rollout: bool = False
    sde_noise_scale: float = 0.7
    # Active adapters for this request (LoRA / ControlNet ids on the base) — the adapter plane (§9.19).
    # Empty ⇒ the base model, unmodified. Part of the cache key (adapter_versions) so adapted ≠ base.
    adapters: tuple[str, ...] = ()


class CaptureMode(str, Enum):
    """What the loop captures while running (design_v3 §9.4, §10)."""
    NONE = "none"            # serve forward: no-grad, no capture
    BEHAVIOR = "behavior"    # rollout forward: emit BehaviorRecord slices for RL/distill


@dataclass(frozen=True)
class OutputSpec:
    modalities: frozenset[str] = frozenset({"video"})
    # streaming flags per modality: True | chunk size (ms / frames). {} => no streaming.
    stream: dict[str, object] = field(default_factory=dict)
    return_latents: bool = False
    return_trajectory: bool = False        # full denoise trajectory (parity / debug)
    capture: CaptureMode = CaptureMode.NONE

    @property
    def streaming(self) -> bool:
        return bool(self.stream)
