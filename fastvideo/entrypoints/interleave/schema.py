# SPDX-License-Identifier: Apache-2.0
"""Schemas for InterleaveThinker-style orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class InterleaveEditRequest(BaseModel):
    """Image edit/generation request used by Interleave orchestration backends.

    InterleaveThinker sends `num_inference_step` while FastVideo uses
    `num_inference_steps`; accept both and let the plural form win when both are
    provided. Unknown fields are tolerated so model-specific knobs can pass
    through without forcing every backend to implement them immediately.
    """

    model_config = ConfigDict(extra="allow")

    prompt: str
    image: str | None = None
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    num_inference_step: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    true_cfg_scale: float | None = None
    output_format: Literal["png", "jpeg", "jpg", "webp"] | None = "png"
    enhance_prompt: bool | None = None

    def resolved_num_inference_steps(self) -> int | None:
        return self.num_inference_steps if self.num_inference_steps is not None else self.num_inference_step


class InterleaveEditResponse(BaseModel):
    success: bool
    edited_image: str | None = None
    file_path: str | None = None
    prompt: str | None = None
    inference_time_s: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class GeneratedImage:
    prompt: str
    image_base64: str | None = None
    file_path: str | None = None
    inference_time_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlannedInterleaveStep:
    prompt: str
    name: str | None = None
    input_image_path: str | None = None
    max_attempts: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlannerInput:
    instruction: str
    initial_image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CriticInput:
    step: PlannedInterleaveStep
    attempt_index: int
    generated: GeneratedImage
    previous_image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CriticDecision:
    success: bool
    refine_prompt: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InterleaveAttempt:
    step_index: int
    attempt_index: int
    prompt: str
    generated: GeneratedImage
    decision: CriticDecision | None = None


@dataclass(slots=True)
class InterleaveTrace:
    instruction: str
    attempts: list[InterleaveAttempt]
    final_image: GeneratedImage | None
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)
