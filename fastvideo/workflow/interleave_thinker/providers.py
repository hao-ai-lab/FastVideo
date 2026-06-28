# SPDX-License-Identifier: Apache-2.0
"""Model-backed planner and critic providers for Interleave orchestration."""

from __future__ import annotations

from typing import Any

from fastvideo.workflow.interleave_thinker.schema import (
    CriticDecision,
    CriticInput,
    PlannedInterleaveStep,
    PlannerInput,
)
from fastvideo.train.methods.rl.rewards import extract_interleave_answer
from fastvideo.train.models.interleave_thinker import (
    InterleavePlannerStep,
    InterleaveThinkerCriticModel,
    InterleaveThinkerPlannerModel,
)


class InterleaveThinkerPlannerProvider:
    """Adapter from ``InterleaveThinkerPlannerModel`` to ``PlannerProvider``."""

    def __init__(
        self,
        model: InterleaveThinkerPlannerModel,
        *,
        num_generations: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        max_attempts_per_step: int = 2,
    ) -> None:
        self.model = model
        self.num_generations = int(num_generations)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.max_attempts_per_step = int(max_attempts_per_step)

    def plan(
        self,
        request: PlannerInput,
    ) -> list[PlannedInterleaveStep]:
        image_paths = [request.initial_image_path] if request.initial_image_path else []
        raw_plan = self.model.generate_interleave_plan(
            request.instruction,
            input_image_paths=image_paths,
            num_generations=self.num_generations,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
        )
        steps = raw_plan.get("steps") or []
        planned_steps: list[PlannedInterleaveStep] = []
        for idx, step in enumerate(steps):
            if not isinstance(step, InterleavePlannerStep):
                continue
            # ``auxiliary_text`` is a text response channel, not an image prompt.
            # Guidance-planner Task A intentionally leaves both image fields
            # unset, so skip those unsupported text-only steps instead of
            # sending their answer to the image generator.
            prompt = step.prompt or step.instruction or ""
            if not prompt:
                continue
            planned_steps.append(
                PlannedInterleaveStep(
                    prompt=prompt,
                    name=step.step_name,
                    input_image_path=request.initial_image_path if idx == 0 else None,
                    max_attempts=max(1, self.max_attempts_per_step),
                    metadata={
                        "planner_step_number": step.step_number,
                        "planner_instruction": step.instruction,
                        "planner_prompt": step.prompt,
                        "planner_auxiliary_text": step.auxiliary_text,
                        "planner_generation_index": raw_plan.get("generation_index"),
                    },
                ))
        return planned_steps


class InterleaveThinkerCriticProvider:
    """Adapter from ``InterleaveThinkerCriticModel`` to ``CriticProvider``."""

    def __init__(
        self,
        model: InterleaveThinkerCriticModel,
        *,
        num_generations: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
    ) -> None:
        self.model = model
        self.num_generations = int(num_generations)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)

    def review(
        self,
        request: CriticInput,
    ) -> CriticDecision:
        if not request.generated.file_path:
            return CriticDecision(
                success=False,
                reason="InterleaveThinker critic requires generated.file_path",
            )
        item = _critic_item_from_request(request)
        rollouts = self.model.generate_interleave_responses(
            {"items": [item]},
            num_generations=self.num_generations,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
        )
        if not rollouts:
            return CriticDecision(
                success=False,
                reason="InterleaveThinker critic returned no rollouts",
            )
        response = str(rollouts[0].get("response", "") or "")
        parsed = extract_interleave_answer(response)
        if parsed is None:
            return CriticDecision(
                success=False,
                reason="InterleaveThinker critic response did not parse",
                metadata={"critic_response": response},
            )
        return CriticDecision(
            success=parsed.previous_step_success,
            refine_prompt=parsed.refine_prompt,
            metadata={"critic_response": response},
        )


def _critic_item_from_request(request: CriticInput) -> dict[str, Any]:
    instruction = _first_text(
        request.step.metadata.get("planner_instruction"),
        request.step.metadata.get("planner_prompt"),
        request.step.prompt,
    )
    return {
        "origin_prompt": instruction,
        "previous_prompt": request.generated.prompt,
        "previous_image_path": request.previous_image_path,
        "edited_image_path": request.generated.file_path,
        "generated_image_path": request.generated.file_path,
        "attempt_index": request.attempt_index,
        "step_name": request.step.name,
        "step_metadata": dict(request.step.metadata),
    }


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


__all__ = [
    "InterleaveThinkerCriticProvider",
    "InterleaveThinkerPlannerProvider",
]
