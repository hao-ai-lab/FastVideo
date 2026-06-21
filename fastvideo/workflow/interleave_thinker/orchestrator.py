# SPDX-License-Identifier: Apache-2.0
"""Provider-based interleaved generation orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from fastvideo.workflow.interleave_thinker.generator import (
    ImageGeneratorBackend,
    encode_file_to_base64,
)
from fastvideo.workflow.interleave_thinker.schema import (
    CriticDecision,
    CriticInput,
    GeneratedImage,
    InterleaveAttempt,
    InterleaveEditRequest,
    InterleaveTrace,
    PlannedInterleaveStep,
    PlannerInput,
)


class PlannerProvider(Protocol):
    """Plans a user instruction into concrete generator calls."""

    def plan(self, request: PlannerInput) -> Sequence[PlannedInterleaveStep]:
        ...


class CriticProvider(Protocol):
    """Reviews one generated step and optionally proposes a refined prompt."""

    def review(self, request: CriticInput) -> CriticDecision:
        ...


class SinglePromptPlanner:
    """Fallback planner that runs the instruction as one generator prompt."""

    def __init__(self, *, max_attempts: int = 1) -> None:
        self.max_attempts = max(1, int(max_attempts))

    def plan(self, request: PlannerInput) -> Sequence[PlannedInterleaveStep]:
        return [
            PlannedInterleaveStep(
                prompt=request.instruction,
                input_image_path=request.initial_image_path,
                max_attempts=self.max_attempts,
            )
        ]


class AcceptAllCritic:
    """Fallback critic for smoke tests and simple generation flows."""

    def review(self, request: CriticInput) -> CriticDecision:
        del request
        return CriticDecision(success=True)


class InterleaveOrchestrator:
    """Run planner -> generator -> critic loops for interleaved workflows."""

    def __init__(
        self,
        *,
        planner: PlannerProvider,
        generator: ImageGeneratorBackend,
        critic: CriticProvider | None = None,
        width: int | None = None,
        height: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.planner = planner
        self.generator = generator
        self.critic = critic
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed

    def run(
        self,
        instruction: str,
        *,
        initial_image_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InterleaveTrace:
        planner_input = PlannerInput(
            instruction=instruction,
            initial_image_path=initial_image_path,
            metadata=dict(metadata or {}),
        )
        planned_steps = list(self.planner.plan(planner_input))
        attempts: list[InterleaveAttempt] = []
        previous_image_path = initial_image_path
        final_image: GeneratedImage | None = None

        if not planned_steps:
            return InterleaveTrace(
                instruction=instruction,
                attempts=[],
                final_image=None,
                success=False,
                metadata={"error": "planner returned no steps"},
            )

        for step_index, step in enumerate(planned_steps):
            accepted = False
            prompt = step.prompt
            step_input_path = step.input_image_path or previous_image_path
            max_attempts = max(1, int(step.max_attempts))

            for attempt_index in range(max_attempts):
                request = self._build_generation_request(
                    prompt,
                    input_image_path=step_input_path,
                )
                generated = self.generator.generate(request)
                decision = None
                if self.critic is not None:
                    decision = self.critic.review(
                        CriticInput(
                            step=step,
                            attempt_index=attempt_index,
                            generated=generated,
                            previous_image_path=step_input_path,
                            metadata=dict(step.metadata),
                        ))

                attempts.append(
                    InterleaveAttempt(
                        step_index=step_index,
                        attempt_index=attempt_index,
                        prompt=prompt,
                        generated=generated,
                        decision=decision,
                    ))

                if decision is None or decision.success:
                    accepted = True
                    final_image = generated
                    previous_image_path = generated.file_path or previous_image_path
                    break

                if decision.refine_prompt:
                    prompt = decision.refine_prompt

            if not accepted:
                return InterleaveTrace(
                    instruction=instruction,
                    attempts=attempts,
                    final_image=final_image,
                    success=False,
                    metadata={"failed_step_index": step_index},
                )

        return InterleaveTrace(
            instruction=instruction,
            attempts=attempts,
            final_image=final_image,
            success=True,
            metadata=dict(metadata or {}),
        )

    def _build_generation_request(
        self,
        prompt: str,
        *,
        input_image_path: str | None,
    ) -> InterleaveEditRequest:
        return InterleaveEditRequest(
            prompt=prompt,
            image=(encode_file_to_base64(input_image_path) if input_image_path else None),
            width=self.width,
            height=self.height,
            seed=self.seed,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        )
