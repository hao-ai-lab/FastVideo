# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from pathlib import Path

from fastvideo.workflow.interleave_thinker.orchestrator import InterleaveOrchestrator
from fastvideo.workflow.interleave_thinker.providers import (
    InterleaveThinkerCriticProvider,
    InterleaveThinkerPlannerProvider,
)
from fastvideo.workflow.interleave_thinker.schema import (
    CriticInput,
    GeneratedImage,
    PlannedInterleaveStep,
    PlannerInput,
)
from fastvideo.train.models.interleave_thinker import InterleavePlannerStep


class _FakePlannerModel:

    def __init__(self) -> None:
        self.calls = []

    def generate_interleave_plan(self, instruction, *, input_image_paths=None, **kwargs):
        self.calls.append((instruction, input_image_paths, kwargs))
        return {
            "generation_index": 0,
            "response": "raw planner response",
            "steps": [
                InterleavePlannerStep(
                    step_number=1,
                    step_name="Base",
                    instruction="Draw the base cat shapes",
                    prompt="simple cat base shapes",
                    auxiliary_text=None,
                ),
                InterleavePlannerStep(
                    step_number=2,
                    step_name="Color",
                    instruction="Color the cat",
                    prompt="color the cat orange",
                    auxiliary_text="Use warm colors.",
                ),
            ],
        }


class _FakeAuxiliaryOnlyPlannerModel:

    def generate_interleave_plan(self, instruction, *, input_image_paths=None, **kwargs):
        del instruction, input_image_paths, kwargs
        return {
            "generation_index": 0,
            "response": "raw auxiliary-only planner response",
            "steps": [
                InterleavePlannerStep(
                    step_number=1,
                    step_name="Answer",
                    instruction=None,
                    prompt=None,
                    auxiliary_text="The requested textual answer.",
                ),
            ],
        }


class _FakeCriticModel:

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_interleave_responses(self, batch, **kwargs):
        self.calls.append((batch, kwargs))
        response = self.responses.pop(0)
        return [{"response": response}]


class _FakeGenerator:

    def __init__(self) -> None:
        self.prompts = []

    def generate(self, request, *, request_id=None):
        del request_id
        self.prompts.append(request.prompt)
        file_path = f"/tmp/generated-{len(self.prompts)}.png"
        Path(file_path).write_bytes(request.prompt.encode("utf-8"))
        return GeneratedImage(
            prompt=request.prompt,
            image_base64=base64.b64encode(request.prompt.encode("utf-8")).decode("utf-8"),
            file_path=file_path,
        )


def _critic_response(success: bool, refine_prompt: str = "refined prompt") -> str:
    success_text = "true" if success else "false"
    return (
        "<think>review</think>"
        f'<answer>{{"previous_step_success": {success_text}, "refine_prompt": "{refine_prompt}"}}</answer>')


def test_interleave_thinker_planner_provider_converts_model_steps():
    model = _FakePlannerModel()
    provider = InterleaveThinkerPlannerProvider(model, max_attempts_per_step=3)

    steps = provider.plan(PlannerInput(instruction="draw a cat", initial_image_path="/tmp/input.png"))

    assert [step.prompt for step in steps] == ["simple cat base shapes", "color the cat orange"]
    assert steps[0].input_image_path == "/tmp/input.png"
    assert steps[1].input_image_path is None
    assert steps[0].max_attempts == 3
    assert steps[0].metadata["planner_step_number"] == 1
    assert steps[1].metadata["planner_auxiliary_text"] == "Use warm colors."
    assert model.calls[0][0] == "draw a cat"
    assert model.calls[0][1] == ["/tmp/input.png"]


def test_interleave_thinker_planner_provider_does_not_generate_from_auxiliary_text():
    provider = InterleaveThinkerPlannerProvider(_FakeAuxiliaryOnlyPlannerModel())
    generator = _FakeGenerator()

    steps = provider.plan(PlannerInput(instruction="answer this question"))
    trace = InterleaveOrchestrator(
        planner=provider,
        generator=generator,
    ).run("answer this question")

    assert steps == []
    assert trace.success is False
    assert trace.metadata["error"] == "planner returned no steps"
    assert generator.prompts == []


def test_interleave_thinker_critic_provider_converts_answer_to_decision():
    model = _FakeCriticModel([_critic_response(False, "make it clearer")])
    provider = InterleaveThinkerCriticProvider(model, max_new_tokens=64)

    decision = provider.review(
        CriticInput(
            step=PlannedInterleaveStep(
                prompt="simple cat base shapes",
                name="Base",
                metadata={"planner_instruction": "Draw the base cat shapes"},
            ),
            attempt_index=0,
            generated=GeneratedImage(prompt="refined cat base shapes", file_path="/tmp/after.png"),
            previous_image_path="/tmp/before.png",
        ))

    assert decision.success is False
    assert decision.refine_prompt == "make it clearer"
    batch, kwargs = model.calls[0]
    item = batch["items"][0]
    assert item["origin_prompt"] == "Draw the base cat shapes"
    assert item["previous_prompt"] == "refined cat base shapes"
    assert item["previous_image_path"] == "/tmp/before.png"
    assert item["edited_image_path"] == "/tmp/after.png"
    assert kwargs["max_new_tokens"] == 64


def test_interleave_thinker_critic_provider_handles_unparseable_response():
    provider = InterleaveThinkerCriticProvider(_FakeCriticModel(["not parseable"]))

    decision = provider.review(
        CriticInput(
            step=PlannedInterleaveStep(prompt="prompt"),
            attempt_index=0,
            generated=GeneratedImage(prompt="prompt", file_path="/tmp/after.png"),
        ))

    assert decision.success is False
    assert decision.reason == "InterleaveThinker critic response did not parse"
    assert decision.metadata["critic_response"] == "not parseable"


def test_interleave_orchestrator_runs_through_model_providers():
    planner = InterleaveThinkerPlannerProvider(_FakePlannerModel(), max_attempts_per_step=2)
    critic_model = _FakeCriticModel([
        _critic_response(False, "better cat base"),
        _critic_response(True, "better cat base"),
        _critic_response(True, "color the cat orange"),
    ])
    critic = InterleaveThinkerCriticProvider(critic_model)
    generator = _FakeGenerator()
    orchestrator = InterleaveOrchestrator(
        planner=planner,
        generator=generator,
        critic=critic,
    )

    trace = orchestrator.run("draw a cat")

    assert trace.success is True
    assert generator.prompts == ["simple cat base shapes", "better cat base", "color the cat orange"]
    assert len(trace.attempts) == 3
    assert trace.attempts[0].decision is not None
    assert trace.attempts[0].decision.success is False
    assert trace.final_image is not None
    assert trace.final_image.prompt == "color the cat orange"
    critic_prompts = [call[0]["items"][0]["previous_prompt"] for call in critic_model.calls]
    assert critic_prompts == ["simple cat base shapes", "better cat base", "color the cat orange"]
