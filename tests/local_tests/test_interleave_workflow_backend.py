# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from fastvideo.api.compat import (
    explicit_request_updates,
    legacy_generate_call_to_request,
)
from fastvideo.api.results import GenerationResult
from fastvideo.workflow.interleave_thinker.generator import (
    FastVideoImageGeneratorBackend,
    NanoBananaImageGeneratorBackend,
)
from fastvideo.workflow.interleave_thinker.orchestrator import InterleaveOrchestrator
from fastvideo.workflow.interleave_thinker.schema import (
    CriticDecision,
    GeneratedImage,
    InterleaveEditRequest,
    PlannedInterleaveStep,
)
from fastvideo.workflow.interleave_thinker.trace import (
    save_trace,
    trace_to_dict,
)


def test_interleave_edit_request_accepts_singular_step_field() -> None:
    request = InterleaveEditRequest(
        prompt="a ceramic cup on a table",
        num_inference_step=4,
    )

    assert request.resolved_num_inference_steps() == 4

    plural_request = InterleaveEditRequest(
        prompt="a ceramic cup on a table",
        num_inference_step=4,
        num_inference_steps=8,
    )
    assert plural_request.resolved_num_inference_steps() == 8


def test_fastvideo_backend_translates_edit_request(tmp_path: Path) -> None:
    class FakeGenerator:
        def __init__(self) -> None:
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            updates = explicit_request_updates(request)
            output_path = Path(updates["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fake-png")
            return GenerationResult(
                prompt=request.prompt,
                video_path=str(output_path),
                generation_time=0.25,
            )

    default_request = legacy_generate_call_to_request(
        "unused",
        None,
        legacy_kwargs={
            "height": 512,
            "width": 768,
            "num_inference_steps": 4,
        },
    )
    fake = FakeGenerator()
    backend = FastVideoImageGeneratorBackend(
        fake,
        output_dir=str(tmp_path),
        default_request=default_request,
    )
    input_b64 = base64.b64encode(b"input-image").decode("utf-8")

    generated = backend.generate(
        InterleaveEditRequest(
            prompt="turn it into a watercolor",
            image=input_b64,
            width=1024,
            seed=7,
        ),
        request_id="abc123",
    )

    assert generated.prompt == "turn it into a watercolor"
    assert generated.image_base64 == base64.b64encode(b"fake-png").decode("utf-8")
    assert generated.file_path is not None
    assert generated.file_path.endswith("abc123.png")

    updates = explicit_request_updates(fake.requests[0])
    assert updates["num_frames"] == 1
    assert updates["fps"] == 1
    assert updates["height"] == 512
    assert updates["width"] == 1024
    assert updates["num_inference_steps"] == 4
    assert updates["seed"] == 7
    assert updates["save_video"] is True
    assert updates["return_frames"] is False
    assert Path(updates["image_path"]).read_bytes() == b"input-image"


def test_nano_banana_client_setup_failure_is_not_retried(tmp_path: Path) -> None:
    class FailingClientBackend(NanoBananaImageGeneratorBackend):

        def __init__(self) -> None:
            super().__init__(
                api_key="fake-key",
                output_dir=str(tmp_path),
                max_attempts=3,
                retry_delay_s=999.0,
            )
            self.client_calls = 0

        def _client_instance(self):
            self.client_calls += 1
            raise RuntimeError("missing Gemini SDK")

    backend = FailingClientBackend()

    with pytest.raises(RuntimeError, match="missing Gemini SDK"):
        backend.generate(InterleaveEditRequest(prompt="draw a red mug"), request_id="client-error")

    assert backend.client_calls == 1


def test_nano_banana_generate_config_failure_is_not_retried(tmp_path: Path) -> None:
    class FailingConfigBackend(NanoBananaImageGeneratorBackend):

        def __init__(self) -> None:
            super().__init__(
                api_key="fake-key",
                output_dir=str(tmp_path),
                max_attempts=3,
                retry_delay_s=999.0,
            )
            self.client_calls = 0
            self.config_calls = 0

        def _client_instance(self):
            self.client_calls += 1
            return object()

        def _make_generate_config(self):
            self.config_calls += 1
            raise ValueError("invalid Gemini config")

    backend = FailingConfigBackend()

    with pytest.raises(ValueError, match="invalid Gemini config"):
        backend.generate(InterleaveEditRequest(prompt="draw a red mug"), request_id="config-error")

    assert backend.client_calls == 1
    assert backend.config_calls == 1


def test_interleave_orchestrator_retries_with_refined_prompt() -> None:
    class FakePlanner:
        def plan(self, request):
            return [
                PlannedInterleaveStep(
                    prompt=request.instruction,
                    max_attempts=2,
                )
            ]

    class FakeGenerator:
        def __init__(self) -> None:
            self.prompts = []
            self.seeds = []

        def generate(self, request, *, request_id=None):
            del request_id
            self.prompts.append(request.prompt)
            self.seeds.append(request.seed)
            return GeneratedImage(
                prompt=request.prompt,
                image_base64=base64.b64encode(request.prompt.encode("utf-8")).decode("utf-8"),
                file_path=f"/tmp/{len(self.prompts)}.png",
            )

    class RefiningCritic:
        def __init__(self) -> None:
            self.calls = 0

        def review(self, request):
            self.calls += 1
            if self.calls == 1:
                return CriticDecision(
                    success=False,
                    refine_prompt="refined prompt",
                    reason="first attempt missed the instruction",
                )
            return CriticDecision(success=True)

    generator = FakeGenerator()
    orchestrator = InterleaveOrchestrator(
        planner=FakePlanner(),
        generator=generator,
        critic=RefiningCritic(),
        seed=100,
    )

    trace = orchestrator.run("initial prompt")

    assert trace.success is True
    assert generator.prompts == ["initial prompt", "refined prompt"]
    assert generator.seeds == [100, 101]
    assert len(trace.attempts) == 2
    assert trace.final_image is not None
    assert trace.final_image.prompt == "refined prompt"


def test_trace_serialization_omits_images_by_default(tmp_path: Path) -> None:
    generated = GeneratedImage(
        prompt="final",
        image_base64="large-payload",
        file_path="/tmp/final.png",
        inference_time_s=0.5,
    )
    trace = InterleaveOrchestrator(
        planner=FakeSingleStepPlanner(),
        generator=FakeSingleStepGenerator(generated),
    ).run("final")

    payload = trace_to_dict(trace)

    assert payload["success"] is True
    assert payload["final_image"]["file_path"] == "/tmp/final.png"
    assert "image_base64" not in payload["final_image"]
    assert "image_base64" not in payload["attempts"][0]["generated"]

    trace_path = tmp_path / "trace.json"
    save_trace(trace, trace_path)
    assert "large-payload" not in trace_path.read_text(encoding="utf-8")

    payload_with_images = trace_to_dict(trace, include_images=True)
    assert payload_with_images["final_image"]["image_base64"] == "large-payload"


class FakeSingleStepPlanner:
    def plan(self, request):
        return [PlannedInterleaveStep(prompt=request.instruction)]


class FakeSingleStepGenerator:
    def __init__(self, generated: GeneratedImage) -> None:
        self.generated = generated

    def generate(self, request, *, request_id=None):
        del request, request_id
        return self.generated
