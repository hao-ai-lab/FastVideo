# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from pathlib import Path

from starlette.testclient import TestClient

from fastvideo.api.compat import (
    explicit_request_updates,
    legacy_generate_call_to_request,
)
from fastvideo.api.results import GenerationResult
from fastvideo.entrypoints.interleave.generator import (
    FastVideoImageGeneratorBackend,
)
from fastvideo.entrypoints.interleave.orchestrator import InterleaveOrchestrator
from fastvideo.entrypoints.interleave.schema import (
    CriticDecision,
    GeneratedImage,
    InterleaveEditRequest,
    PlannedInterleaveStep,
)
from fastvideo.entrypoints.interleave.server import build_app


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


def test_interleave_server_returns_compatible_edit_response() -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.requests = []

        def generate(self, request, *, request_id=None):
            self.requests.append((request, request_id))
            return GeneratedImage(
                prompt=request.prompt,
                image_base64="ZmFrZQ==",
                file_path="/tmp/fake.png",
                inference_time_s=0.1,
                metadata={"backend": "fake"},
            )

    backend = FakeBackend()
    app = build_app(backend=backend)

    with TestClient(app) as client:
        response = client.post(
            "/edit",
            json={
                "prompt": "add a red umbrella",
                "num_inference_step": 4,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["edited_image"] == "ZmFrZQ=="
    assert payload["file_path"] == "/tmp/fake.png"
    assert payload["metadata"]["backend"] == "fake"
    assert backend.requests[0][0].prompt == "add a red umbrella"
    assert backend.requests[0][1]


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

        def generate(self, request, *, request_id=None):
            del request_id
            self.prompts.append(request.prompt)
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
    )

    trace = orchestrator.run("initial prompt")

    assert trace.success is True
    assert generator.prompts == ["initial prompt", "refined prompt"]
    assert len(trace.attempts) == 2
    assert trace.final_image is not None
    assert trace.final_image.prompt == "refined prompt"
