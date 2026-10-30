# SPDX-License-Identifier: Apache-2.0
"""Open-H3-Regenerate-2K through the OpenAI-compatible video routes, with a fake generator (no GPU)."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fastvideo.entrypoints.cli.inference_config import build_serve_config
from fastvideo.entrypoints.openai.api_server import create_app
from fastvideo.entrypoints.openai.media_policy import MediaPolicy
from fastvideo.entrypoints.openai.stores import VIDEO_STORE

ROOT = Path(__file__).resolve().parents[3]
MODEL = "Efficient-Large-Model/Open-H3-Regenerate-2K"
ARTIFACT = b"fake-2k-video-for-transport-tests"
PROMPT = "a quiet street at dusk"


class FileGenerator:

    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        output = Path(request.output.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(ARTIFACT)
        return SimpleNamespace(video_path=str(output), generation_time=0.01)

    def shutdown(self):
        pass


@pytest.fixture
def served(monkeypatch, tmp_path):
    generator = FileGenerator()
    monkeypatch.setattr("fastvideo.entrypoints.openai.api_server.VideoGenerator.from_fastvideo_args",
                        lambda args: generator)
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "draft.mp4").write_bytes(b"draft")
    Image.new("RGB", (8, 8)).save(drafts / "still.png")
    args = SimpleNamespace(model_path=MODEL, lora_path=None, lora_nickname="default", lora_strength=1.0,
                           override_pipeline_cls_name=None)
    config = build_serve_config(SimpleNamespace(config=str(ROOT / "examples/serving/openai_open_h3_regenerate_2k.yaml")),
                                overrides=[])
    app = create_app(args,
                     str(tmp_path / "server"),
                     config.default_request,
                     config.server.served_model_name,
                     media_policy=MediaPolicy(roots=(str(drafts), )))
    with TestClient(app) as client:
        yield client, generator, drafts
    asyncio.run(VIDEO_STORE.clear())


def _body(drafts, **extra):
    return {"prompt": PROMPT, "video_reference": {"video_url": str(drafts / "draft.mp4")}, **extra}


def test_serving_yaml_pins_the_2k_defaults():
    config = build_serve_config(SimpleNamespace(config=str(ROOT / "examples/serving/openai_open_h3_regenerate_2k.yaml")),
                                overrides=[])
    sampling = config.default_request.sampling
    assert (sampling.height, sampling.width, sampling.num_frames, sampling.num_inference_steps) == (1440, 2560, 124, 3)
    assert config.generator.model_path == MODEL


def test_sync_route_returns_the_mp4_and_maps_the_draft_to_video_path(served):
    client, generator, drafts = served
    response = client.post("/v1/videos/sync", json=_body(drafts, seed=7))
    assert response.status_code == 200, response.text
    assert response.content == ARTIFACT
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["x-model"] == "open-h3-regenerate-2k"
    request = generator.requests[0]
    assert request.prompt == PROMPT
    assert request.inputs.video_path == str((drafts / "draft.mp4").resolve())
    assert (request.sampling.height, request.sampling.width, request.sampling.num_frames) == (1440, 2560, 124)
    assert request.sampling.num_inference_steps == 3 and request.sampling.seed == 7


def test_async_route_completes_and_serves_the_content(served):
    client, generator, drafts = served
    created = client.post("/v1/videos", json=_body(drafts))
    assert created.status_code == 200, created.text
    video_id = created.json()["id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = client.get(f"/v1/videos/{video_id}").json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.02)
    assert status == "completed"
    assert client.get(f"/v1/videos/{video_id}/content").content == ARTIFACT
    assert generator.requests[0].inputs.video_path.endswith("draft.mp4")


def test_two_refine_steps_can_be_requested(served):
    client, generator, drafts = served
    assert client.post("/v1/videos/sync", json=_body(drafts, num_inference_steps=2)).status_code == 200
    assert generator.requests[0].sampling.num_inference_steps == 2


@pytest.mark.parametrize("reference, message", [
    ({"video_url": "data:video/mp4;base64,AAAA"}, "Data video references are not supported"),
    ({"video_url": "/etc/passwd"}, "Local media paths are not accepted"),
    ({"file_id": "file_1"}, "file_id video references are not supported"),
])
def test_unusable_video_references_are_rejected(served, reference, message):
    client, generator, _ = served
    response = client.post("/v1/videos/sync", json={"prompt": PROMPT, "video_reference": reference})
    assert response.status_code == 400 and message in response.text
    assert not generator.requests


@pytest.mark.parametrize("extra, message", [
    ({"video_reference": None}, "exactly one video_reference"),
    ({"image_reference": {"image_url": "STILL"}}, "exactly one video_reference"),
    ({"size": "1920x1080"}, "always renders 2560x1440"),
    ({"seconds": 5}, "first 121 frames"),
    ({"fps": 30}, "requires fps=24"),
])
def test_requests_the_pipeline_cannot_honor_are_rejected(served, extra, message):
    client, generator, drafts = served
    body = {**_body(drafts), **extra}
    if "image_reference" in body:
        body["image_reference"] = {"image_url": str(drafts / "still.png")}
    body = {key: value for key, value in body.items() if value is not None}
    response = client.post("/v1/videos/sync", json=body)
    assert response.status_code == 400 and message in response.text
    assert not generator.requests


def test_two_drafts_are_rejected(served):
    client, generator, drafts = served
    reference = {"video_url": str(drafts / "draft.mp4")}
    response = client.post("/v1/videos/sync", json={"prompt": PROMPT, "video_reference": [reference, reference]})
    assert response.status_code == 400 and "exactly one video_reference" in response.text
