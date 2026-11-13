# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 draft handoff over the OpenAI API: return_draft_bundle, the draft_bundle route and draft_bundle input."""

import asyncio
import base64
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from fastvideo.api.compat import request_to_batch_extra
from fastvideo.entrypoints.cli.inference_config import build_serve_config
from fastvideo.entrypoints.openai.api_server import create_app
from fastvideo.entrypoints.openai.media_policy import MediaPolicy
from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest
from fastvideo.entrypoints.openai.request_adapter import RequestAdaptationError, build_generation_request
from fastvideo.entrypoints.openai.stores import VIDEO_STORE
from fastvideo.pipelines.basic.minimax_h3.draft_bundle import BUNDLE_MEDIA_TYPE, BUNDLE_PATH_EXTRA_KEY

ROOT = Path(__file__).resolve().parents[3]
REGEN = "Efficient-Large-Model/Open-H3-Regenerate-2K"
H3 = "MiniMaxAI/MiniMax-H3"


def _args(model_path: str):
    return SimpleNamespace(model_path=model_path, lora_path=None, lora_nickname="default", lora_strength=1.0,
                           override_pipeline_cls_name=None)


def test_return_draft_bundle_puts_the_bundle_next_to_the_video(tmp_path):
    request = VideoGenerationRequest(prompt="a fox", size="832x480", num_frames=124, return_draft_bundle=True)
    adapted = build_generation_request("video_gen_x", request, _args(H3), served_model_name="minimax-h3",
                                       output_dir=str(tmp_path))
    bundle_path = request_to_batch_extra(adapted)[BUNDLE_PATH_EXTRA_KEY]
    assert Path(bundle_path) == Path(adapted.output.output_path).with_name("video_gen_x.h3draft.pt")


def test_without_return_draft_bundle_no_bundle_is_written(tmp_path):
    request = VideoGenerationRequest(prompt="a fox", size="832x480", num_frames=124)
    adapted = build_generation_request("video_gen_x", request, _args(H3), served_model_name="minimax-h3",
                                       output_dir=str(tmp_path))
    assert BUNDLE_PATH_EXTRA_KEY not in request_to_batch_extra(adapted)


def test_return_draft_bundle_needs_a_minimax_h3_server(tmp_path):
    request = VideoGenerationRequest(prompt="a fox", return_draft_bundle=True,
                                     video_reference={"video_url": "/tmp/draft.mp4"})
    with pytest.raises(RequestAdaptationError, match="return_draft_bundle"):
        build_generation_request("video_gen_x", request, _args(REGEN), served_model_name="regen",
                                 output_dir=str(tmp_path))


def test_draft_bundle_is_refused_by_other_servers(tmp_path):
    request = VideoGenerationRequest(prompt="a fox", size="832x480", num_frames=124, draft_bundle="/tmp/x.h3draft.pt")
    with pytest.raises(RequestAdaptationError, match="draft_bundle"):
        build_generation_request("video_gen_x", request, _args(H3), served_model_name="minimax-h3",
                                 output_dir=str(tmp_path))


class FileGenerator:

    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        output = Path(request.output.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"2k")
        return SimpleNamespace(video_path=str(output), generation_time=0.01)

    def shutdown(self):
        pass


@pytest.fixture
def regen(monkeypatch, tmp_path):
    generator = FileGenerator()
    monkeypatch.setattr("fastvideo.entrypoints.openai.api_server.VideoGenerator.from_fastvideo_args",
                        lambda args: generator)
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    torch.save({"format": 1, "latent": torch.zeros(24, 2, 3, 4)}, drafts / "clip.h3draft.pt")
    (drafts / "draft.mp4").write_bytes(b"draft")
    config = build_serve_config(SimpleNamespace(config=str(ROOT / "examples/serving/openai_open_h3_regenerate_2k.yaml")),
                                overrides=[])
    app = create_app(_args(REGEN),
                     str(tmp_path / "server"),
                     config.default_request,
                     config.server.served_model_name,
                     media_policy=MediaPolicy(roots=(str(drafts), )))
    with TestClient(app) as client:
        yield client, generator, drafts, tmp_path
    asyncio.run(VIDEO_STORE.clear())


def test_a_local_draft_bundle_becomes_the_draft(regen):
    client, generator, drafts, _ = regen
    response = client.post("/v1/videos/sync", json={"prompt": "a fox", "draft_bundle": str(drafts / "clip.h3draft.pt")})
    assert response.status_code == 200, response.text
    assert generator.requests[0].inputs.video_path == str((drafts / "clip.h3draft.pt").resolve())


def test_an_inline_draft_bundle_is_written_then_removed(regen):
    client, generator, drafts, _ = regen
    payload = base64.b64encode((drafts / "clip.h3draft.pt").read_bytes()).decode()
    response = client.post("/v1/videos/sync",
                           json={"prompt": "a fox", "draft_bundle": f"data:{BUNDLE_MEDIA_TYPE};base64,{payload}"})
    assert response.status_code == 200, response.text
    video_path = Path(generator.requests[0].inputs.video_path)
    assert video_path.name.endswith(".h3draft.pt") and not video_path.exists()


@pytest.mark.parametrize("extra, message", [
    ({"draft_bundle": "data:video/mp4;base64,AAAA"}, "data:application/x-fastvideo-h3-draft"),
    ({"draft_bundle": "/etc/passwd"}, ""),
])
def test_bad_draft_bundles_are_rejected(regen, extra, message):
    client, generator, _, _ = regen
    response = client.post("/v1/videos/sync", json={"prompt": "a fox", **extra})
    assert response.status_code == 400 and message in response.text
    assert not generator.requests


def test_draft_bundle_and_video_reference_conflict(regen):
    client, _, drafts, _ = regen
    response = client.post("/v1/videos/sync",
                           json={"prompt": "a fox", "draft_bundle": str(drafts / "clip.h3draft.pt"),
                                 "video_reference": {"video_url": str(drafts / "draft.mp4")}})
    assert response.status_code == 400 and "either draft_bundle" in response.text


def test_the_draft_bundle_route_serves_the_bundle(regen):
    client, _, drafts, tmp_path = regen
    created = client.post("/v1/videos", json={"prompt": "a fox", "video_reference": {"video_url": str(drafts / "draft.mp4")}})
    video_id = created.json()["id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and client.get(f"/v1/videos/{video_id}").json()["status"] != "completed":
        time.sleep(0.02)
    assert client.get(f"/v1/videos/{video_id}/draft_bundle").status_code == 404
    video_path = Path(client.get(f"/v1/videos/{video_id}").json()["file_path"])
    video_path.with_name(f"{video_id}.h3draft.pt").write_bytes(b"bundle")
    response = client.get(f"/v1/videos/{video_id}/draft_bundle")
    assert response.status_code == 200 and response.content == b"bundle"
    assert response.headers["content-type"] == BUNDLE_MEDIA_TYPE
