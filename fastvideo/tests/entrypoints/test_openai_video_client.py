# SPDX-License-Identifier: Apache-2.0
"""Real OpenAI SDK and HTTP routes with a fake generator, no GPU or cloud calls."""

import asyncio
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen
import json

import openai
import pytest
import uvicorn

from fastvideo.entrypoints.cli.inference_config import build_serve_config
from fastvideo.entrypoints.openai.api_server import create_app
from fastvideo.entrypoints.openai.stores import VIDEO_STORE
from fastvideo.api.compat import normalize_generation_request
from fastvideo.entrypoints.openai.mlx_server import MLXH3Generator, create_mlx_app, load_config, validate_mlx_video_request
from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest

ROOT = Path(__file__).resolve().parents[3]
MODEL = "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"
ARTIFACT = b"fake-video-content-for-transport-tests"


class FileGenerator:
    def __init__(self):
        self.requests = []
        self.fail = False
        self.loads = 0

    def generate(self, request):
        self.requests.append(request)
        if self.fail or request.prompt == "fail":
            raise RuntimeError("Test generation failure")
        output = Path(request.output.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(ARTIFACT)
        return SimpleNamespace(video_path=str(output), generation_time=0.01)

    def shutdown(self):
        pass


@pytest.fixture(params=["cuda", "mlx"])
def local_server(request, monkeypatch, tmp_path):
    generator = FileGenerator()
    generator.runtime = request.param
    generator.native_calls = []
    generator.thread_ids = []
    def load(args):
        generator.loads += 1
        return generator

    monkeypatch.setattr(
        "fastvideo.entrypoints.openai.api_server.VideoGenerator.from_fastvideo_args",
        load,
    )
    args = SimpleNamespace(
        model_path=MODEL, lora_path=None, lora_nickname="default",
        lora_strength=1.0, override_pipeline_cls_name=None,
    )
    config = build_serve_config(
        SimpleNamespace(config=str(ROOT / "examples/serving/openai_fasth3.yaml")),
        overrides=["--server.host", "127.0.0.1"],
    )
    assert config.server.host == "127.0.0.1"
    assert config.generator.model_path == MODEL
    assert config.generator.engine.num_gpus == 4
    app = create_app(args, str(tmp_path / "server"), config.default_request, config.server.served_model_name)
    if request.param == "mlx":
        def native_generate(prompt, **kwargs):
            generator.thread_ids.append(threading.get_ident())
            generator.native_calls.append(kwargs)
            request = normalize_generation_request({
                "prompt": prompt,
                "sampling": {key: kwargs[key] for key in ["height", "width", "num_frames", "seed"]},
                "output": {"output_path": kwargs["output_path"]},
            })
            request.sampling.num_inference_steps = kwargs["num_steps"] + 1
            return generator.generate(request)

        def load_native(config):
            generator.loads += 1
            generator.thread_ids.append(threading.get_ident())
            return SimpleNamespace(generate=native_generate)

        monkeypatch.setattr(MLXH3Generator, "_load", staticmethod(load_native))
        monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_pipeline._cleanup_mlx", lambda: None)
        mlx_config = load_config(str(ROOT / "examples/serving/mlx_fasth3.yaml"))
        mlx_config.server.output_dir = str(tmp_path / "server")
        app = create_mlx_app(mlx_config)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        base_url = f"http://127.0.0.1:{listener.getsockname()[1]}/v1"
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 15
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started, "Local test server did not start"
            with openai.OpenAI(base_url=base_url, api_key="local", max_retries=0, timeout=5) as client:
                yield client, generator, base_url
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive(), "Local test server did not stop"
            asyncio.run(VIDEO_STORE.clear())


def test_openai_video_lifecycle(local_server):
    client, generator, _ = local_server
    assert client.models.list().data[0].id == "fasth3"
    video = client.videos.create_and_poll(model="fasth3", prompt="a fox", poll_interval_ms=1)
    assert video.status == "completed"
    assert video.model == "fasth3"
    assert video.size == ("832x480" if generator.runtime == "mlx" else "1344x768")
    assert video.error is None
    request = generator.requests[0]
    assert request.sampling.num_frames == 124
    assert request.sampling.num_inference_steps == 5
    assert request.sampling.guidance_scale == 1.0
    assert client.videos.retrieve(video.id).status == "completed"
    assert client.videos.list(limit=1).data[0].id == video.id
    assert client.videos.download_content(video.id, variant="video").read() == ARTIFACT
    assert client.videos.delete(video.id).deleted
    with pytest.raises(openai.NotFoundError):
        client.videos.retrieve(video.id)


def test_openai_poll_returns_failed_resource(local_server):
    client, _, _ = local_server
    video = client.videos.create_and_poll(model="fasth3", prompt="fail", poll_interval_ms=1)
    assert video.status == "failed"
    assert video.error.code == "generation_failed"
    assert video.error.message == "Test generation failure"
    assert client.videos.retrieve(video.id).status == "failed"
    with pytest.raises(openai.UnprocessableEntityError):
        client.videos.download_content(video.id)


def test_openai_admission_errors_are_not_jobs(local_server):
    client, generator, _ = local_server
    for params in [{"model": "missing", "prompt": "a fox"}, {"model": "fasth3", "prompt": " "}]:
        with pytest.raises(openai.BadRequestError):
            client.videos.create(**params)
    assert not client.videos.list().data
    assert not generator.requests


def test_openai_extra_body_and_unsupported_content_variant(local_server):
    client, generator, _ = local_server
    video = client.videos.create_and_poll(
        model="fasth3", prompt="a fox", extra_body={"seed": 42}, poll_interval_ms=1,
    )
    assert generator.requests[0].sampling.seed == 42
    with pytest.raises(openai.BadRequestError, match="Only the video"):
        client.videos.download_content(video.id, variant="thumbnail")


def test_prompt_iteration_reuses_the_loaded_generator(local_server):
    client, generator, _ = local_server
    first = client.videos.create_and_poll(model="fasth3", prompt="a fox", poll_interval_ms=1)
    second = client.videos.create_and_poll(model="fasth3", prompt="a fox in snow", poll_interval_ms=1)
    assert first.id != second.id
    assert first.status == second.status == "completed"
    assert [request.prompt for request in generator.requests] == ["a fox", "a fox in snow"]
    assert generator.loads == 1
    if generator.runtime == "mlx":
        assert len(set(generator.thread_ids)) == 1
        assert [call["num_steps"] for call in generator.native_calls] == [4, 4]


def test_playground_assets_and_config_do_not_generate(local_server):
    _, generator, base_url = local_server
    origin = base_url.removesuffix("/v1")
    with urlopen(origin + "/playground/") as response:
        assert response.status == 200
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        html = response.read().decode()
        assert "Generate video" in html
        assert 'src="./playground.js"' in html
    for asset in ["playground.js", "playground.css"]:
        with urlopen(origin + "/playground/" + asset) as response:
            assert response.status == 200
    with urlopen(origin + "/playground/config") as response:
        config = json.load(response)
    assert config == {
        "model": "fasth3",
        "runtime": generator.runtime,
        "defaults": {"width": 832, "height": 480, "num_frames": 124, "fps": 24, "seed": 2026}
        if generator.runtime == "mlx" else
        {"width": 1344, "height": 768, "num_frames": 124, "fps": 24, "seed": 1000},
    }
    with pytest.raises(HTTPError) as error:
        urlopen(origin + "/playground/api_server.py")
    assert error.value.code == 404
    assert not generator.requests
    assert generator.loads == 1


def test_playground_does_not_advertise_implicit_defaults(local_server, monkeypatch):
    _, _, base_url = local_server
    request = normalize_generation_request({"sampling": {"seed": 42}})
    monkeypatch.setattr("fastvideo.entrypoints.openai.playground.get_default_request", lambda: request)
    with urlopen(base_url.removesuffix("/v1") + "/playground/config") as response:
        config = json.load(response)
    assert config["defaults"] == {"width": None, "height": None, "num_frames": None, "fps": None, "seed": 42}


@pytest.mark.parametrize("model,override", [
    ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", None),
    (MODEL, "MiniMaxH3Ref2VAModularPipeline"),
])
def test_playground_is_limited_to_h3_text_generation(local_server, monkeypatch, model, override):
    _, generator, base_url = local_server
    monkeypatch.setattr(
        "fastvideo.entrypoints.openai.playground.get_server_args",
        lambda: SimpleNamespace(model_path=model, override_pipeline_cls_name=override),
    )
    with pytest.raises(HTTPError) as error:
        urlopen(base_url.removesuffix("/v1") + "/playground/")
    assert error.value.code == 404
    assert not generator.requests


@pytest.mark.parametrize("fields", [
    {"image_reference": "https://example.com/input.png"}, {"task": "fl2va"},
    {"guidance_scale": 2}, {"num_inference_steps": 4}, {"negative_prompt": "bad"},
    {"seed": -1}, {"seed": 2**32}, {"extra_params": {"vsa_mode": "exempt"}},
    {"enable_frame_interpolation": True},
])
def test_mlx_rejects_unsupported_options_before_generation(fields):
    if "image_reference" in fields:
        fields = {"image_reference": {"image_url": fields["image_reference"]}}
    with pytest.raises(ValueError):
        validate_mlx_video_request(VideoGenerationRequest(prompt="a fox", **fields))


def test_mlx_http_rejects_reference_media_and_image_routes(local_server):
    client, generator, base_url = local_server
    if generator.runtime != "mlx":
        return
    with pytest.raises(openai.BadRequestError, match="image_reference"):
        client.videos.create(model="fasth3", prompt="a fox", extra_body={
            "image_reference": {"image_url": "http://127.0.0.1:1/must-not-fetch"},
        })
    assert not generator.requests
    assert not client.videos.list().data
    with urlopen(base_url.removesuffix("/v1") + "/openapi.json") as response:
        spec = json.load(response)
    assert "/v1/images" not in spec["paths"]


@pytest.mark.parametrize("filename,runner", [("video.py", sys.executable), ("video.sh", "bash"), ("video.mjs", "node")])
@pytest.mark.parametrize("fail", [False, True])
def test_cookbook_clients_handle_completion_and_failure(local_server, tmp_path, filename, runner, fail):
    if filename == "video.sh" and not shutil.which("jq"):
        pytest.skip("The cURL example requires jq")
    if filename == "video.mjs" and (
        not shutil.which("node") or not (ROOT / "examples/serving/clients/node_modules/openai").is_dir()
    ):
        pytest.skip("Run npm ci --prefix examples/serving/clients to test the JavaScript example")
    _, generator, base_url = local_server
    generator.fail = fail
    env = dict(os.environ, FASTVIDEO_BASE_URL=base_url, FASTVIDEO_MODEL="fasth3", FASTVIDEO_API_KEY="local")
    result = subprocess.run(
        [runner, str(ROOT / "examples/serving/clients" / filename)],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    outputs = list(tmp_path.glob("*.mp4"))
    if fail:
        assert result.returncode != 0
        assert "Test generation failure" in result.stderr
        assert not outputs
        return
    assert result.returncode == 0, result.stderr
    assert len(outputs) == 1
    assert outputs[0].read_bytes() == ARTIFACT
