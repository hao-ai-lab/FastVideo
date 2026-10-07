# SPDX-License-Identifier: Apache-2.0
"""Image generation route compatibility with a mocked generator."""

import base64
from io import BytesIO

from PIL import Image
import torch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.tests.entrypoints.test_video_generator import (
    _single_video_args, _single_video_generator, _single_video_output_batch, _small_sampling_param,
)
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.entrypoints.openai import image_api
from fastvideo.entrypoints.openai.stores import AsyncDictStore


@pytest.mark.parametrize("path", [
    pytest.param("/v1/images/generations", id="openai-route"),
    pytest.param("/v1/images", id="legacy-route"),
])
@pytest.mark.parametrize("response_format", [
    pytest.param("b64_json", id="b64-json"),
    pytest.param("url", id="url"),
])
@pytest.mark.parametrize("output_format", [None, "png", "jpeg", "webp"])
def test_image_generation_routes(path, response_format, output_format, monkeypatch, tmp_path):
    image_bytes = b"test-image-content"
    generator = VideoGenerator.__new__(VideoGenerator)
    generator.fastvideo_args = SimpleNamespace(workload_type=WorkloadType.from_string("t2i"))

    def save_image(**kwargs):
        output_path = generator._prepare_output_path(kwargs["output_path"], kwargs["prompt"])
        Path(output_path).write_bytes(image_bytes)

    generate = Mock(side_effect=save_image)

    async def run_serialized(fn, **kwargs):
        return fn(**kwargs)

    engine = SimpleNamespace(
        generator=SimpleNamespace(generate_video=generate),
        run_serialized=AsyncMock(side_effect=run_serialized),
    )
    monkeypatch.setattr(image_api, "get_serving_engine", lambda: engine)
    monkeypatch.setattr(image_api, "get_output_dir", lambda: str(tmp_path))
    monkeypatch.setattr(image_api, "IMAGE_STORE", AsyncDictStore())
    app = FastAPI()
    app.include_router(image_api.router)

    with TestClient(app) as client:
        request = {"prompt": "a cat", "response_format": response_format}
        if output_format is not None:
            request["output_format"] = output_format
        response = client.post(path, json=request)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["data"][0]["revised_prompt"] == "a cat"
        if response_format == "b64_json":
            assert base64.b64decode(body["data"][0]["b64_json"]) == image_bytes
        else:
            assert body["data"][0]["url"] == f"/v1/images/{body['id']}/content"
            content = client.get(body["data"][0]["url"])
            assert content.status_code == 200
            assert content.content == image_bytes

    generate.assert_called_once()
    assert generate.call_args.kwargs["prompt"] == "a cat"
    assert generate.call_args.kwargs["num_frames"] == 1
    engine.run_serialized.assert_awaited_once()


@pytest.mark.parametrize("path", ["/v1/images/generations", "/v1/images", "/v1/images/edits"])
@pytest.mark.parametrize("response_format", ["url", "b64_json"])
@pytest.mark.parametrize("output_format", ["png", "jpeg", "webp"])
def test_image_batch_returns_individual_images(path, response_format, output_format, monkeypatch, tmp_path):
    monkeypatch.setattr("fastvideo.entrypoints.video_generator.pixels_to_uint8",
                        Mock(side_effect=AssertionError("Unexpected batch preview conversion")))
    monkeypatch.setattr("fastvideo.entrypoints.video_generator.torchvision.utils.make_grid",
                        Mock(side_effect=AssertionError("Unexpected batch preview grid")))
    samples = torch.zeros(2, 3, 1, 16, 16)
    samples[0, 0] = 1
    samples[1, 2] = 1

    args = _single_video_args()
    args.model_path = "test-model"
    args.prompt_txt = None
    args.workload_type = WorkloadType.from_string("t2i")
    generator = _single_video_generator(_single_video_output_batch(samples), args)
    monkeypatch.setattr(SamplingParam, "from_pretrained", lambda model_path: _small_sampling_param())

    def generate_images(**kwargs):
        return generator.generate_video(**kwargs)

    generate = Mock(side_effect=generate_images)

    async def run_serialized(fn, **kwargs):
        return fn(**kwargs)

    engine = SimpleNamespace(generator=SimpleNamespace(generate_video=generate),
                             run_serialized=AsyncMock(side_effect=run_serialized))
    monkeypatch.setattr(image_api, "get_serving_engine", lambda: engine)
    monkeypatch.setattr(image_api, "get_output_dir", lambda: str(tmp_path))
    monkeypatch.setattr(image_api, "IMAGE_STORE", AsyncDictStore())
    app = FastAPI()
    app.include_router(image_api.router)
    payload = {"prompt": "a cat", "n": 2, "response_format": response_format, "output_format": output_format}
    with TestClient(app) as client:
        if path.endswith("edits"):
            response = client.post(path, data=payload, files={"image": ("cat.png", b"input", "image/png")})
        else:
            response = client.post(path, json=payload)
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert len(data) == 2
        for index, item in enumerate(data):
            if response_format == "url":
                content = client.get(item["url"])
                assert content.status_code == 200
                content = content.content
            else:
                content = base64.b64decode(item["b64_json"])
            with Image.open(BytesIO(content)) as image:
                assert image.size == (16, 16)
                assert image.format == {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}[output_format]
                color = image.convert("RGB").getpixel((8, 8))
                assert color[0 if index == 0 else 2] >= 250
                assert color[2 if index == 0 else 0] <= 5
    # Keep the model's native batch, rather than making a separate inference call per image.
    generate.assert_called_once()
    assert generate.call_args.kwargs["num_videos_per_prompt"] == 2
    assert generate.call_args.kwargs["save_video"] is False
    assert generate.call_args.kwargs["return_frames"] is False
    assert generate.call_args.kwargs["return_samples"] is True
