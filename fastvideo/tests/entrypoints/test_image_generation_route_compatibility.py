# SPDX-License-Identifier: Apache-2.0
"""Image generation route compatibility with a mocked generator."""

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
def test_image_generation_routes(path, response_format, monkeypatch, tmp_path):
    image_bytes = b"test-image-content"
    generate = Mock(side_effect=lambda **kwargs: Path(kwargs["output_path"]).write_bytes(image_bytes))

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
        response = client.post(path, json={"prompt": "a cat", "response_format": response_format})
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
