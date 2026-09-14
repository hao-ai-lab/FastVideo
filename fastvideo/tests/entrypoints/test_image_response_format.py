# SPDX-License-Identifier: Apache-2.0
"""Validate image response formats before invoking the generator."""

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastvideo.entrypoints.openai import image_api
from fastvideo.entrypoints.openai.stores import AsyncDictStore


@pytest.fixture
def image_client(monkeypatch, tmp_path):
    generate = Mock(side_effect=lambda **kwargs: Path(kwargs["output_path"]).write_bytes(b"test-image-content"))

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
        yield client, engine


@pytest.mark.parametrize("response_format", ["invalid", "INVALID"])
def test_invalid_response_format_does_not_generate(image_client, response_format, tmp_path):
    client, engine = image_client
    response = client.post("/v1/images", json={"prompt": "a cat", "response_format": response_format})

    assert response.status_code == 400
    assert response.json()["detail"] == "response_format=invalid is not supported"
    engine.run_serialized.assert_not_awaited()
    engine.generator.generate_video.assert_not_called()
    assert not (tmp_path / "images").exists()


@pytest.mark.parametrize("response_format", [None, "", "b64_json", "B64_JSON", "url", "URL"])
def test_supported_response_formats_still_generate(image_client, response_format):
    client, engine = image_client
    response = client.post("/v1/images", json={"prompt": "a cat", "response_format": response_format})

    assert response.status_code == 200, response.text
    body = response.json()
    if response_format and response_format.lower() == "url":
        assert body["data"][0]["url"] == f"/v1/images/{body['id']}/content"
    else:
        assert base64.b64decode(body["data"][0]["b64_json"]) == b"test-image-content"
    engine.run_serialized.assert_awaited_once()
    engine.generator.generate_video.assert_called_once()
