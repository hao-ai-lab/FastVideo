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


def test_edits_invalid_response_format_does_not_generate(image_client, tmp_path):
    client, engine = image_client
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "a cat", "response_format": "invalid"},
        files={"image": ("cat.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "response_format=invalid is not supported"
    engine.run_serialized.assert_not_awaited()
    engine.generator.generate_video.assert_not_called()
    assert not (tmp_path / "uploads").exists()


@pytest.mark.parametrize(
    "payload,expected",
    [
        pytest.param({"prompt": "a cat"}, "url", id="omitted-uses-protocol-default"),
        pytest.param({"prompt": "a cat", "response_format": None}, "b64_json", id="null-falls-back-to-b64-json"),
        pytest.param({"prompt": "a cat", "response_format": ""}, "b64_json", id="empty-falls-back-to-b64-json"),
        pytest.param({"prompt": "a cat", "response_format": "b64_json"}, "b64_json", id="b64-json"),
        pytest.param({"prompt": "a cat", "response_format": "B64_JSON"}, "b64_json", id="b64-json-uppercase"),
        pytest.param({"prompt": "a cat", "response_format": "url"}, "url", id="url"),
        pytest.param({"prompt": "a cat", "response_format": "URL"}, "url", id="url-uppercase"),
    ],
)
def test_supported_response_formats_still_generate(image_client, payload, expected):
    client, engine = image_client
    response = client.post("/v1/images", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    if expected == "url":
        assert body["data"][0]["url"] == f"/v1/images/{body['id']}/content"
    else:
        assert base64.b64decode(body["data"][0]["b64_json"]) == b"test-image-content"
    engine.run_serialized.assert_awaited_once()
    engine.generator.generate_video.assert_called_once()


def test_edits_supported_response_format_still_generates(image_client):
    client, engine = image_client
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "a cat", "response_format": "b64_json"},
        files={"image": ("cat.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200, response.text
    assert base64.b64decode(response.json()["data"][0]["b64_json"]) == b"test-image-content"
    engine.run_serialized.assert_awaited_once()
    engine.generator.generate_video.assert_called_once()
