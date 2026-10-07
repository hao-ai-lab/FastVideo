# SPDX-License-Identifier: Apache-2.0
"""Reject mismatched image model names before invoking the generator."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastvideo.entrypoints.openai import image_api
from fastvideo.entrypoints.openai.stores import AsyncDictStore

SERVED_MODEL_NAME = "my-image-model"


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
    monkeypatch.setattr(image_api, "get_served_model_name", lambda: SERVED_MODEL_NAME)
    monkeypatch.setattr(
        image_api,
        "get_server_args",
        lambda: SimpleNamespace(lora_path=None, lora_nickname="default"),
    )
    monkeypatch.setattr(image_api, "IMAGE_STORE", AsyncDictStore())
    app = FastAPI()
    app.include_router(image_api.router)
    with TestClient(app) as client:
        yield client, engine


@pytest.mark.parametrize("path", ["/v1/images", "/v1/images/generations"])
def test_mismatched_model_does_not_generate(image_client, path, tmp_path):
    client, engine = image_client
    response = client.post(path, json={"prompt": "a cat", "model": "other-model"})

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Model mismatch: request specifies 'other-model'; this server provides my-image-model.")
    engine.run_serialized.assert_not_awaited()
    engine.generator.generate_video.assert_not_called()
    assert not (tmp_path / "images").exists()


def test_edits_mismatched_model_does_not_generate(image_client, tmp_path):
    client, engine = image_client
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "a cat", "model": "other-model"},
        files={"image": ("cat.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Model mismatch: request specifies 'other-model'; this server provides my-image-model.")
    engine.run_serialized.assert_not_awaited()
    engine.generator.generate_video.assert_not_called()
    assert not (tmp_path / "uploads").exists()


@pytest.mark.parametrize("path", ["/v1/images", "/v1/images/generations"])
@pytest.mark.parametrize("model", [None, SERVED_MODEL_NAME])
def test_omitted_or_matching_model_still_generates(image_client, path, model):
    client, engine = image_client
    payload = {"prompt": "a cat", "response_format": "b64_json"}
    if model is not None:
        payload["model"] = model

    response = client.post(path, json=payload)

    assert response.status_code == 200, response.text
    engine.run_serialized.assert_awaited_once()
    engine.generator.generate_video.assert_called_once()


@pytest.mark.parametrize("model", [None, SERVED_MODEL_NAME])
def test_edits_omitted_or_matching_model_still_generates(image_client, model):
    client, engine = image_client
    data = {"prompt": "a cat", "response_format": "b64_json"}
    if model is not None:
        data["model"] = model

    response = client.post(
        "/v1/images/edits",
        data=data,
        files={"image": ("cat.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200, response.text
    engine.run_serialized.assert_awaited_once()
    engine.generator.generate_video.assert_called_once()
