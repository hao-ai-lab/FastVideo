# SPDX-License-Identifier: Apache-2.0
"""Per-request LoRA selection: ``POST /v1/streamv2/generate`` and ``GET /v1/streamv2/capabilities``.

CPU-only: a fake generator records the adapter and request it receives, and a fake encoder returns fixed chunks.
"""
from __future__ import annotations

import base64
import io
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from fastvideo.api.results import GenerationResult
from fastvideo.api.schema import (
    EngineConfig,
    GeneratorConfig,
    LoRAAdapterConfig,
    QuantizationConfig,
    StreamingV2LoRAConfig,
)
from fastvideo.entrypoints.streaming_v2.lora import adapters_attach_as_side_branch
from fastvideo.entrypoints.streaming_v2.routes import generate as generate_route
from fastvideo.entrypoints.streaming_v2.server import create_app
from fastvideo.layers.quantization.nvfp4_config import H3_NVFP4_DIT_EXPORT_FILENAME

_CHUNKS = [b"init-segment", b"fragment-1"]


def _png_base64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 120, 200)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "prompt": "a lighthouse at dusk",
        "references": [{"media_type": "image", "data": _png_base64()}],
        "width": 64,
        "height": 48,
        "num_frames": 4,
        "seed": 7,
    }
    body.update(overrides)
    return body


class _FakeGenerator:
    """Records each adapter change and each generation, in the order they arrive."""

    def __init__(self, change_error: Exception | None = None) -> None:
        self.change_error = change_error
        self.calls: list[tuple[str, str, str, float]] = []
        self.requests: list[Any] = []

    def set_lora_adapter(self, lora_nickname: str, lora_path: str | None = None, strength: float = 1.0) -> None:
        self.calls.append(("set_lora_adapter", lora_nickname, lora_path or "", strength))
        if self.change_error is not None:
            raise self.change_error

    def generate(self, request: Any) -> GenerationResult:
        self.calls.append(("generate", "", "", 0.0))
        self.requests.append(request)
        frames = [np.full((48, 64, 3), 30, dtype=np.uint8)]
        return GenerationResult(frames=frames, audio=None, audio_sample_rate=None)


async def _fake_encode(frames: list[np.ndarray], fps: int, audio: Any = None, sample_rate: Any = None) -> AsyncIterator[bytes]:
    for chunk in _CHUNKS:
        yield chunk


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate_route, "encode_fmp4", _fake_encode)


def _lora_config(*adapters: LoRAAdapterConfig) -> StreamingV2LoRAConfig:
    return StreamingV2LoRAConfig(adapters=list(adapters))


def _two_adapters() -> StreamingV2LoRAConfig:
    return _lora_config(
        LoRAAdapterConfig(id="realism", path="fal/MiniMax-H3-Realism-People-LoRA", default_strength=0.8),
        LoRAAdapterConfig(id="orbit", path="pablodawson/MiniMax-H3-360-Orbit-LoRA", default_strength=1.0,
                          min_strength=0.2, max_strength=1.5, source_url="https://example.test/orbit"),
    )


def test_capabilities_report_no_lora_without_a_config() -> None:
    generator = _FakeGenerator()
    with TestClient(create_app(generator)) as client:
        body = client.get("/v1/streamv2/capabilities").json()

    assert body["lora"] == {"supported": False, "runtime_switching": False, "adapters": []}


def test_capabilities_report_each_adapter_without_its_load_path() -> None:
    generator = _FakeGenerator()
    with TestClient(create_app(generator, lora_config=_two_adapters(), lora_runtime_switching=True)) as client:
        body = client.get("/v1/streamv2/capabilities").json()

    assert body["lora"]["supported"] is True
    assert body["lora"]["runtime_switching"] is True
    assert body["lora"]["base_strength"] == 0.0
    assert body["lora"]["adapters"] == [
        {"id": "realism", "default_strength": 0.8, "min_strength": 0.0, "max_strength": 2.0},
        {"id": "orbit", "default_strength": 1.0, "min_strength": 0.2, "max_strength": 1.5,
         "source_url": "https://example.test/orbit"},
    ]
    assert "fal/MiniMax-H3-Realism-People-LoRA" not in json.dumps(body)


def test_capabilities_report_runtime_switching_only_when_the_server_can_switch() -> None:
    with TestClient(create_app(_FakeGenerator(), lora_config=_two_adapters())) as client:
        body = client.get("/v1/streamv2/capabilities").json()

    assert body["lora"]["supported"] is True
    assert body["lora"]["runtime_switching"] is False


def _generator_config(tmp_path: Path, quant: str | None, packed: bool) -> GeneratorConfig:
    transformer = tmp_path / "transformer"
    transformer.mkdir(exist_ok=True)
    if packed:
        (transformer / H3_NVFP4_DIT_EXPORT_FILENAME).touch()
    quantization = None if quant is None else QuantizationConfig(transformer_quant=quant)
    return GeneratorConfig(model_path=str(tmp_path), engine=EngineConfig(quantization=quantization))


@pytest.mark.parametrize(
    ("quant", "packed", "side_branch_env", "expected"),
    [
        ("NVFP4", True, False, True),
        ("NVFP4", False, False, False),
        (None, True, False, False),
        (None, False, False, False),
        (None, False, True, True),
    ],
)
def test_adapters_attach_as_side_branch_follows_the_checkpoint_and_env(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, quant: str | None, packed: bool, side_branch_env: bool,
        expected: bool) -> None:
    if side_branch_env:
        monkeypatch.setenv("FASTVIDEO_LORA_SIDE_BRANCH", "1")
    else:
        monkeypatch.delenv("FASTVIDEO_LORA_SIDE_BRANCH", raising=False)

    assert adapters_attach_as_side_branch(_generator_config(tmp_path, quant, packed)) is expected


def test_generate_applies_the_adapter_to_a_text_request() -> None:
    generator = _FakeGenerator()
    with TestClient(create_app(generator, lora_config=_two_adapters())) as client:
        response = client.post("/v1/streamv2/generate", json=_body(references=[], lora={"id": "realism"}))

    assert response.status_code == 200
    assert generator.calls[0][:2] == ("set_lora_adapter", "realism")
    assert generator.requests[0].inputs.references == []


def test_generate_applies_the_adapter_before_generating() -> None:
    generator = _FakeGenerator()
    with TestClient(create_app(generator, lora_config=_two_adapters())) as client:
        response = client.post("/v1/streamv2/generate", json=_body(lora={"id": "realism"}))

    assert response.status_code == 200
    assert generator.calls == [
        ("set_lora_adapter", "realism", "fal/MiniMax-H3-Realism-People-LoRA", 0.8),
        ("generate", "", "", 0.0),
    ]


def test_generate_applies_an_explicit_strength() -> None:
    generator = _FakeGenerator()
    with TestClient(create_app(generator, lora_config=_two_adapters())) as client:
        client.post("/v1/streamv2/generate", json=_body(lora={"id": "realism", "strength": 0}))

    assert generator.calls[0] == ("set_lora_adapter", "realism", "fal/MiniMax-H3-Realism-People-LoRA", 0.0)


def test_generate_switches_adapter_and_strength_between_requests() -> None:
    generator = _FakeGenerator()
    with TestClient(create_app(generator, lora_config=_two_adapters())) as client:
        client.post("/v1/streamv2/generate", json=_body(lora={"id": "realism", "strength": 0.5}))
        client.post("/v1/streamv2/generate", json=_body(lora={"id": "realism", "strength": 0.5}))
        client.post("/v1/streamv2/generate", json=_body(lora={"id": "orbit", "strength": 1.2}))

    changes = [call for call in generator.calls if call[0] == "set_lora_adapter"]
    assert changes == [
        ("set_lora_adapter", "realism", "fal/MiniMax-H3-Realism-People-LoRA", 0.5),
        ("set_lora_adapter", "orbit", "pablodawson/MiniMax-H3-360-Orbit-LoRA", 1.2),
    ]


def test_generate_without_lora_returns_the_base_model_after_an_adapter() -> None:
    generator = _FakeGenerator()
    with TestClient(create_app(generator, lora_config=_two_adapters())) as client:
        client.post("/v1/streamv2/generate", json=_body(lora={"id": "realism"}))
        client.post("/v1/streamv2/generate", json=_body())
        client.post("/v1/streamv2/generate", json=_body())

    changes = [call for call in generator.calls if call[0] == "set_lora_adapter"]
    assert changes == [
        ("set_lora_adapter", "realism", "fal/MiniMax-H3-Realism-People-LoRA", 0.8),
        ("set_lora_adapter", "realism", "fal/MiniMax-H3-Realism-People-LoRA", 0.0),
    ]


@pytest.mark.parametrize(
    ("lora", "message"),
    [
        ({"id": "unknown"}, "Unknown LoRA adapter 'unknown'; this server serves: orbit, realism."),
        ({"id": "orbit", "strength": 2}, "LoRA adapter 'orbit' takes a strength from 0.2 to 1.5; got 2."),
        ({"id": "realism", "strength": -1}, None),
        ({"id": "realism", "strength": "high"}, None),
        ({"id": "realism", "scale": 1}, None),
    ],
)
def test_generate_rejects_invalid_selections_with_http_400(lora: dict[str, Any], message: str | None) -> None:
    generator = _FakeGenerator()
    with TestClient(create_app(generator, lora_config=_two_adapters())) as client:
        response = client.post("/v1/streamv2/generate", json=_body(lora=lora))

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"
    if message is not None:
        assert response.json()["message"] == message
    assert generator.calls == []


def test_generate_rejects_a_selection_when_the_server_declares_no_lora() -> None:
    generator = _FakeGenerator()
    with TestClient(create_app(generator)) as client:
        response = client.post("/v1/streamv2/generate", json=_body(lora={"id": "realism"}))

    assert response.status_code == 400
    assert response.json() == {"code": "invalid_request",
                               "message": "This server declares no LoRA adapter; omit the lora field."}
    assert generator.calls == []


def test_generate_reports_an_adapter_that_fails_to_load_as_an_invalid_request_event() -> None:
    generator = _FakeGenerator(
        change_error=RuntimeError("Runtime LoRA adapter changes are unsupported after NVFP4 quantization removed "
                                  "the BF16 weights."))
    with TestClient(create_app(generator, lora_config=_two_adapters())) as client:
        response = client.post("/v1/streamv2/generate", json=_body(lora={"id": "realism"}))

    assert response.status_code == 200
    assert response.text.startswith("event: error\n")
    payload = json.loads(response.text.split("data: ", 1)[1])
    assert payload["code"] == "invalid_request"
    assert "realism" in payload["message"] and "could not be applied" in payload["message"]
    assert "generate" not in [call[0] for call in generator.calls]


def test_generate_retries_the_adapter_after_a_failed_load() -> None:
    generator = _FakeGenerator(change_error=RuntimeError("CUDA out of memory"))
    with TestClient(create_app(generator, lora_config=_two_adapters())) as client:
        client.post("/v1/streamv2/generate", json=_body(lora={"id": "realism"}))
        generator.change_error = None
        response = client.post("/v1/streamv2/generate", json=_body(lora={"id": "realism"}))

    assert response.status_code == 200
    assert [call[0] for call in generator.calls].count("set_lora_adapter") == 2
    assert generator.requests != []


def test_lora_adapter_config_refuses_invalid_strengths_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="min_strength <= default_strength <= max_strength"):
        LoRAAdapterConfig(id="x", path="p", min_strength=1.0, default_strength=0.5)
    with pytest.raises(ValueError, match="repeats id"):
        _lora_config(LoRAAdapterConfig(id="x", path="a"), LoRAAdapterConfig(id="x", path="b"))
    with pytest.raises(ValueError, match="empty path"):
        LoRAAdapterConfig(id="x", path="")


def test_serve_config_parses_the_lora_adapters_block() -> None:
    from fastvideo.api.parser import parse_config
    from fastvideo.api.schema import ServeConfig

    serve = parse_config(ServeConfig, {
        "generator": {"model_path": "/models/server"},
        "streaming_v2": {
            "lora": {
                "adapters": [{
                    "id": "realism",
                    "path": "fal/MiniMax-H3-Realism-People-LoRA",
                    "default_strength": 0.8,
                    "description": "photoreal people",
                }],
            },
        },
    })

    adapter = serve.streaming_v2.lora.adapters[0]
    assert (adapter.id, adapter.path, adapter.default_strength) == ("realism", "fal/MiniMax-H3-Realism-People-LoRA", 0.8)
    assert (adapter.min_strength, adapter.max_strength) == (0.0, 2.0)
    assert adapter.description == "photoreal people"
