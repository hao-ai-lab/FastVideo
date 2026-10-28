# SPDX-License-Identifier: Apache-2.0
"""Per-request community LoRA on the OpenAI video API (CPU only, fake generator)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.testclient import TestClient

from fastvideo import envs
from fastvideo.api.parser import parse_config
from fastvideo.api.schema import (
    EngineConfig,
    GeneratorConfig,
    LoRAAdapterConfig,
    QuantizationConfig,
    ServeConfig,
    ServeLoRAConfig,
)
from fastvideo.entrypoints.openai.api_server import create_app
from fastvideo.entrypoints.openai.lora import LoraRuntime, adapters_attach_as_side_branch
from fastvideo.entrypoints.openai.stores import VIDEO_STORE
from fastvideo.layers.quantization.nvfp4_config import H3_NVFP4_DIT_EXPORT_FILENAME

ARTIFACT = b"fake-video"
REALISM_PATH = "fal/MiniMax-H3-Realism-People-LoRA"


class _FakeGenerator:
    """Records each adapter change and each generation, in the order they arrive."""

    def __init__(self, failing: set[tuple[str, float]] | None = None) -> None:
        self.failing = failing or set()
        self.events: list[tuple[Any, ...]] = []

    def set_lora_adapter(self, nickname: str, path: str | None = None, strength: float = 1.0) -> None:
        self.events.append(("set", nickname, path, strength))
        if (nickname, strength) in self.failing or (nickname, None) in self.failing:
            raise RuntimeError(f"cannot load {nickname}")

    def generate(self, request: Any) -> SimpleNamespace:
        self.events.append(("generate", request.prompt))
        output = Path(request.output.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(ARTIFACT)
        return SimpleNamespace(video_path=str(output), generation_time=0.01)

    def shutdown(self) -> None:
        pass


def _adapters() -> ServeLoRAConfig:
    return ServeLoRAConfig(adapters=[
        LoRAAdapterConfig(id="realism", path=REALISM_PATH, default_strength=0.8, max_strength=1.5,
                          description="People realism", license="Apache-2.0", source_url="https://example.test/r"),
        LoRAAdapterConfig(id="orbit", path="pablodawson/Orbit", default_strength=1.0, min_strength=0.2,
                          max_strength=1.5),
    ])


def _args(lora_path: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(model_path="m", lora_path=lora_path, lora_nickname="startup", lora_strength=0.5,
                           override_pipeline_cls_name=None)


@pytest.fixture
def make_client(tmp_path: Path):
    clients: list[TestClient] = []

    def make(generator: _FakeGenerator, *, allow_list: bool = True, switching: bool = True,
             lora_path: str | None = None) -> TestClient:
        runtime = LoraRuntime(_adapters(), runtime_switching=switching) if allow_list else None
        app = create_app(_args(lora_path), str(tmp_path / "out"), generator_factory=lambda: generator,
                         lora_runtime=runtime)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)
    import asyncio
    asyncio.run(VIDEO_STORE.clear())


def _sync(client: TestClient, prompt: str = "a fox", **lora: Any):
    body: dict[str, Any] = {"prompt": prompt}
    if lora:
        body["lora"] = lora
    return client.post("/v1/videos/sync", json=body)


def _job(client: TestClient, prompt: str = "a fox", **lora: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"prompt": prompt}
    if lora:
        body["lora"] = lora
    response = client.post("/v1/videos", json=body)
    assert response.status_code == 200, response.text
    video_id = response.json()["id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/v1/videos/{video_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


# --- config ---------------------------------------------------------------------------------------------------


def _serve_raw(**extra: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {"generator": {"model_path": "/models/h3"}}
    raw.update(extra)
    return raw


def test_allow_list_parses_with_defaults_and_metadata() -> None:
    config = parse_config(ServeConfig, _serve_raw(lora={"adapters": [
        {"id": "realism", "path": REALISM_PATH, "license": "Apache-2.0"},
    ]}))

    assert config.lora is not None
    adapter = config.lora.adapters[0]
    assert (adapter.default_strength, adapter.min_strength, adapter.max_strength) == (0.8, 0.0, 2.0)
    assert adapter.license == "Apache-2.0"


def test_serve_config_has_no_allow_list_by_default() -> None:
    assert parse_config(ServeConfig, _serve_raw()).lora is None


@pytest.mark.parametrize(
    ("adapter", "message"),
    [
        ({"id": " ", "path": "p"}, "must not be empty"),
        ({"id": "a", "path": ""}, "empty path"),
        ({"id": "a", "path": "p", "default_strength": 3.0}, "min_strength <= default_strength"),
        ({"id": "a", "path": "p", "min_strength": 1.0, "default_strength": 0.8}, "min_strength <= default_strength"),
        ({"id": "a", "path": "p", "max_strength": float("inf")}, "non-finite"),
    ],
)
def test_allow_list_rejects_invalid_adapter(adapter: dict[str, Any], message: str) -> None:
    with pytest.raises(Exception, match=message):
        parse_config(ServeConfig, _serve_raw(lora={"adapters": [adapter]}))


def test_allow_list_rejects_duplicate_ids() -> None:
    adapters = [{"id": "a", "path": "p"}, {"id": "a", "path": "q"}]
    with pytest.raises(Exception, match="repeats id"):
        parse_config(ServeConfig, _serve_raw(lora={"adapters": adapters}))


def test_allow_list_with_startup_adapter_is_refused() -> None:
    raw = _serve_raw(lora={"adapters": [{"id": "a", "path": "p"}]})
    raw["generator"]["pipeline"] = {"components": {"lora_path": "/x/adapter.safetensors"}}

    with pytest.raises(Exception, match="startup adapter"):
        parse_config(ServeConfig, raw)


def test_startup_adapter_without_allow_list_still_parses() -> None:
    raw = _serve_raw()
    raw["generator"]["pipeline"] = {"components": {"lora_path": "/x/adapter.safetensors"}}

    assert parse_config(ServeConfig, raw).generator.pipeline.components.lora_path == "/x/adapter.safetensors"


@pytest.mark.parametrize(
    ("quant", "packed", "side_branch_env", "expected"),
    [("NVFP4", True, False, True), ("NVFP4", False, False, False), (None, True, False, False),
     ("FP8", True, False, False), (None, False, True, True)],
)
def test_runtime_switching_matches_side_branch_attachment(tmp_path: Path, quant: str | None, packed: bool,
                                                          side_branch_env: bool, expected: bool) -> None:
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    if packed:
        (transformer / H3_NVFP4_DIT_EXPORT_FILENAME).touch()
    quantization = None if quant is None else QuantizationConfig(transformer_quant=quant)
    config = GeneratorConfig(model_path=str(tmp_path), engine=EngineConfig(quantization=quantization))

    with envs.FASTVIDEO_LORA_SIDE_BRANCH.override(side_branch_env):
        assert adapters_attach_as_side_branch(config) is expected


# --- selector validation --------------------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["/v1/videos", "/v1/videos/sync"])
@pytest.mark.parametrize(
    ("lora", "message"),
    [
        ({"name": "nope"}, "Unknown LoRA adapter"),
        ({"name": "realism", "scale": 2.0}, "takes a scale from"),
        ({"name": "orbit", "scale": 0.0}, "takes a scale from"),
        ({"name": "realism", "scale": "high"}, "Invalid LoRA scale"),
        ({"name": "realism", "path": "/etc/passwd"}, "not accepted"),
        ({"path": "/etc/passwd"}, "not accepted"),
        ({"lora_path": REALISM_PATH, "name": "realism"}, "not accepted"),
        ({"scale": 0.5}, "must provide a name"),
    ],
)
def test_bad_selection_is_400_before_any_job_is_stored(make_client, route: str, lora: dict[str, Any],
                                                       message: str) -> None:
    generator = _FakeGenerator()
    client = make_client(generator)

    response = client.post(route, json={"prompt": "a fox", "lora": lora})

    assert response.status_code == 400
    assert message in response.json()["error"]["message"]
    assert generator.events == []
    assert client.get("/v1/videos").json()["data"] == []


def test_selector_aliases_and_default_strength(make_client) -> None:
    generator = _FakeGenerator()
    client = make_client(generator)

    assert _sync(client, lora_name="realism").status_code == 200
    assert _sync(client, adapter="orbit", lora_scale=0.5).status_code == 200

    assert [event for event in generator.events if event[0] == "set"] == [
        ("set", "realism", REALISM_PATH, 0.8),
        ("set", "orbit", "pablodawson/Orbit", 0.5),
    ]


def test_selector_without_allow_list_keeps_startup_behaviour(make_client) -> None:
    generator = _FakeGenerator()
    client = make_client(generator, allow_list=False)

    response = _sync(client, name="realism")

    assert response.status_code == 400
    assert "no startup LoRA" in response.json()["error"]["message"]
    assert _sync(client).status_code == 200
    assert [event[0] for event in generator.events] == ["generate"]


def test_startup_adapter_selector_without_allow_list_is_unchanged(make_client) -> None:
    client = make_client(_FakeGenerator(), allow_list=False, lora_path="/x/adapter.safetensors")

    assert _sync(client, name="startup", scale=0.5).status_code == 200
    other = _sync(client, name="realism")
    assert other.status_code == 400
    assert "not the startup adapter" in other.json()["error"]["message"]


# --- apply -----------------------------------------------------------------------------------------------------


def test_sync_applies_adapter_before_generate(make_client) -> None:
    generator = _FakeGenerator()
    client = make_client(generator)

    response = _sync(client, "first", name="realism", scale=0.8)

    assert response.status_code == 200
    assert response.content == ARTIFACT
    assert generator.events == [("set", "realism", REALISM_PATH, 0.8), ("generate", "first")]


def test_job_applies_adapter_before_generate(make_client) -> None:
    generator = _FakeGenerator()
    client = make_client(generator)

    job = _job(client, "first", name="realism", scale=0.8)

    assert job["status"] == "completed"
    assert generator.events == [("set", "realism", REALISM_PATH, 0.8), ("generate", "first")]


@pytest.mark.parametrize("run", [_sync, _job], ids=["sync", "job"])
def test_request_without_lora_restores_base_after_an_adapter(make_client, run) -> None:
    generator = _FakeGenerator()
    client = make_client(generator)

    run(client, "styled", name="realism", scale=0.8)
    run(client, "plain")
    run(client, "plain again")

    assert generator.events == [
        ("set", "realism", REALISM_PATH, 0.8),
        ("generate", "styled"),
        ("set", "realism", REALISM_PATH, 0.0),
        ("generate", "plain"),
        ("generate", "plain again"),
    ]


def test_base_only_server_never_loads_an_adapter(make_client) -> None:
    generator = _FakeGenerator()
    client = make_client(generator)

    _sync(client)
    _sync(client)

    assert generator.events == [("generate", "a fox"), ("generate", "a fox")]


def test_same_selection_is_not_reapplied(make_client) -> None:
    generator = _FakeGenerator()
    client = make_client(generator)

    _sync(client, name="realism", scale=0.8)
    _sync(client, name="realism", scale=0.8)

    assert [event[0] for event in generator.events] == ["set", "generate", "generate"]


def test_scale_zero_sets_the_adapter_to_zero(make_client) -> None:
    generator = _FakeGenerator()
    client = make_client(generator)

    assert _sync(client, name="realism", scale=0).status_code == 200

    assert generator.events[0] == ("set", "realism", REALISM_PATH, 0.0)


def test_failed_change_fails_the_request_and_does_not_generate(make_client) -> None:
    generator = _FakeGenerator(failing={("realism", None)})
    client = make_client(generator)

    sync = _sync(client, name="realism", scale=0.8)
    job = _job(client, name="realism", scale=0.8)

    assert sync.status_code == 500
    assert "could not be applied" in sync.json()["error"]["message"]
    assert job["status"] == "failed"
    assert all(event[0] == "set" for event in generator.events)


def test_failed_change_never_leaves_an_unrequested_adapter(make_client) -> None:
    generator = _FakeGenerator(failing={("orbit", 1.0)})
    client = make_client(generator)

    _sync(client, "styled", name="realism", scale=0.8)
    failed = _sync(client, "orbit", name="orbit")
    plain = _sync(client, "plain")

    assert failed.status_code == 500
    assert plain.status_code == 200
    # The failed change resets the previous adapter to 0, so "plain" renders the base model.
    assert generator.events[-3:] == [("set", "orbit", "pablodawson/Orbit", 1.0),
                                     ("set", "realism", REALISM_PATH, 0.0), ("generate", "plain")]


def test_unrecoverable_state_refuses_a_plain_request(make_client) -> None:
    generator = _FakeGenerator(failing={("realism", None), ("orbit", None)})
    client = make_client(generator)

    _sync(client, name="realism")
    plain = _sync(client, "plain")

    assert plain.status_code == 500
    assert "restore the base model" in plain.json()["error"]["message"]
    assert ("generate", "plain") not in generator.events


def test_unknown_state_recovers_once_an_adapter_loads(make_client) -> None:
    generator = _FakeGenerator(failing={("realism", 0.8)})
    client = make_client(generator)

    assert _sync(client, name="realism", scale=0.8).status_code == 500
    assert _sync(client, "plain").status_code == 200

    assert generator.events[-2:] == [("set", "realism", REALISM_PATH, 0.0), ("generate", "plain")]


# --- discovery -------------------------------------------------------------------------------------------------


def test_model_info_lists_adapters_without_paths(make_client) -> None:
    client = make_client(_FakeGenerator())

    body = client.get("/v1/model_info").json()

    assert body["lora"] == {
        "supported": True,
        "runtime_switching": True,
        "base_strength": 0.0,
        "adapters": [
            {"id": "realism", "default_strength": 0.8, "min_strength": 0.0, "max_strength": 1.5,
             "description": "People realism", "license": "Apache-2.0", "source_url": "https://example.test/r"},
            {"id": "orbit", "default_strength": 1.0, "min_strength": 0.2, "max_strength": 1.5},
        ],
    }
    assert REALISM_PATH not in json.dumps(body)


def test_model_info_reports_no_runtime_switching(make_client) -> None:
    client = make_client(_FakeGenerator(), switching=False)

    assert client.get("/v1/model_info").json()["lora"]["runtime_switching"] is False


def test_model_info_without_allow_list_keeps_startup_info(make_client) -> None:
    assert make_client(_FakeGenerator(), allow_list=False).get("/v1/model_info").json()["lora"] is None


def test_model_info_startup_adapter_is_unchanged(make_client) -> None:
    client = make_client(_FakeGenerator(), allow_list=False, lora_path="/x/adapter.safetensors")

    assert client.get("/v1/model_info").json()["lora"] == {"name": "startup", "path": "/x/adapter.safetensors",
                                                           "scale": 0.5}
