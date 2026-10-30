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
from fastvideo.entrypoints.openai import lora as lora_module
from fastvideo.entrypoints.openai.lora import LoraRuntime, adapters_attach_as_side_branch, build_lora_runtime
from fastvideo.entrypoints.openai.stores import VIDEO_STORE
from fastvideo.layers.quantization.nvfp4_config import H3_NVFP4_DIT_EXPORT_FILENAME

ARTIFACT = b"fake-video"
REALISM_PATH = "fal/MiniMax-H3-Realism-People-LoRA"


class _FakeGenerator:
    """Records each adapter change, clear and generation, in the order they arrive."""

    def __init__(self, failing: set[tuple[str, float | None]] | None = None, clear_fails: bool = False) -> None:
        self.failing = failing or set()
        self.clear_fails = clear_fails
        self.held: set[str] = set()
        self.events: list[tuple[Any, ...]] = []

    def clear_lora_adapter(self) -> None:
        self.events.append(("clear",))
        if self.clear_fails:
            raise RuntimeError("cannot clear")

    def set_lora_adapter(self, nickname: str, path: str | None = None, strength: float = 1.0) -> None:
        self.events.append(("set", nickname, path, strength))
        if path is None and nickname not in self.held:
            raise ValueError(f"Adapter {nickname} not found in the pipeline.")
        if (nickname, strength) in self.failing or (nickname, None) in self.failing:
            raise RuntimeError(f"cannot load {nickname}")
        self.held.add(nickname)

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


def _check(adapter: LoRAAdapterConfig, unusable: frozenset[str]) -> None:
    if adapter.id in unusable:
        raise ValueError(f"{adapter.id} is missing")


def _args(lora_path: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(model_path="m", lora_path=lora_path, lora_nickname="startup", lora_strength=0.5,
                           override_pipeline_cls_name=None)


@pytest.fixture
def make_client(tmp_path: Path):
    clients: list[TestClient] = []

    def make(generator: _FakeGenerator, *, allow_list: bool = True, lora_path: str | None = None,
             unusable: frozenset[str] = frozenset()) -> TestClient:
        runtime = LoraRuntime(_adapters(), check_adapter=lambda adapter: _check(adapter, unusable)) if allow_list else None
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
    assert all(event[0] != "generate" for event in generator.events)


def test_failed_change_never_leaves_an_unrequested_adapter(make_client) -> None:
    generator = _FakeGenerator(failing={("orbit", 1.0)})
    client = make_client(generator)

    _sync(client, "styled", name="realism", scale=0.8)
    failed = _sync(client, "orbit", name="orbit")
    plain = _sync(client, "plain")

    assert failed.status_code == 500
    assert plain.status_code == 200
    # The failed change clears every layer back to the base model, so "plain" needs no further change.
    assert generator.events[-3:] == [("set", "orbit", "pablodawson/Orbit", 1.0), ("clear",), ("generate", "plain")]


def test_failed_pre_load_check_leaves_the_base_model_in_place(make_client) -> None:
    generator = _FakeGenerator()
    client = make_client(generator, unusable=frozenset({"orbit"}))

    failed = _sync(client, "orbit", name="orbit")
    plain = _sync(client, "plain")

    assert failed.status_code == 500
    assert "cannot be loaded on this server" in failed.json()["error"]["message"]
    assert plain.status_code == 200
    assert generator.events == [("generate", "plain")]


def test_failed_first_load_leaves_the_base_model_serving(make_client) -> None:
    generator = _FakeGenerator(failing={("realism", None)})
    client = make_client(generator)

    failed = _sync(client, name="realism")
    plain = _sync(client, "plain")
    again = _sync(client, "plain again")

    assert failed.status_code == 500
    assert plain.status_code == 200 and again.status_code == 200
    assert generator.events == [("set", "realism", REALISM_PATH, 0.8), ("clear",), ("generate", "plain"),
                                ("generate", "plain again")]


def test_a_failed_change_whose_clear_also_fails_refuses_until_a_clear_succeeds(make_client) -> None:
    generator = _FakeGenerator(failing={("orbit", None)}, clear_fails=True)
    client = make_client(generator)
    assert _sync(client, "styled", name="realism").status_code == 200

    assert _sync(client, name="orbit").status_code == 500
    refused = _sync(client, "plain")
    generator.clear_fails = False
    recovered = _sync(client, "plain again")

    assert refused.status_code == 500
    assert "clearing the adapter" in refused.json()["error"]["message"]
    assert ("generate", "plain") not in generator.events
    assert recovered.status_code == 200
    assert generator.events[-2:] == [("clear",), ("generate", "plain again")]


def test_a_selection_after_a_failed_change_applies_normally(make_client) -> None:
    generator = _FakeGenerator(failing={("realism", 0.8)})
    client = make_client(generator)

    assert _sync(client, name="realism", scale=0.5).status_code == 200
    assert _sync(client, name="realism", scale=0.8).status_code == 500
    assert _sync(client, name="realism", scale=0.5).status_code == 200

    assert generator.events[-4:] == [("set", "realism", REALISM_PATH, 0.8), ("clear",),
                                     ("set", "realism", REALISM_PATH, 0.5), ("generate", "a fox")]


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


def test_model_info_without_allow_list_keeps_startup_info(make_client) -> None:
    assert make_client(_FakeGenerator(), allow_list=False).get("/v1/model_info").json()["lora"] is None


def test_model_info_startup_adapter_is_unchanged(make_client) -> None:
    client = make_client(_FakeGenerator(), allow_list=False, lora_path="/x/adapter.safetensors")

    assert client.get("/v1/model_info").json()["lora"] == {"name": "startup", "path": "/x/adapter.safetensors",
                                                           "scale": 0.5}


def _serve_config(tmp_path: Path, *, quant: str | None = "NVFP4", packed: bool = True) -> ServeConfig:
    transformer = tmp_path / "model" / "transformer"
    transformer.mkdir(parents=True)
    if packed:
        (transformer / H3_NVFP4_DIT_EXPORT_FILENAME).touch()
    quantization = None if quant is None else QuantizationConfig(transformer_quant=quant)
    generator = GeneratorConfig(model_path=str(tmp_path / "model"), engine=EngineConfig(quantization=quantization))
    return ServeConfig(generator=generator, lora=_adapters())


def test_build_lora_runtime_checks_every_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checked: list[str] = []
    monkeypatch.setattr(lora_module, "check_adapter_file", lambda adapter: checked.append(adapter.id))

    assert build_lora_runtime(_serve_config(tmp_path)) is not None
    assert checked == ["realism", "orbit"]


def test_build_lora_runtime_refuses_merged_lora_layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lora_module, "check_adapter_file", lambda adapter: None)

    with envs.FASTVIDEO_LORA_SIDE_BRANCH.override(False), pytest.raises(ValueError, match="side-branch"):
        build_lora_runtime(_serve_config(tmp_path, quant=None, packed=False))


def test_build_lora_runtime_refuses_an_unusable_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(adapter: LoRAAdapterConfig) -> None:
        raise ValueError(f"LoRA adapter {adapter.id!r} carries dense replacement tensors")

    monkeypatch.setattr(lora_module, "check_adapter_file", refuse)

    with pytest.raises(ValueError, match="dense replacement"):
        build_lora_runtime(_serve_config(tmp_path))


def test_build_lora_runtime_is_none_without_an_allow_list(tmp_path: Path) -> None:
    config = _serve_config(tmp_path)

    assert build_lora_runtime(ServeConfig(generator=config.generator)) is None


@pytest.mark.parametrize(("file_name", "dense", "message"), [
    ("adapter.safetensors", True, "dense replacement"),
    (None, False, "not a file"),
])
def test_check_adapter_file_refuses_what_the_pipeline_would(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            file_name: str | None, dense: bool, message: str) -> None:
    target = tmp_path
    if file_name is not None:
        target = tmp_path / file_name
        target.touch()
    monkeypatch.setattr(lora_module, "maybe_download_lora", lambda path: str(target))
    monkeypatch.setattr(lora_module.DenseLoRAPatch, "from_adapter", lambda path: object() if dense else None)

    with pytest.raises(ValueError, match=message):
        lora_module.check_adapter_file(LoRAAdapterConfig(id="x", path="org/x"))


def test_side_branch_check_resolves_a_hub_model_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / H3_NVFP4_DIT_EXPORT_FILENAME).touch()
    requested: list[tuple[str, list[str]]] = []

    def resolve(model_path: str, allow_patterns: list[str]) -> str:
        requested.append((model_path, allow_patterns))
        return str(tmp_path)

    monkeypatch.setattr(lora_module, "maybe_download_model", resolve)
    config = GeneratorConfig(model_path="org/model",
                             engine=EngineConfig(quantization=QuantizationConfig(transformer_quant="NVFP4")))

    with envs.FASTVIDEO_LORA_SIDE_BRANCH.override(False):
        assert adapters_attach_as_side_branch(config) is True
    assert requested == [("org/model", [f"transformer/{H3_NVFP4_DIT_EXPORT_FILENAME}"])]
