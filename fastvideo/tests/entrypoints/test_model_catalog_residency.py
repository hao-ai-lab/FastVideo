# SPDX-License-Identifier: Apache-2.0
"""Resident component ownership and request isolation without loading real weights."""
from pathlib import Path
from types import SimpleNamespace as NS
import weakref

import pytest

from fastvideo.entrypoints.openai.model_catalog import ModelProfile, ResidentModels
from fastvideo.entrypoints.openai.serving_engine import OpenAIServingEngine
from fastvideo.entrypoints.openai.state import request_model_profile
from fastvideo.worker.uniproc_executor import UniprocExecutor


class Weight:
    pass


def profiles(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "weights.bin").write_bytes(b"shared encoder")
    result = []
    for name in ("pika", "omni"):
        root = tmp_path / name
        root.mkdir()
        (root / "text_encoder").symlink_to(shared, target_is_directory=True)
        args = NS(model_path=str(root), override_pipeline_cls_name=name, text_encoder_weights=None,
                  pipeline_config=NS(text_encoder_configs=(), text_encoder_precisions=("bf16",)),
                  num_gpus=1, tp_size=1, sp_size=1)
        offload = NS(lazy_module_load=False, dit=False, vae=False, text_encoder=False)
        config = NS(generator=NS(model_path=str(root), engine=NS(offload=offload)))
        result.append(ModelProfile(name, "ref2va", root / "serve.json", config, args, {}))
    return result


def manager(tmp_path, monkeypatch):
    first, second = profiles(tmp_path)
    encoder, transformer = Weight(), Weight()
    previous = weakref.ref(transformer)
    executor = object.__new__(UniprocExecutor)
    executor.driver_worker = NS(worker=NS(pipeline=NS(modules={"text_encoder": encoder, "transformer": transformer})))
    executor._generation_open = False
    executor.shutting_down = True  # the fixture does not own distributed process groups
    del transformer

    def attach(args, components):
        assert previous() is None, "The previous transformer must be freed before allocating the next one"
        executor.driver_worker.worker.pipeline = NS(modules={**components, "transformer": Weight()})

    monkeypatch.setattr(executor, "attach_model", attach)
    generator = NS(executor=executor, fastvideo_args=first.args, config=None)
    catalog = NS(profiles={(p.model, p.task): p for p in (first, second)}, shared_components=("text_encoder",))
    runtime = ResidentModels(generator, catalog, first.args, first.model)
    return runtime, first, second, encoder


def test_switch_retains_encoder_and_releases_old_transformer(tmp_path, monkeypatch):
    runtime, first, second, encoder = manager(tmp_path, monkeypatch)
    runtime.activate(second)
    assert runtime.active is second
    assert runtime.generator.executor.driver_worker.worker.pipeline.modules["text_encoder"] is encoder
    assert runtime.reused == ["text_encoder"]
    runtime.activate(first)
    assert runtime.active is first
    assert runtime.generator.executor.driver_worker.worker.pipeline.modules["text_encoder"] is encoder


def test_explicit_unload_keeps_only_shared_objects(tmp_path, monkeypatch):
    runtime, _, second, encoder = manager(tmp_path, monkeypatch)
    runtime.unload()
    assert runtime.active is None
    assert runtime.shared == {"text_encoder": encoder}
    assert runtime.generator.executor.driver_worker.worker.pipeline is None
    runtime.activate(second)
    assert runtime.generator.executor.driver_worker.worker.pipeline.modules["text_encoder"] is encoder


def test_missing_local_model_does_not_unload_current_model(tmp_path, monkeypatch):
    runtime, first, second, _ = manager(tmp_path, monkeypatch)
    second.config.generator.model_path = str(tmp_path / "absent")
    with pytest.raises(ValueError, match="not installed"):
        runtime.activate(second)
    assert runtime.active is first


def test_changed_shared_weights_are_not_reused(tmp_path, monkeypatch):
    runtime, _, second, _ = manager(tmp_path, monkeypatch)
    (tmp_path / "shared/weights.bin").write_bytes(b"a different encoder")
    runtime.activate(second)
    assert runtime.reused == []


def test_failed_attach_reports_no_loaded_model(tmp_path, monkeypatch):
    runtime, _, second, _ = manager(tmp_path, monkeypatch)

    def fail(*_):
        raise RuntimeError("bad checkpoint")

    monkeypatch.setattr(runtime.generator.executor, "attach_model", fail)
    with pytest.raises(RuntimeError, match="bad checkpoint"):
        runtime.activate(second)
    assert runtime.status()["loaded_model"] is None
    assert runtime.status()["loading_model"] is None
    assert runtime.status()["error"] == "bad checkpoint"


def test_same_profile_does_not_reload(tmp_path, monkeypatch):
    runtime, first, _, _ = manager(tmp_path, monkeypatch)
    pipeline = runtime.generator.executor.driver_worker.worker.pipeline
    runtime.activate(first)
    assert runtime.generator.executor.driver_worker.worker.pipeline is pipeline


def test_queued_requests_keep_their_profile_and_serialize():
    import asyncio

    seen = []
    runtime = NS(active=None)

    def activate(profile):
        runtime.active = profile

    runtime.activate = activate

    def generate(_):
        seen.append(runtime.active)
        return runtime.active

    engine = OpenAIServingEngine(NS(generate=generate), model_manager=runtime)

    async def submit(profile):
        token = request_model_profile.set(profile)
        try:
            return await engine.generate(None)
        finally:
            request_model_profile.reset(token)

    async def run():
        return await asyncio.gather(submit("pika"), submit("omni"), submit("pika"))

    results = asyncio.run(run())
    assert results == ["pika", "omni", "pika"]
    assert seen == results


def test_a_lazy_profile_shares_nothing_and_frees_the_previous_model(tmp_path, monkeypatch):
    runtime, first, second, encoder = manager(tmp_path, monkeypatch)
    second.config.generator.engine.offload.lazy_module_load = True
    runtime.activate(second)
    assert runtime.reused == []
    assert "text_encoder" not in runtime.generator.executor.driver_worker.worker.pipeline.modules
    # Back to the eager model: the lazy one handed nothing over either.
    runtime.activate(first)
    assert runtime.reused == []


def test_catalog_environment_goes_through_the_registry(tmp_path, monkeypatch):
    import fastvideo.envs as envs

    runtime, first, second, _ = manager(tmp_path, monkeypatch)
    second.environment = {"FASTVIDEO_H3_SOL_ATTN": "sol_engine_fast"}
    runtime._baseline = {"FASTVIDEO_H3_SOL_ATTN": (False, "off")}
    with envs.FASTVIDEO_H3_SOL_ATTN.override(None):
        runtime.activate(second)
        assert envs.FASTVIDEO_H3_SOL_ATTN.get() == "sol_engine_fast"
        runtime.activate(first)
        assert not envs.FASTVIDEO_H3_SOL_ATTN.is_set()
