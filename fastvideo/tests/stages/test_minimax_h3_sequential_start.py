# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for MiniMax-H3 Mac-style sequential module loading."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

import fastvideo.pipelines.composed_pipeline_base as composed_pipeline_base
from fastvideo.api.overrides import apply_overrides, parse_cli_overrides
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
    _DENOISE_MODULE_NAMES,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.tests.stages._resolved_config import make_resolved_config


class _Profiler:

    def region(self, name):
        del name
        return nullcontext()


def _stub_module(name: str) -> SimpleNamespace:
    if name in {"scheduler"}:
        return SimpleNamespace(shift=12.0, name=name)
    if name in {"audio_scheduler"}:
        return SimpleNamespace(shift=3.0, name=name)
    if name == "transformer":
        # LoRAPipeline reads exclude_lora_layers off the DiT arch config.
        return SimpleNamespace(
            name=name,
            config=SimpleNamespace(arch_config=SimpleNamespace(exclude_lora_layers=[])),
        )
    return SimpleNamespace(name=name)


def _h3_config(*, enable_stage_verification: bool = True, lazy_module_load: bool | None = None, **minimax_h3):
    """Resolved config for an unregistered model path, which carries the generic ``PipelineConfig``."""
    return make_resolved_config(PipelineConfig(), raw={
        "engine": {
            "enable_stage_verification": enable_stage_verification,
            "offload": {"lazy_module_load": lazy_module_load},
        },
        "pipeline": {"minimax_h3": minimax_h3},
    })


def _patch_pipeline_construction(monkeypatch, events: list) -> None:
    monkeypatch.setattr(
        composed_pipeline_base,
        "maybe_init_distributed_environment_and_model_parallel",
        lambda *args, **kwargs: events.append(("distributed", None)),
    )
    monkeypatch.setattr(composed_pipeline_base, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(composed_pipeline_base, "get_world_group", lambda: SimpleNamespace(local_rank=0))
    monkeypatch.setattr(composed_pipeline_base, "get_or_create_profiler", lambda trace_dir: _Profiler())
    monkeypatch.setattr(composed_pipeline_base, "warmup_sequence_parallel_communication", lambda: None)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)


def test_inference_defers_dit_and_vae_until_after_conditioning(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, resolved_config, loaded_modules=None):
        del resolved_config
        requested = list(self.required_config_modules)
        loads.append(requested)
        modules = dict(loaded_modules or {})
        for name in requested:
            modules.setdefault(name, _stub_module(name))
        return modules

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)

    pipeline = MiniMaxH3Pipeline("unused/for-this-test", _h3_config(enable_stage_verification=False,
                                                                     sequential_load=True))
    pipeline.post_init()

    assert loads, "condition modules should load during construction"
    assert "text_encoder" in loads[0]
    assert all(name not in loads[0] for name in _DENOISE_MODULE_NAMES)
    assert pipeline.get_module("text_encoder") is not None
    assert pipeline.get_module("transformer") is None
    assert list(pipeline._stage_name_mapping) == ["input_preparation_stage", "conditioning_stage"]

    condition_stage = pipeline._stage_name_mapping["conditioning_stage"]
    passthrough = lambda batch, _args: batch
    monkeypatch.setattr(pipeline._stage_name_mapping["input_preparation_stage"], "forward", passthrough)
    monkeypatch.setattr(condition_stage, "forward", passthrough)

    original_add_denoise = pipeline._add_denoise_stages

    def fake_add_denoise(*, ref2va: bool) -> None:
        original_add_denoise(ref2va=ref2va)
        for name in (
                "latent_preparation_stage",
                "denoising_stage",
                "video_decoding_stage",
                "audio_decoding_stage",
        ):
            monkeypatch.setattr(pipeline._stage_name_mapping[name], "forward", passthrough)

    monkeypatch.setattr(pipeline, "_add_denoise_stages", fake_add_denoise)

    batch = ForwardBatch(data_type="video", prompt="alpine dancer")
    out = pipeline.forward(batch, pipeline.resolved_config)

    assert out is batch
    assert len(loads) == 2
    assert "transformer" in loads[1]
    assert "vae" in loads[1]
    assert "text_encoder" not in loads[1]
    assert pipeline.get_module("text_encoder") is None
    assert condition_stage.conditioner is None
    assert pipeline.get_module("transformer") is not None
    assert pipeline._denoise_stages_ready is True

    second = pipeline.forward(ForwardBatch(data_type="video", prompt="second clip"), pipeline.resolved_config)
    assert second is not None
    assert len(loads) == 3
    assert loads[2] == ["text_encoder"]
    assert pipeline.get_module("text_encoder") is None
    assert condition_stage.conditioner is None


def test_injected_denoise_weights_skip_the_deferred_split(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, resolved_config, loaded_modules=None):
        del resolved_config
        loads.append(list(self.required_config_modules))
        return dict(loaded_modules or {})

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    injected = {name: _stub_module(name) for name in MiniMaxH3Pipeline._required_config_modules}
    MiniMaxH3Pipeline("unused/for-this-test", _h3_config(sequential_load=True), loaded_modules=injected)

    assert loads == [list(MiniMaxH3Pipeline._required_config_modules)]


def test_explicit_false_loads_encoder_dit_and_vae_together(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, resolved_config, loaded_modules=None):
        del resolved_config, loaded_modules
        loads.append(list(self.required_config_modules))
        return {name: _stub_module(name) for name in self.required_config_modules}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    MiniMaxH3Pipeline("unused/for-this-test", _h3_config(sequential_load=False))

    assert loads == [list(MiniMaxH3Pipeline._required_config_modules)]
    assert "text_encoder" in loads[0]
    assert all(name in loads[0] for name in _DENOISE_MODULE_NAMES)


def test_auto_defers_on_unified_memory(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, resolved_config, loaded_modules=None):
        del resolved_config, loaded_modules
        loads.append(list(self.required_config_modules))
        return {name: _stub_module(name) for name in self.required_config_modules}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    MiniMaxH3Pipeline("unused/for-this-test", _h3_config(lazy_module_load=False))

    assert loads
    assert "text_encoder" in loads[0]
    assert all(name not in loads[0] for name in _DENOISE_MODULE_NAMES)


def test_lazy_module_load_owns_deferral_when_both_would_arm(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, resolved_config, loaded_modules=None):
        del resolved_config, loaded_modules
        loads.append(list(self.required_config_modules))
        return {name: _stub_module(name) for name in self.required_config_modules}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    # Resolution turns lazy_module_load on for unified memory; the pipeline reads that decision.
    MiniMaxH3Pipeline("unused/for-this-test", _h3_config(lazy_module_load=True, sequential_load=True))

    assert loads == [list(MiniMaxH3Pipeline._required_config_modules)]
    assert all(name in loads[0] for name in _DENOISE_MODULE_NAMES)


def test_auto_loads_together_without_unified_memory(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, resolved_config, loaded_modules=None):
        del resolved_config, loaded_modules
        loads.append(list(self.required_config_modules))
        return {name: _stub_module(name) for name in self.required_config_modules}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    MiniMaxH3Pipeline("unused/for-this-test", _h3_config())

    assert loads == [list(MiniMaxH3Pipeline._required_config_modules)]


def test_cli_tri_state_h3_sequential_load() -> None:

    def sequential_load(argv: list[str]) -> bool | None:
        raw = apply_overrides({}, parse_cli_overrides(argv))
        return make_resolved_config(raw=raw).pipeline.minimax_h3.sequential_load

    assert sequential_load([]) is None
    assert sequential_load(["--pipeline.minimax_h3.sequential_load", "true"]) is True
    assert sequential_load(["--pipeline.minimax_h3.sequential_load", "false"]) is False


def test_taeh3_t2va_skips_video_vae_on_the_deferred_load(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, resolved_config, loaded_modules=None):
        del resolved_config
        requested = list(self.required_config_modules)
        loads.append(requested)
        modules = dict(loaded_modules or {})
        for name in requested:
            modules.setdefault(name, _stub_module(name))
        return modules

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    resolved_config = _h3_config(enable_stage_verification=False, sequential_load=True, video_decode_backend="taeh3")
    pipeline = MiniMaxH3Pipeline("unused/for-this-test", resolved_config)
    pipeline.post_init()

    condition_stage = pipeline._stage_name_mapping["conditioning_stage"]
    passthrough = lambda batch, _args: batch
    monkeypatch.setattr(pipeline._stage_name_mapping["input_preparation_stage"], "forward", passthrough)
    monkeypatch.setattr(condition_stage, "forward", passthrough)
    original_add_denoise = pipeline._add_denoise_stages

    def fake_add_denoise(*, ref2va: bool) -> None:
        original_add_denoise(ref2va=ref2va)
        for name in (
                "latent_preparation_stage",
                "denoising_stage",
                "video_decoding_stage",
                "audio_decoding_stage",
        ):
            monkeypatch.setattr(pipeline._stage_name_mapping[name], "forward", passthrough)

    monkeypatch.setattr(pipeline, "_add_denoise_stages", fake_add_denoise)
    pipeline.forward(ForwardBatch(data_type="video", prompt="alpine dancer"), pipeline.resolved_config)

    assert "text_encoder" in loads[0]
    assert all(name not in loads[0] for name in _DENOISE_MODULE_NAMES)
    assert "transformer" in loads[1]
    assert "vae" not in loads[1]
    assert pipeline.get_module("vae") is None
    assert pipeline.get_module("transformer") is not None


def test_generic_pipeline_config_does_not_crash_geometry_overlay(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)

    def fake_load(self, resolved_config, loaded_modules=None):
        del resolved_config, loaded_modules
        return {name: _stub_module(name) for name in self.required_config_modules}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    pipeline = MiniMaxH3Pipeline("unused/for-this-test", _h3_config(sequential_load=True))
    pipeline.post_init()
    assert pipeline.get_module("text_encoder") is not None


def test_resident_path_does_not_reread_encoder_on_later_request(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, resolved_config, loaded_modules=None):
        del resolved_config
        requested = list(self.required_config_modules)
        loads.append(requested)
        modules = dict(loaded_modules or {})
        for name in requested:
            modules.setdefault(name, _stub_module(name))
        return modules

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    pipeline = MiniMaxH3Pipeline(
        "unused/for-this-test",
        _h3_config(enable_stage_verification=False, lazy_module_load=False, sequential_load=False),
    )
    pipeline.post_init()
    passthrough = lambda batch, _args: batch
    for stage in pipeline._stages:
        monkeypatch.setattr(stage, "forward", passthrough)

    first = pipeline.forward(ForwardBatch(data_type="video", prompt="one"), pipeline.resolved_config)
    second = pipeline.forward(ForwardBatch(data_type="video", prompt="two"), pipeline.resolved_config)
    assert first is not None and second is not None
    assert len(loads) == 1
    assert pipeline.get_module("text_encoder") is not None
