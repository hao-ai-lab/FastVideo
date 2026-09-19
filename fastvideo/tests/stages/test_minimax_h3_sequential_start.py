# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for MiniMax-H3 Mac-style sequential module loading."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

import fastvideo.pipelines.composed_pipeline_base as composed_pipeline_base
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.utils import FlexibleArgumentParser
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
    _DENOISE_MODULE_NAMES,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


class _Profiler:

    def region(self, name):
        del name
        return nullcontext()


def _stub_module(name: str) -> SimpleNamespace:
    if name in {"scheduler"}:
        module = SimpleNamespace(shift=12.0, name=name)
    elif name in {"audio_scheduler"}:
        module = SimpleNamespace(shift=3.0, name=name)
    elif name == "transformer":
        # LoRAPipeline reads exclude_lora_layers off the DiT arch config.
        module = SimpleNamespace(
            name=name,
            config=SimpleNamespace(arch_config=SimpleNamespace(exclude_lora_layers=[])),
        )
    else:
        module = SimpleNamespace(name=name)

    def to(device):
        module.device = device
        module.moved_to.append(device)
        return module

    module.device = None
    module.moved_to = []
    module.to = to
    return module


def _patch_pipeline_construction(monkeypatch, events: list, *, unified_memory: bool = False) -> None:
    monkeypatch.setattr(
        composed_pipeline_base,
        "maybe_init_distributed_environment_and_model_parallel",
        lambda *args, **kwargs: events.append(("distributed", None)),
    )
    monkeypatch.setattr(composed_pipeline_base, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(composed_pipeline_base, "get_world_group", lambda: SimpleNamespace(local_rank=0))
    monkeypatch.setattr(composed_pipeline_base, "get_or_create_profiler", lambda trace_dir: _Profiler())
    monkeypatch.setattr(composed_pipeline_base, "warmup_sequence_parallel_communication", lambda: None)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: unified_memory)
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)


def test_inference_defers_dit_and_vae_until_after_conditioning(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del fastvideo_args
        requested = list(self.required_config_modules)
        loads.append(requested)
        modules = dict(loaded_modules or {})
        for name in requested:
            modules.setdefault(name, _stub_module(name))
        return modules

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)

    args = FastVideoArgs(
        model_path="unused/for-this-test",
        enable_stage_verification=False,
        h3_sequential_load=True,
    )
    pipeline = MiniMaxH3Pipeline("unused/for-this-test", args)
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

    encoder = pipeline.get_module("text_encoder")
    batch = ForwardBatch(data_type="video", prompt="alpine dancer")
    out = pipeline.forward(batch, args)

    assert out is batch
    assert len(loads) == 2
    assert "transformer" in loads[1]
    assert "vae" in loads[1]
    assert "text_encoder" not in loads[1]
    assert pipeline.get_module("text_encoder") is encoder
    assert encoder.moved_to[-1] == "cpu"
    assert condition_stage.conditioner is not None
    assert pipeline.get_module("transformer") is not None
    assert pipeline._denoise_stages_ready is True

    moves_before_second = len(encoder.moved_to)
    second = pipeline.forward(ForwardBatch(data_type="video", prompt="second clip"), args)
    assert second is not None
    assert len(loads) == 2
    assert pipeline.get_module("text_encoder") is encoder
    assert condition_stage.conditioner is not None
    assert len(encoder.moved_to) > moves_before_second
    assert encoder.moved_to[moves_before_second] == torch.device("cpu")


def test_sequential_skips_host_offload_for_dtensor_params(monkeypatch) -> None:
    class _FakeDTensor:
        pass

    monkeypatch.setattr(
        "fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline.DTensor",
        _FakeDTensor,
    )
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def _dtensor_stub(name: str) -> SimpleNamespace:
        module = _stub_module(name)

        def parameters():
            yield _FakeDTensor()

        module.parameters = parameters
        return module

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del fastvideo_args
        requested = list(self.required_config_modules)
        loads.append(requested)
        modules = dict(loaded_modules or {})
        for name in requested:
            if name in modules:
                continue
            modules[name] = _dtensor_stub(name) if name == "text_encoder" else _stub_module(name)
        return modules

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    args = FastVideoArgs(
        model_path="unused/for-this-test",
        enable_stage_verification=False,
        h3_sequential_load=True,
    )
    pipeline = MiniMaxH3Pipeline("unused/for-this-test", args)
    pipeline.post_init()

    passthrough = lambda batch, _args: batch
    monkeypatch.setattr(pipeline._stage_name_mapping["input_preparation_stage"], "forward", passthrough)
    monkeypatch.setattr(pipeline._stage_name_mapping["conditioning_stage"], "forward", passthrough)
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
    encoder = pipeline.get_module("text_encoder")
    pipeline.forward(ForwardBatch(data_type="video", prompt="alpine dancer"), args)
    pipeline.forward(ForwardBatch(data_type="video", prompt="second clip"), args)

    assert len(loads) == 2
    assert encoder.moved_to == []
    transformer = pipeline.get_module("transformer")
    assert transformer is not None
    assert transformer.moved_to[-1] == torch.device("cpu")


def test_sequential_skips_host_offload_when_dense_params_precede_dtensors(monkeypatch) -> None:
    class _FakeDTensor:
        pass

    monkeypatch.setattr(
        "fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline.DTensor",
        _FakeDTensor,
    )
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def _mixed_stub(name: str) -> SimpleNamespace:
        module = _stub_module(name)

        def parameters():
            yield torch.zeros(1)
            yield _FakeDTensor()

        module.parameters = parameters
        return module

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del fastvideo_args
        requested = list(self.required_config_modules)
        loads.append(requested)
        modules = dict(loaded_modules or {})
        for name in requested:
            if name in modules:
                continue
            modules[name] = _mixed_stub(name) if name == "text_encoder" else _stub_module(name)
        return modules

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    args = FastVideoArgs(
        model_path="unused/for-this-test",
        enable_stage_verification=False,
        h3_sequential_load=True,
    )
    pipeline = MiniMaxH3Pipeline("unused/for-this-test", args)
    pipeline.post_init()

    passthrough = lambda batch, _args: batch
    monkeypatch.setattr(pipeline._stage_name_mapping["input_preparation_stage"], "forward", passthrough)
    monkeypatch.setattr(pipeline._stage_name_mapping["conditioning_stage"], "forward", passthrough)
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
    encoder = pipeline.get_module("text_encoder")
    pipeline.forward(ForwardBatch(data_type="video", prompt="alpine dancer"), args)
    assert encoder.moved_to == []


def test_unified_memory_sequential_deletes_encoder_and_reloads(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events, unified_memory=True)
    loads: list[list[str]] = []

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del fastvideo_args
        requested = list(self.required_config_modules)
        loads.append(requested)
        modules = dict(loaded_modules or {})
        for name in requested:
            modules.setdefault(name, _stub_module(name))
        return modules

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    args = FastVideoArgs(
        model_path="unused/for-this-test",
        enable_stage_verification=False,
        h3_sequential_load=True,
        lazy_module_load=False,
    )
    pipeline = MiniMaxH3Pipeline("unused/for-this-test", args)
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
    pipeline.forward(ForwardBatch(data_type="video", prompt="alpine dancer"), args)

    assert pipeline.get_module("text_encoder") is None
    assert condition_stage.conditioner is None
    assert len(loads) == 2

    pipeline.forward(ForwardBatch(data_type="video", prompt="second clip"), args)
    assert len(loads) == 3
    assert loads[2] == ["text_encoder"]
    assert pipeline.get_module("text_encoder") is None
    assert condition_stage.conditioner is None


def test_injected_denoise_weights_skip_the_deferred_split(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del fastvideo_args
        loads.append(list(self.required_config_modules))
        return dict(loaded_modules or {})

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    injected = {name: _stub_module(name) for name in MiniMaxH3Pipeline._required_config_modules}
    args = FastVideoArgs(model_path="unused/for-this-test", h3_sequential_load=True)
    MiniMaxH3Pipeline("unused/for-this-test", args, loaded_modules=injected)

    assert loads == [list(MiniMaxH3Pipeline._required_config_modules)]


def test_explicit_false_loads_encoder_dit_and_vae_together(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del fastvideo_args, loaded_modules
        loads.append(list(self.required_config_modules))
        return {name: _stub_module(name) for name in self.required_config_modules}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    args = FastVideoArgs(model_path="unused/for-this-test", h3_sequential_load=False)
    MiniMaxH3Pipeline("unused/for-this-test", args)

    assert loads == [list(MiniMaxH3Pipeline._required_config_modules)]
    assert "text_encoder" in loads[0]
    assert all(name in loads[0] for name in _DENOISE_MODULE_NAMES)


def test_auto_defers_on_unified_memory(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del fastvideo_args, loaded_modules
        loads.append(list(self.required_config_modules))
        return {name: _stub_module(name) for name in self.required_config_modules}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    args = FastVideoArgs(model_path="unused/for-this-test", lazy_module_load=False)
    MiniMaxH3Pipeline("unused/for-this-test", args)

    assert loads
    assert "text_encoder" in loads[0]
    assert all(name not in loads[0] for name in _DENOISE_MODULE_NAMES)


def test_lazy_module_load_owns_deferral_when_both_would_arm(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del fastvideo_args, loaded_modules
        loads.append(list(self.required_config_modules))
        return {name: _stub_module(name) for name in self.required_config_modules}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    args = FastVideoArgs(model_path="unused/for-this-test", h3_sequential_load=True)
    MiniMaxH3Pipeline("unused/for-this-test", args)

    assert loads == [list(MiniMaxH3Pipeline._required_config_modules)]
    assert all(name in loads[0] for name in _DENOISE_MODULE_NAMES)


def test_auto_loads_together_without_unified_memory(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del fastvideo_args, loaded_modules
        loads.append(list(self.required_config_modules))
        return {name: _stub_module(name) for name in self.required_config_modules}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    args = FastVideoArgs(model_path="unused/for-this-test")
    MiniMaxH3Pipeline("unused/for-this-test", args)

    assert loads == [list(MiniMaxH3Pipeline._required_config_modules)]


def test_cli_tri_state_h3_sequential_load() -> None:
    parser = FastVideoArgs.add_cli_args(FlexibleArgumentParser())
    assert parser.parse_args([]).h3_sequential_load is None
    assert parser.parse_args(["--h3-sequential-load"]).h3_sequential_load is True
    assert parser.parse_args(["--no-h3-sequential-load"]).h3_sequential_load is False


def test_taeh3_t2va_skips_video_vae_on_the_deferred_load(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del fastvideo_args
        requested = list(self.required_config_modules)
        loads.append(requested)
        modules = dict(loaded_modules or {})
        for name in requested:
            modules.setdefault(name, _stub_module(name))
        return modules

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    args = FastVideoArgs(
        model_path="unused/for-this-test",
        enable_stage_verification=False,
        h3_sequential_load=True,
        video_decode_backend="taeh3",
    )
    pipeline = MiniMaxH3Pipeline("unused/for-this-test", args)
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
    pipeline.forward(ForwardBatch(data_type="video", prompt="alpine dancer"), args)

    assert "text_encoder" in loads[0]
    assert all(name not in loads[0] for name in _DENOISE_MODULE_NAMES)
    assert "transformer" in loads[1]
    assert "vae" not in loads[1]
    assert pipeline.get_module("vae") is None
    assert pipeline.get_module("transformer") is not None


def test_generic_pipeline_config_does_not_crash_geometry_overlay(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del fastvideo_args, loaded_modules
        return {name: _stub_module(name) for name in self.required_config_modules}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    args = FastVideoArgs(model_path="unused/for-this-test", h3_sequential_load=True)
    pipeline = MiniMaxH3Pipeline("unused/for-this-test", args)
    pipeline.post_init()
    assert pipeline.get_module("text_encoder") is not None


def test_resident_path_does_not_reread_encoder_on_later_request(monkeypatch) -> None:
    events: list = []
    _patch_pipeline_construction(monkeypatch, events)
    loads: list[list[str]] = []

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del fastvideo_args
        requested = list(self.required_config_modules)
        loads.append(requested)
        modules = dict(loaded_modules or {})
        for name in requested:
            modules.setdefault(name, _stub_module(name))
        return modules

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    args = FastVideoArgs(
        model_path="unused/for-this-test",
        enable_stage_verification=False,
        h3_sequential_load=False,
        lazy_module_load=False,
    )
    pipeline = MiniMaxH3Pipeline("unused/for-this-test", args)
    pipeline.post_init()
    passthrough = lambda batch, _args: batch
    for stage in pipeline._stages:
        monkeypatch.setattr(stage, "forward", passthrough)

    first = pipeline.forward(ForwardBatch(data_type="video", prompt="one"), args)
    second = pipeline.forward(ForwardBatch(data_type="video", prompt="two"), args)
    assert first is not None and second is not None
    assert len(loads) == 1
    assert pipeline.get_module("text_encoder") is not None
