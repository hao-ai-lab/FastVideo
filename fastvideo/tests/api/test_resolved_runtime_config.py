# SPDX-License-Identifier: Apache-2.0
"""The resolved runtime config: materialization, typed attributes, device policy, overrides, and roots."""
from __future__ import annotations

import json
import pickle
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from fastvideo.api import device_policy
from fastvideo.api.errors import ConfigValidationError
from fastvideo.api.inference_resolution import resolve_inference_config
from fastvideo.api.parser import parse_config
from fastvideo.api.schema import ExecutionMode, GeneratorConfig, WorkloadType
from fastvideo.api.training_schema import (PreprocessRunConfig, TrainingRunConfig, load_resolved_run_config,
                                           resolve_preprocess_config, resolve_training_config)
from fastvideo.tests.api.config_snapshot import isolated_environment

WAN_T2V = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _resolve(raw, env_values=None):
    with isolated_environment(env_values):
        return resolve_inference_config(raw)


def _resolve_with_device_policy(raw):
    """Resolve inside the isolation with the device-policy steps on, which the isolation turns off like downloads."""
    with isolated_environment(), patch.object(device_policy, "APPLY_DEVICE_POLICY", True):
        return resolve_inference_config(raw)


def test_resolution_builds_and_freezes_the_pipeline_config_once(monkeypatch):
    from fastvideo.configs.pipelines.base import PipelineConfig

    builds = []
    original = PipelineConfig.from_source.__func__
    monkeypatch.setattr(PipelineConfig, "from_source",
                        classmethod(lambda cls, *args: builds.append(args) or original(cls, *args)))
    resolved = _resolve({"model_path": WAN_T2V, "pipeline": {"flow_shift": 5.0, "dit": {"prefix": "Probe"}}})

    assert len(builds) == 1
    assert type(resolved.pipeline_config).__name__ == "WanT2V480PConfig"
    assert resolved.pipeline_config.flow_shift == resolved.pipeline.flow_shift == 5.0
    assert resolved.pipeline_config.dit_config.prefix == "Probe"
    with pytest.raises(AttributeError, match="read-only"):
        resolved.pipeline_config.flow_shift = 1.0


def test_only_typed_paths_and_reserved_names_are_attributes():
    resolved = _resolve({"model_path": WAN_T2V, "pipeline": {"workload_type": "t2v"}})

    assert resolved.pipeline.workload_type is WorkloadType.T2V
    assert resolved.mode is ExecutionMode.INFERENCE and resolved.inference_mode and not resolved.training_mode
    assert getattr(resolved, "log_level_progress", "default") == "default"
    for name in ("num_gpus", "dit_cpu_offload", "workload_type", "learning_rate", "preprocess_config", "model_paths"):
        with pytest.raises(AttributeError, match=f"GeneratorConfig has no field '{name}'"):
            getattr(resolved, name)
    with pytest.raises(AttributeError, match="read-only"):
        resolved.model_paths = {}


def test_experimental_key_with_a_typed_path_is_rejected():
    with pytest.raises(ValueError, match=r"pipeline.experimental.flow_shift -> pipeline.flow_shift"):
        _resolve({"model_path": WAN_T2V, "pipeline": {"experimental": {"flow_shift": 5.0}}})


def test_fields_without_a_runtime_reader_are_rejected():
    with pytest.raises(NotImplementedError, match="pipeline.components.vae_weights"):
        _resolve({"model_path": WAN_T2V, "pipeline": {"components": {"vae_weights": "/weights/vae.safetensors"}}})


def test_experimental_model_only_attribute_reaches_the_pipeline_config():
    resolved = _resolve({"model_path": WAN_T2V, "pipeline": {"experimental": {"flow_shift_sr": 2.0}}})

    assert resolved.pipeline_config.flow_shift_sr == 2.0
    assert resolved.pipeline.experimental["flow_shift_sr"] == 2.0


def test_overrides_and_pickling_carry_the_frozen_pipeline_config():
    resolved = _resolve({"model_path": WAN_T2V})

    overridden = resolved.with_override("test:source", {"engine.offload.vae": False})
    copy = pickle.loads(pickle.dumps(overridden))

    assert copy.pipeline_config._frozen
    assert copy.engine.offload.vae is False and resolved.engine.offload.vae is True
    with pytest.raises(AttributeError, match="read-only"):
        resolved.engine.num_gpus = 4


def test_override_of_a_typed_home_leaves_the_pipeline_config_unchanged():
    resolved = _resolve({"model_path": WAN_T2V})
    materialized_steps = resolved.pipeline_config.dmd_denoising_steps

    overridden = resolved.with_override("checkpoint:test", {"pipeline.dmd_denoising_steps": [999, 500]})

    assert overridden.pipeline.dmd_denoising_steps == (999, 500)
    assert overridden.pipeline_config is resolved.pipeline_config
    assert overridden.pipeline_config.dmd_denoising_steps == materialized_steps
    assert overridden.provenance("pipeline.dmd_denoising_steps").source == "checkpoint:test"


def test_unified_memory_policy_is_decided_during_resolution(monkeypatch):
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)

    resolved = _resolve_with_device_policy({"model_path": WAN_T2V, "engine": {"offload": {"dit_layerwise": False}}})

    device_decisions = [(source, values) for source, values in resolved.decisions
                        if source in ("apply_unified_memory_offload_policy", "fill_lazy_module_load")]
    assert device_decisions == [
        ("apply_unified_memory_offload_policy", {
            "engine.offload.dit": False,
            "engine.offload.text_encoder": False,
            "engine.offload.image_encoder": False,
            "engine.offload.vae": False,
        }),
        ("fill_lazy_module_load", {
            "engine.offload.lazy_module_load": True
        }),
    ]
    assert resolved.engine.offload.lazy_module_load is True and resolved.engine.offload.dit is False
    assert resolved.override_log == ()
    assert resolved.provenance("engine.offload.dit").raw_value is True


def test_discrete_device_leaves_offload_requests_and_turns_lazy_module_load_off(monkeypatch):
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)

    resolved = _resolve_with_device_policy({"model_path": WAN_T2V, "engine": {"offload": {"dit_layerwise": False}}})

    assert resolved.engine.offload.dit is True and resolved.engine.offload.text_encoder is True
    assert resolved.engine.offload.lazy_module_load is False
    assert resolved.provenance("engine.offload.lazy_module_load").source == "fill_lazy_module_load"


def test_layerwise_offload_turns_off_conflicting_modes(monkeypatch):
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)

    resolved = _resolve_with_device_policy({"model_path": WAN_T2V, "engine": {"use_fsdp_inference": True}})

    assert [values for source, values in resolved.decisions if source == "apply_layerwise_offload_conflicts"] == [{
        "engine.use_fsdp_inference": False,
        "engine.offload.dit": False,
    }]
    assert resolved.engine.use_fsdp_inference is False and resolved.engine.offload.dit is False


def test_direct_pipeline_keeps_the_resolved_config(monkeypatch):
    import fastvideo.pipelines.composed_pipeline_base as composed_pipeline_base
    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

    class _Pipeline(ComposedPipelineBase):

        def load_modules(self, resolved_config, loaded_modules=None):
            self.loaded_with = resolved_config
            return {}

        def create_pipeline_stages(self, resolved_config):
            pass

    profiler = SimpleNamespace(region=lambda name: nullcontext())
    monkeypatch.setattr(composed_pipeline_base, "maybe_init_distributed_environment_and_model_parallel",
                        lambda *args: None)
    monkeypatch.setattr(composed_pipeline_base, "get_local_torch_device", lambda: torch.device("cuda:2"))
    monkeypatch.setattr(composed_pipeline_base, "get_world_group", lambda: SimpleNamespace(local_rank=2))
    monkeypatch.setattr(composed_pipeline_base, "get_or_create_profiler", lambda trace_dir: profiler)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    resolved = _resolve_with_device_policy({"model_path": WAN_T2V})

    pipeline = _Pipeline("unused", resolved, required_config_modules=[])

    assert pipeline.resolved_config is resolved and pipeline.loaded_with is resolved
    assert resolved.engine.offload.lazy_module_load is True


def test_pipeline_records_component_paths_and_shares_its_component_state(monkeypatch, tmp_path):
    import fastvideo.pipelines.composed_pipeline_base as composed_pipeline_base
    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
    from fastvideo.pipelines.stages import PipelineStage

    class _Pipeline(ComposedPipelineBase):
        _required_config_modules = ["transformer", "vae"]

        def create_pipeline_stages(self, resolved_config):
            pass

    class _Stage(PipelineStage):

        def forward(self, batch, resolved_config):
            return batch

    profiler = SimpleNamespace(region=lambda name: nullcontext())
    monkeypatch.setattr(composed_pipeline_base, "maybe_init_distributed_environment_and_model_parallel",
                        lambda *args: None)
    monkeypatch.setattr(composed_pipeline_base, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(composed_pipeline_base, "get_world_group", lambda: SimpleNamespace(local_rank=0))
    monkeypatch.setattr(composed_pipeline_base, "get_or_create_profiler", lambda trace_dir: profiler)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    monkeypatch.setattr(
        _Pipeline, "_load_config", lambda self, model_path: {
            "_class_name": "Probe",
            "_diffusers_version": "0",
            "transformer": ["diffusers", "Transformer"],
            "vae": ["diffusers", "VAE"],
        })
    monkeypatch.setattr(composed_pipeline_base.PipelineComponentLoader, "load_module",
                        staticmethod(lambda **kwargs: object()))
    resolved = _resolve({"model_path": WAN_T2V})

    pipeline = _Pipeline(str(tmp_path), resolved)
    stage = _Stage()
    pipeline.add_stage("probe_stage", stage)

    assert pipeline.component_state.model_paths == {
        "transformer": str(tmp_path / "transformer"),
        "vae": str(tmp_path / "vae"),
    }
    assert stage.component_state is pipeline.component_state


def test_training_resolution_keeps_the_dit_on_the_device():
    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

    built = {}

    class _Pipeline(ComposedPipelineBase):

        def __init__(self, model_path, resolved_config, required_config_modules=None, loaded_modules=None):
            built["resolved_config"] = resolved_config

        def post_init(self):
            pass

        def create_pipeline_stages(self, resolved_config):
            pass

    with isolated_environment():
        resolved = resolve_training_config({
            "model_path": WAN_T2V,
            "engine": {
                "parallelism": {
                    "sp_size": 1,
                    "hsdp_shard_dim": 1
                },
                "precision": {
                    "dit": "fp32"
                },
                "offload": {
                    "dit": True
                },
            },
        })

    _Pipeline.from_pretrained(WAN_T2V, resolved_config=resolved)

    assert resolved.engine.offload.dit is False
    assert resolved.provenance("engine.offload.dit").source == "keep_training_dit_on_device"
    assert built["resolved_config"] is resolved


def test_ltx2_checkpoint_refine_defaults_are_resolved_from_a_local_checkpoint(tmp_path):
    (tmp_path / "transformer").mkdir()
    (tmp_path / "model_index.json").write_text(
        json.dumps({
            "_class_name": "LTX2Pipeline",
            "_diffusers_version": "0",
            "transformer": ["diffusers", "Model"],
            "fastvideo_refine_lora_path": "FastVideo/LTX2-Distilled-LoRA",
            "fastvideo_refine_num_inference_steps": 2,
        }))

    from fastvideo.pipelines.basic.ltx2.pipeline_configs import LTX2T2VConfig

    resolved = _resolve({
        "model_path": str(tmp_path),
        "pipeline": {
            "experimental": {
                "pipeline_config": LTX2T2VConfig()
            }
        }
    })

    refine_decisions = [values for source, values in resolved.decisions if source == "fill_ltx2_refine_from_checkpoint"]
    assert refine_decisions == [{
        "pipeline.ltx2.refine.lora_path": "FastVideo/LTX2-Distilled-LoRA",
        "pipeline.ltx2.refine.num_inference_steps": 2,
    }]
    refine = resolved.pipeline.ltx2.refine
    assert (refine.lora_path, refine.num_inference_steps) == ("FastVideo/LTX2-Distilled-LoRA", 2)
    assert (refine.enabled, refine.add_noise, refine.guidance_scale) == (False, True, 1.0)
    assert resolved.provenance("pipeline.ltx2.refine.enabled").source == "fill_runtime_defaults"


def test_minimax_h3_checkpoint_schedule_is_resolved_and_reaches_the_first_forward(tmp_path, monkeypatch):
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3ModularPipeline
    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

    steps = [999, 874, 749, 624, 500, 375, 250, 125]
    (tmp_path / "fastvideo_inference.json").write_text(
        json.dumps({
            "schema_version": "fasth3-inference-contract-v1",
            "dmd_denoising_steps": steps,
            "num_inference_steps": 9,
            "transformer_forwards": 8,
        }))
    (tmp_path / "transformer").mkdir()
    (tmp_path / "model_index.json").write_text(
        json.dumps({
            "_class_name": "MiniMaxH3ModularPipeline",
            "_diffusers_version": "0",
            "transformer": ["diffusers", "Model"],
        }))
    resolved = _resolve({
        "model_path": str(tmp_path),
        "pipeline": {
            "experimental": {
                "pipeline_config": MiniMaxH3PipelineConfig()
            }
        }
    })
    assert resolved.pipeline.dmd_denoising_steps == tuple(steps)
    assert resolved.provenance("pipeline.dmd_denoising_steps").source == "fill_dmd_schedule_from_checkpoint"

    pipeline = object.__new__(MiniMaxH3ModularPipeline)
    pipeline.model_path = str(tmp_path)
    pipeline.resolved_config = resolved
    pipeline.modules = {"scheduler": MiniMaxH3Scheduler(shift=10.0), "audio_scheduler": MiniMaxH3Scheduler(shift=3.0)}
    pipeline.post_init_called = False
    seen = []
    monkeypatch.setattr(ComposedPipelineBase, "post_init",
                        lambda self: (setattr(self, "post_init_called", True), self._validate_checkpoint_schedule(
                            self.resolved_config)))
    monkeypatch.setattr(MiniMaxH3ModularPipeline, "_defer_denoise_modules", lambda self, args: False)
    monkeypatch.setattr(ComposedPipelineBase, "_stages", [], raising=False)
    monkeypatch.setattr(ComposedPipelineBase, "_stage_name_mapping", {}, raising=False)
    monkeypatch.setattr(ComposedPipelineBase, "stages",
                        property(lambda self: [lambda batch, args: seen.append(args) or batch]))

    pipeline.forward(SimpleNamespace(), resolved)

    assert seen == [resolved] and pipeline.resolved_config is resolved


def test_training_root_uses_command_line_defaults_and_typed_values(tmp_path):
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps({
            "model_path": WAN_T2V,
            "engine": {
                "parallelism": {
                    "sp_size": 1,
                    "hsdp_shard_dim": 1
                },
                "precision": {
                    "dit": "fp32"
                }
            },
            "training": {
                "optimizer": {
                    "learning_rate": 1e-5
                },
                "validation": {
                    "sampling_steps": [8, 50],
                    "guidance_scale": 6.0
                },
                "lora": {
                    "rank": 32
                },
            },
        }))

    with isolated_environment():
        resolved = load_resolved_run_config(TrainingRunConfig,
                                            ["--config", str(config_path), "--training.optimizer.lr_warmup_steps", "5"],
                                            mode=ExecutionMode.DISTILLATION)

    assert type(resolved.to_config()) is TrainingRunConfig
    assert resolved.mode is ExecutionMode.DISTILLATION and resolved.training_mode and not resolved.inference_mode
    offload = resolved.engine.offload
    training = resolved.training
    assert (offload.dit, offload.dit_layerwise, offload.pin_cpu_memory) == (False, False, False)
    assert (training.data.seed, training.ema.decay, training.optimizer.lr_warmup_steps,
            training.optimizer.learning_rate) == (42, 0.999, 5, 1e-5)
    assert training.optimizer.betas == (0.9, 0.999) and training.validation.sampling_steps == (8, 50)
    assert training.validation.guidance_scale == 6.0
    assert training.lora.alpha == 32 and resolved.provenance("training.lora.alpha").source == (
        "derive_lora_alpha_from_rank")
    assert resolved.model_path == WAN_T2V and resolved.pipeline.components.transformer_weights is None


def test_training_root_requires_parallel_sizes():
    with isolated_environment(), pytest.raises(ValueError, match="sp_size must be set for training"):
        resolve_training_config({"model_path": WAN_T2V, "engine": {"parallelism": {"hsdp_shard_dim": 1}}})


def test_preprocess_root_fills_model_path_loads_the_encoder_and_validates():
    with isolated_environment():
        resolved = resolve_preprocess_config({
            "model_path": WAN_T2V,
            "preprocess": {
                "dataset_path": "/data",
                "dataset_type": "merged"
            }
        })
        with pytest.raises(ValueError, match="dataset_path must be set"):
            resolve_preprocess_config({"model_path": WAN_T2V})

    assert type(resolved.to_config()) is PreprocessRunConfig
    assert resolved.mode is ExecutionMode.PREPROCESS and resolved.inference_mode
    assert resolved.preprocess.model_path == WAN_T2V
    assert resolved.preprocess.dataset_type.value == "merged"
    assert resolved.pipeline_config.vae_config.load_encoder is True


def test_parser_parses_enum_values_and_rejects_unknown_ones():
    config = parse_config(GeneratorConfig, {"model_path": WAN_T2V, "mode": "preprocess",
                                            "pipeline": {"workload_type": "i2v"}})

    assert config.mode is ExecutionMode.PREPROCESS and config.pipeline.workload_type is WorkloadType.I2V
    assert parse_config(GeneratorConfig, {"model_path": WAN_T2V, "mode": ExecutionMode.FINETUNING}).mode is (
        ExecutionMode.FINETUNING)
    with pytest.raises(ConfigValidationError, match="pipeline.workload_type"):
        parse_config(GeneratorConfig, {"model_path": WAN_T2V, "pipeline": {"workload_type": "x2y"}})


def test_decode_strategy_validation_matches_the_parallel_vae_module():
    from fastvideo.api.inference_resolution import VAE_PARALLEL_DECODE_STRATEGIES
    from fastvideo.models.vaes.minimax_h3_parallel import DECODE_GATHER_STRATEGIES

    assert VAE_PARALLEL_DECODE_STRATEGIES == tuple(DECODE_GATHER_STRATEGIES)
    with pytest.raises(ValueError, match="vae_parallel_decode_strategy"):
        _resolve({"model_path": WAN_T2V}, {"FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY": "scatter"})
