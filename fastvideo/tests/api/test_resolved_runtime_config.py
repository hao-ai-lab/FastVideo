# SPDX-License-Identifier: Apache-2.0
"""The resolved runtime config: materialization, flat names, runtime state, device policy, overrides, and roots."""
from __future__ import annotations

import dataclasses
import json
import pickle
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from fastvideo.api import device_policy
from fastvideo.api.compat import generator_config_to_fastvideo_args
from fastvideo.api.errors import ConfigValidationError
from fastvideo.api.inference_resolution import resolve_inference_config
from fastvideo.api.parser import parse_config
from fastvideo.api.schema import ExecutionMode, GeneratorConfig, WorkloadType
from fastvideo.api.training_schema import (PreprocessRunConfig, TrainingRunConfig, load_resolved_run_config,
                                           resolve_preprocess_config, resolve_training_config)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.tests.api.config_snapshot import isolated_environment, to_jsonable

WAN_T2V = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
LTX2 = "FastVideo/LTX2-Distilled-Diffusers"
FASTH3 = "FastVideo/FastVideo-FastH3-8-Step-V2"


def _resolve(raw, env_values=None):
    with isolated_environment(env_values):
        return resolve_inference_config(raw)


def test_resolution_builds_and_freezes_the_pipeline_config_once(monkeypatch):
    from fastvideo.configs.pipelines.base import PipelineConfig

    builds = []
    original = PipelineConfig.from_kwargs.__func__
    monkeypatch.setattr(PipelineConfig, "from_kwargs",
                        classmethod(lambda cls, kwargs: builds.append(dict(kwargs)) or original(cls, kwargs)))
    resolved = _resolve({"model_path": WAN_T2V, "pipeline": {"flow_shift": 5.0, "dit": {"prefix": "Probe"}}})

    assert len(builds) == 1
    assert type(resolved.pipeline_config).__name__ == "WanT2V480PConfig"
    assert resolved.pipeline_config.flow_shift == resolved.pipeline.flow_shift == 5.0
    assert resolved.pipeline_config.dit_config.prefix == "Probe"
    with pytest.raises(AttributeError, match="read-only"):
        resolved.pipeline_config.flow_shift = 1.0


@pytest.mark.parametrize("raw", [
    {
        "model_path": WAN_T2V,
        "engine": {
            "num_gpus": 2
        },
        "pipeline": {
            "experimental": {
                "master_port": 29600,
                "prompt_txt": "prompts.txt",
                "refine_enabled": True
            }
        },
    },
    {
        "model_path": LTX2,
        "pipeline": {
            "preset_overrides": {
                "refine": {
                    "enabled": True,
                    "num_inference_steps": 2,
                    "add_noise": False
                }
            },
            "ltx2": {
                "vae_spatial_tile_size_in_pixels": 512
            },
        },
    },
])
def test_flat_names_return_the_values_that_fastvideo_args_held(raw):
    resolved = _resolve(raw)
    with isolated_environment():
        resolved_config = generator_config_to_fastvideo_args(resolved)

    # disable_autocast and boundary_ratio held their FastVideoArgs defaults; the typed field holds the input.
    skipped = {"pipeline_config", "disable_autocast", "boundary_ratio"}
    for config_field in dataclasses.fields(FastVideoArgs):
        if config_field.name not in skipped:
            expected = to_jsonable(getattr(resolved_config, config_field.name))
            assert to_jsonable(getattr(resolved, config_field.name)) == expected, config_field.name
    assert to_jsonable(resolved.pipeline_config) == to_jsonable(resolved_config.pipeline_config)
    assert resolved.workload_type is WorkloadType.T2V
    assert resolved.mode is ExecutionMode.INFERENCE and resolved.inference_mode and not resolved.training_mode


def test_unknown_and_training_only_names_raise_attribute_error():
    resolved = _resolve({"model_path": WAN_T2V})

    assert getattr(resolved, "log_level_progress", "default") == "default"
    assert not hasattr(resolved, "_loading_teacher_critic_model")
    with pytest.raises(AttributeError, match="exists on TrainingRunConfig"):
        resolved.learning_rate
    assert resolved.preprocess_config is None


def test_runtime_state_is_shared_by_overrides_and_survives_pickling():
    resolved = _resolve({"model_path": WAN_T2V, "pipeline": {"experimental": {"ray_runtime_env": {"pip": ["x"]}}}})

    resolved.model_loaded["vae"] = False
    resolved.model_paths["transformer"] = "/weights/transformer"
    resolved.ray_placement_group = "placement-group"
    overridden = resolved.with_override("test:source", {"engine.offload.vae": False})
    copy = pickle.loads(pickle.dumps(overridden))

    assert overridden.model_loaded is resolved.model_loaded
    assert copy.model_loaded == {"transformer": True, "vae": False, "upsampler": True}
    assert copy.model_paths == {"transformer": "/weights/transformer"}
    assert copy.ray_placement_group == "placement-group"
    assert copy.ray_runtime_env == {"pip": ["x"]}
    assert copy.pipeline_config._frozen and copy.vae_cpu_offload is False and resolved.vae_cpu_offload is True
    with pytest.raises(AttributeError, match="read-only"):
        resolved.num_gpus = 4


def test_override_of_a_typed_home_updates_its_pipeline_config_mirror():
    resolved = _resolve({"model_path": WAN_T2V})

    overridden = resolved.with_override("checkpoint:test", {"pipeline.dmd_denoising_steps": [999, 500]})

    assert overridden.pipeline.dmd_denoising_steps == (999, 500)
    assert overridden.pipeline_config is resolved.pipeline_config
    assert overridden.pipeline_config.dmd_denoising_steps == [999, 500]
    assert overridden.provenance("pipeline.dmd_denoising_steps").source == "checkpoint:test"


def test_unified_memory_policy_returns_recorded_overrides_once(monkeypatch):
    resolved = _resolve({"model_path": WAN_T2V, "engine": {"offload": {"dit_layerwise": False}}})
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: device_id == 1)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)

    decided = device_policy.finalize_device_offload_policy(resolved, 1)
    again = device_policy.finalize_device_offload_policy(decided, 1)

    assert [source for source, _ in decided.override_log] == [
        "device_policy:unified_memory",
        "device_policy:lazy_module_load",
    ]
    assert decided.override_log[0][1] == {
        "engine.offload.dit": False,
        "engine.offload.text_encoder": False,
        "engine.offload.image_encoder": False,
        "engine.offload.vae": False,
    }
    assert decided.lazy_module_load is True and decided.dit_cpu_offload is False
    assert again.override_log == decided.override_log
    assert resolved.override_log == () and resolved.dit_cpu_offload is True
    assert device_policy.offload_disabled_on_unified_memory(1, "text_encoder_cpu_offload")
    assert not device_policy.offload_disabled_on_unified_memory(0, "text_encoder_cpu_offload")
    assert device_policy.finalize_device_offload_policy(resolved, 0).lazy_module_load is False


def test_layerwise_offload_turns_off_conflicting_modes():
    resolved = _resolve({"model_path": WAN_T2V, "engine": {"use_fsdp_inference": True}})

    decided = device_policy.resolve_device_offload_conflicts(resolved)

    assert [values for _, values in decided.override_log] == [{
        "engine.use_fsdp_inference": False
    }, {
        "engine.offload.dit": False
    }]


def test_direct_pipeline_keeps_the_policy_result_of_a_resolved_config(monkeypatch):
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
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    resolved = _resolve({"model_path": WAN_T2V})

    pipeline = _Pipeline("unused", resolved, required_config_modules=[])

    assert pipeline.resolved_config is pipeline.loaded_with
    assert pipeline.resolved_config.lazy_module_load is True and resolved.lazy_module_load is None


def test_ltx2_checkpoint_refine_defaults_rebind_the_pipeline_config(tmp_path):
    from fastvideo.pipelines.basic.ltx2.ltx2_pipeline import LTX2Pipeline

    resolved = _resolve({"model_path": LTX2})
    pipeline = object.__new__(LTX2Pipeline)
    pipeline.model_path = str(tmp_path)
    pipeline.resolved_config = resolved
    pipeline._required_config_modules = list(LTX2Pipeline._required_config_modules)
    model_index = {name: ["diffusers", "Model"] for name in pipeline._required_config_modules}
    pipeline._load_config = lambda model_path: {
        "_class_name": "LTX2Pipeline",
        "_diffusers_version": "0",
        "fastvideo_refine_lora_path": "FastVideo/LTX2-Distilled-LoRA",
        "fastvideo_refine_num_inference_steps": 2,
        **model_index,
    }

    pipeline.load_modules(resolved, {name: object() for name in model_index})

    rebound = pipeline.resolved_config
    assert rebound is not resolved
    assert rebound.override_log == (("checkpoint:model_index.json", {
        "pipeline.ltx2.refine.lora_path": "FastVideo/LTX2-Distilled-LoRA",
        "pipeline.ltx2.refine.num_inference_steps": 2,
    }), )
    assert rebound.ltx2_refine_lora_path == "FastVideo/LTX2-Distilled-LoRA"
    assert rebound.ltx2_refine_num_inference_steps == 2 and resolved.ltx2_refine_num_inference_steps == 3


def test_minimax_h3_checkpoint_schedule_rebinds_and_reaches_the_first_forward(tmp_path, monkeypatch):
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
    resolved = _resolve({"model_path": FASTH3})
    pipeline = object.__new__(MiniMaxH3ModularPipeline)
    pipeline.model_path = str(tmp_path)
    pipeline.resolved_config = resolved
    pipeline.modules = {"scheduler": MiniMaxH3Scheduler(shift=10.0), "audio_scheduler": MiniMaxH3Scheduler(shift=3.0)}
    pipeline.post_init_called = False
    seen = []
    monkeypatch.setattr(ComposedPipelineBase, "post_init",
                        lambda self: (setattr(self, "post_init_called", True), self._load_checkpoint_schedule(
                            self.resolved_config)))
    monkeypatch.setattr(MiniMaxH3ModularPipeline, "_defer_denoise_modules", lambda self, args: False)
    monkeypatch.setattr(ComposedPipelineBase, "_stages", [], raising=False)
    monkeypatch.setattr(ComposedPipelineBase, "_stage_name_mapping", {}, raising=False)
    monkeypatch.setattr(ComposedPipelineBase, "stages",
                        property(lambda self: [lambda batch, args: seen.append(args) or batch]))

    pipeline.forward(SimpleNamespace(), pipeline.resolved_config)

    assert seen == [pipeline.resolved_config] and pipeline.resolved_config is not resolved
    assert pipeline.resolved_config.pipeline.dmd_denoising_steps == tuple(steps)
    assert pipeline.resolved_config.pipeline_config.dmd_denoising_steps == steps
    assert pipeline.resolved_config.provenance("pipeline.dmd_denoising_steps").source == (
        "checkpoint:fastvideo_inference.json")


def test_training_root_uses_command_line_defaults_and_flat_formats(tmp_path):
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
    assert (resolved.dit_cpu_offload, resolved.dit_layerwise_offload, resolved.pin_cpu_memory) == (False, False, False)
    assert (resolved.seed, resolved.ema_decay, resolved.lr_warmup_steps, resolved.learning_rate) == (42, 0.999, 5, 1e-5)
    assert resolved.betas == "0.9,0.999" and resolved.validation_sampling_steps == "8,50"
    assert resolved.validation_guidance_scale == "6.0"
    assert resolved.lora_alpha == 32 and resolved.provenance("training.lora.alpha").source == (
        "derive_lora_alpha_from_rank")
    assert resolved.pretrained_model_name_or_path == WAN_T2V and resolved.init_weights_from_safetensors is None
    with pytest.raises(AttributeError, match="no readers"):
        resolved.mixed_precision


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
    assert resolved.preprocess_config.model_path == WAN_T2V
    assert resolved.preprocess_config.dataset_type.value == "merged"
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
