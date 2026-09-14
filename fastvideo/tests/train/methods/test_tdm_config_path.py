# SPDX-License-Identifier: Apache-2.0
"""Package-level regressions for the Wan TDM training config path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch.testing import assert_close

import fastvideo.train.callbacks.validation as validation_module
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from fastvideo.pipelines.basic.wan.stages.dmd import DmdDenoisingStage
from fastvideo.train.callbacks.validation import ValidationCallback
from fastvideo.train.methods.distribution_matching.tdm import TDMMethod
from fastvideo.train.models.wan import WanModel
from fastvideo.train.utils.builder import build_from_config
from fastvideo.train.utils.config import load_run_config


_TDM_CONFIG = "examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml"
_TDM_OVERFIT_CONFIG = "examples/train/configs/distribution_matching/wan/tdm_t2v_lora_overfit.yaml"
_WAN_DMD_PIPELINE = "fastvideo.pipelines.basic.wan.wan_dmd_pipeline.WanDMDPipeline"


class _DenoisingStageTransformerStub(torch.nn.Module):

    hidden_size = 128
    num_attention_heads = 8


def _patch_lightweight_wan(monkeypatch) -> None:

    def fake_load_transformer(self: WanModel, **kwargs: Any) -> torch.nn.Module:
        del self, kwargs
        return torch.nn.Linear(1, 1)

    def fake_enable_lora(self: WanModel, transformer: torch.nn.Module) -> bool:
        del self, transformer
        return False

    def fake_init_preprocessors(self: WanModel, training_config: object) -> None:
        del training_config
        self.dataloader = ["fake-batch"]
        self.start_step = 0

    monkeypatch.setattr(WanModel, "_load_transformer", fake_load_transformer)
    monkeypatch.setattr(WanModel, "_enable_lora_if_configured", fake_enable_lora)
    monkeypatch.setattr(WanModel, "init_preprocessors", fake_init_preprocessors)


def test_wan_tdm_example_builds_method_with_pipeline_scheduler_shift(monkeypatch) -> None:
    _patch_lightweight_wan(monkeypatch)
    cfg = load_run_config(_TDM_CONFIG)

    training_config, method, dataloader, start_step = build_from_config(cfg)

    assert isinstance(method, TDMMethod)
    assert dataloader == ["fake-batch"]
    assert start_step == 0
    assert training_config.pipeline_config is not None
    assert training_config.pipeline_config.flow_shift == 8
    assert training_config.pipeline_config.dmd_sample_type == "ode"
    assert method.method_config["tdm_denoising_steps"] == [1000, 750, 500, 250]
    assert method._rollout_sample_type == "ode"
    assert method.student.noise_scheduler.shift == 8.0
    assert method.critic.noise_scheduler.shift == 8.0

    sigmas = method._timestep_to_sigma(torch.tensor([1000, 750, 500, 250]))
    expected = torch.tensor([
        1.0,
        8.0 * 0.75 / (1.0 + 7.0 * 0.75),
        8.0 * 0.5 / (1.0 + 7.0 * 0.5),
        8.0 * 0.25 / (1.0 + 7.0 * 0.25),
    ])
    assert_close(sigmas, expected)


def test_wan_tdm_overfit_example_uses_four_gpu_text_only_diagnostic() -> None:
    cfg = load_run_config(_TDM_OVERFIT_CONFIG)

    assert cfg.training.distributed.num_gpus == 4
    assert cfg.training.distributed.hsdp_shard_dim == 4
    assert cfg.training.data.preprocessed_data_type == "text_only"
    assert cfg.training.data.train_batch_size == 1
    assert cfg.training.loop.max_train_steps == 100
    assert cfg.training.checkpoint.training_state_checkpointing_steps == 25
    assert cfg.training.tracker.trackers == ["jsonl"]
    assert cfg.method["generator_update_interval"] == 1
    assert cfg.method["noise_interval_mode"] == "next_step"
    assert cfg.method["use_randmid"] is False
    assert cfg.method["cfg_uncond"] == {
        "text": "zero",
        "on_missing": "ignore",
    }
    assert cfg.callbacks["validation"]["every_steps"] == 0
    assert "ema" not in cfg.callbacks


def test_wan_tdm_training_and_validation_use_same_shifted_model_labels(monkeypatch) -> None:
    _patch_lightweight_wan(monkeypatch)
    cfg = load_run_config(_TDM_CONFIG)
    training_config, method, _, _ = build_from_config(cfg)
    assert isinstance(method, TDMMethod)
    assert training_config.pipeline_config is not None

    raw_steps = torch.tensor(method.method_config["tdm_denoising_steps"])
    training_sigmas = method._timestep_to_sigma(raw_steps)
    training_labels = method._model_timestep_for_sigma(training_sigmas, method.student)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=training_config.pipeline_config.flow_shift)
    stage = DmdDenoisingStage(
        transformer=_DenoisingStageTransformerStub(),
        scheduler=scheduler,
    )

    validation_labels = stage._set_dmd_timesteps(
        raw_steps,
        device=torch.device("cpu"),
        scheduler_space=training_config.pipeline_config.dmd_denoising_steps_are_scheduler_space,
    )

    assert training_config.pipeline_config.dmd_denoising_steps_are_scheduler_space is False
    assert_close(validation_labels, training_labels)
    assert_close(scheduler.sigmas[:-1], training_sigmas)
    assert validation_labels[1].item() > 900.0


class _FakeWanDMDPipeline:

    from_pretrained_kwargs: dict[str, Any] = {}
    last_batch: Any | None = None
    last_inference_args: Any | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        **kwargs: Any,
    ) -> "_FakeWanDMDPipeline":
        cls.from_pretrained_kwargs = {
            "model_path": model_path,
            **kwargs,
        }
        return cls()

    def get_module(
        self,
        name: str,
    ) -> None:
        del name
        return None

    def forward(
        self,
        batch: Any,
        inference_args: Any,
    ) -> SimpleNamespace:
        type(self).last_batch = batch
        type(self).last_inference_args = inference_args
        return SimpleNamespace(output=torch.zeros(1, 3, 1, 2, 2))


def test_wan_tdm_validation_propagates_sampling_timesteps_to_dmd_pipeline(monkeypatch) -> None:
    cfg = load_run_config(_TDM_CONFIG)
    validation_cfg = dict(cfg.callbacks["validation"])
    assert validation_cfg["pipeline_target"] == _WAN_DMD_PIPELINE
    assert validation_cfg["offload_training_state"] is True
    assert validation_cfg["unload_pipeline_after_validation"] is True

    def fake_resolve_target(target: str) -> type[_FakeWanDMDPipeline]:
        assert target == _WAN_DMD_PIPELINE
        return _FakeWanDMDPipeline

    monkeypatch.setattr(validation_module, "resolve_target", fake_resolve_target)
    monkeypatch.setattr(
        validation_module,
        "ValidationDataset",
        lambda filename: [{
            "caption": "tiny prompt",
            "prompt": "tiny prompt",
        }],
    )
    monkeypatch.setattr(validation_module, "DataLoader", lambda dataset, batch_size, num_workers: dataset)
    monkeypatch.setattr(SamplingParam, "from_pretrained", classmethod(lambda cls, model_path: cls()))

    callback = ValidationCallback(**validation_cfg)
    assert callback.offload_training_state is True
    assert callback.unload_pipeline_after_validation is True
    callback.training_config = cfg.training
    callback.rank_in_sp_group = 0
    callback.validation_random_generator = torch.Generator(device="cpu").manual_seed(0)

    result = callback._run_validation_for_steps(
        4,
        transformer=torch.nn.Linear(1, 1),
    )

    inference_args = _FakeWanDMDPipeline.last_inference_args
    batch = _FakeWanDMDPipeline.last_batch
    assert inference_args is not None
    assert batch is not None
    assert result.captions == ["tiny prompt"]
    assert batch.timesteps.tolist() == [1000, 750, 500, 250]
    assert inference_args.pipeline_config.dmd_denoising_steps == [1000, 750, 500, 250]
    assert inference_args.pipeline_config.dmd_sample_type == "ode"
    assert inference_args.pipeline_config.flow_shift == 8
    assert inference_args.pipeline_config.dmd_denoising_steps_are_scheduler_space is False
    assert _FakeWanDMDPipeline.from_pretrained_kwargs["flow_shift"] == 8.0
