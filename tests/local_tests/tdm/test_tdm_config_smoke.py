# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the shipped Wan TDM example config."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from fastvideo.models.wan.pipeline_config import (
    FastWan2_1_T2V_480P_Config,
    FastWan2_2_TI2V_5B_Config,
)
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from fastvideo.pipelines.basic.wan.stages.dmd import DmdDenoisingStage
from fastvideo.train.methods.distribution_matching.tdm import (
    TDMMethod,
    flow_transition_to_noisier_sigma,
)
from fastvideo.train.models.wan import WanModel
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.instantiate import resolve_target


class _DenoisingStageTransformerStub(torch.nn.Module):

    hidden_size = 128
    num_attention_heads = 8


def test_tdm_wan_lora_config_resolves_without_loading_weights() -> None:
    config_path = Path("examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml")

    cfg = load_run_config(str(config_path))

    assert resolve_target(str(cfg.method["_target_"])) is TDMMethod
    assert set(cfg.models) == {"student", "teacher", "critic"}
    assert cfg.models["student"]["trainable"] is True
    assert cfg.models["teacher"]["trainable"] is False
    assert cfg.models["critic"]["trainable"] is True
    assert cfg.models["student"]["lora"]["enable"] is True
    assert cfg.models["critic"]["lora"]["enable"] is True

    assert cfg.method["rollout_mode"] == "simulate"
    assert cfg.method["tdm_denoising_steps"] == [1000, 750, 500, 250]
    assert cfg.method["warmup_steps"] == 200
    assert cfg.method["noise_interval_mode"] == "separate"
    assert cfg.method["student_sample_type"] == "ode"
    assert cfg.method["use_randmid"] is False
    assert cfg.method["max_grad_norm"] == 1.0
    assert cfg.method["cfg_uncond"] == {
        "text": "negative_prompt",
        "on_missing": "error",
    }
    assert "enable_gradient_in_rollout" not in cfg.method
    assert "grad_clip" not in cfg.callbacks
    assert cfg.training.pipeline_config is not None
    assert getattr(cfg.training.pipeline_config, "flow_shift") == 8
    assert getattr(cfg.training.pipeline_config, "dmd_sample_type") == "ode"


def test_tdm_wan_config_flow_shift_reaches_role_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path("examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml")
    cfg = load_run_config(str(config_path))

    def fake_load_transformer(self: WanModel, **kwargs: object) -> torch.nn.Module:
        del self, kwargs
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(WanModel, "_load_transformer", fake_load_transformer)

    model = WanModel(
        init_from=str(cfg.models["student"]["init_from"]),
        training_config=cfg.training,
        trainable=False,
    )

    assert model.noise_scheduler.shift == 8.0
    assert model.timestep_shift == 8.0


def test_dmd_denoising_stage_preserves_configured_scheduler_shift() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    stage = DmdDenoisingStage(
        transformer=_DenoisingStageTransformerStub(),
        scheduler=scheduler,
    )

    assert stage.scheduler is scheduler
    assert stage.scheduler.shift == 5.0


@pytest.mark.parametrize(
    "config_cls",
    [FastWan2_1_T2V_480P_Config, FastWan2_2_TI2V_5B_Config],
)
def test_dmd_denoising_stage_preserves_legacy_scheduler_space_steps(config_cls: type) -> None:
    config = config_cls()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=config.flow_shift)
    stage = DmdDenoisingStage(
        transformer=_DenoisingStageTransformerStub(),
        scheduler=scheduler,
    )

    timesteps = stage._set_dmd_timesteps(
        torch.tensor(config.dmd_denoising_steps),
        device=torch.device("cpu"),
        scheduler_space=config.dmd_denoising_steps_are_scheduler_space,
    )

    assert config.dmd_denoising_steps_are_scheduler_space is True
    assert_close(timesteps, torch.tensor([1000.0, 757.0, 522.0]))
    assert_close(scheduler.sigmas, torch.tensor([1.0, 0.757, 0.522, 0.0]))


def test_tdm_flow_bridge_smoke() -> None:
    clean = torch.ones(1, 1, 1, 1, 1)
    eps_from = torch.zeros_like(clean)
    proposal = torch.ones_like(clean)
    sigma_from = torch.tensor([0.25])
    sigma_to = torch.tensor([0.5])
    noisy_from = (1.0 - sigma_from) * clean + sigma_from * eps_from

    noisy_to, mixed_noise, _ = flow_transition_to_noisier_sigma(
        noisy_from=noisy_from,
        clean_latents=clean,
        eps_from=eps_from,
        sigma_from=sigma_from,
        sigma_to=sigma_to,
        proposal_noise=proposal,
    )

    assert_close(noisy_to, (1.0 - sigma_to) * clean + sigma_to * mixed_noise)
