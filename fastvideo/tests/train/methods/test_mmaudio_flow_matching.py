# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MMAudio flow-matching training adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fastvideo.train.methods.fine_tuning.flow_matching import (
    FlowMatchingFineTuneMethod,
)
from fastvideo.train.models.mmaudio import MMAudioModel
from fastvideo.train.utils.config import RunConfig
from fastvideo.train.utils.training_config import (
    DataConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingLoopConfig,
)


class _TinyMMAudioTransformer(nn.Module):
    latent_seq_len = 4
    latent_dim = 3
    clip_seq_len = 6
    sync_seq_len = 9

    def __init__(self, *, v2: bool = False) -> None:
        super().__init__()
        self.v2 = v2
        self.unshard_calls = 0
        arch = SimpleNamespace(
            clip_dim=8,
            sync_dim=10,
            text_seq_len=5,
            text_dim=7,
        )
        self.config = SimpleNamespace(arch_config=arch)
        self.proj = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.latent_mean = nn.Parameter(
            torch.tensor([[[0.5, -0.25, 1.0]]]),
            requires_grad=False,
        )
        self.latent_std = nn.Parameter(
            torch.tensor([[[2.0, 0.5, 4.0]]]),
            requires_grad=False,
        )
        self.empty_string_feat = nn.Parameter(
            torch.full((5, 7), 13.0),
            requires_grad=False,
        )
        self.empty_clip_feat = nn.Parameter(torch.full((1, 8), 11.0))
        self.empty_sync_feat = nn.Parameter(torch.full((1, 10), 12.0))

    def unshard(self) -> None:
        self.unshard_calls += 1

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return value.sub(self.latent_mean).div(self.latent_std)

    def get_empty_clip_sequence(self, batch_size: int) -> torch.Tensor:
        return self.empty_clip_feat.unsqueeze(0).expand(
            batch_size, self.clip_seq_len, -1
        )

    def get_empty_sync_sequence(self, batch_size: int) -> torch.Tensor:
        return self.empty_sync_feat.unsqueeze(0).expand(
            batch_size, self.sync_seq_len, -1
        )

    def get_empty_string_sequence(self, batch_size: int) -> torch.Tensor:
        return self.empty_string_feat.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: dict[str, torch.Tensor],
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del timestep
        condition_dependency = (
            encoder_hidden_states["clip_features"].sum()
            + encoder_hidden_states["sync_features"].sum()
        ) * 0.0
        return self.proj(hidden_states) + condition_dependency


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        data=DataConfig(
            data_path="",
            preprocessed_data_type="mmaudio_features",
            train_batch_size=2,
            training_cfg_rate=0.4,
            seed=123,
        ),
        optimizer=OptimizerConfig(
            learning_rate=1e-4,
            betas=(0.9, 0.95),
            eps=1e-6,
        ),
        loop=TrainingLoopConfig(max_train_steps=2),
    )


def _raw_batch() -> dict[str, torch.Tensor]:
    return {
        "audio_latent_mean": (
            torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 10
        ),
        "audio_latent_std": torch.full((2, 4, 3), 0.2),
        "clip_features": torch.zeros(2, 6, 8),
        "sync_features": torch.zeros(2, 9, 10),
        "text_features": torch.zeros(2, 5, 7),
        "video_exists": torch.tensor([False, True]),
        "text_exists": torch.tensor([True, False]),
    }


def test_prepare_batch_matches_official_rng_and_flow_contract() -> None:
    config = _training_config()
    transformer = _TinyMMAudioTransformer()
    model = MMAudioModel(
        init_from="unused",
        training_config=config,
        transformer=transformer,
    )
    raw = _raw_batch()
    actual_generator = torch.Generator().manual_seed(7)
    expected_generator = torch.Generator().manual_seed(7)

    batch = model.prepare_batch(raw, generator=actual_generator)

    posterior_noise = torch.empty_like(raw["audio_latent_mean"]).normal_(
        generator=expected_generator
    )
    clean = raw["audio_latent_mean"] + raw["audio_latent_std"] * posterior_noise
    clean = (clean - transformer.latent_mean) / transformer.latent_std
    timestep = torch.sigmoid(
        torch.randn(2, generator=expected_generator)
        * config.model.logit_std
        + config.model.logit_mean
    )
    prior = torch.empty_like(clean).normal_(generator=expected_generator)
    noisy = (1 - timestep[:, None, None]) * prior + timestep[:, None, None] * clean
    null_video = torch.rand(2, generator=expected_generator) < 0.4
    null_text = torch.rand(2, generator=expected_generator) < 0.4

    torch.testing.assert_close(batch.latents, clean)
    torch.testing.assert_close(batch.timesteps, timestep)
    torch.testing.assert_close(batch.noise, prior)
    torch.testing.assert_close(batch.noisy_model_input, noisy)
    torch.testing.assert_close(batch.training_target, clean - prior)
    assert transformer.unshard_calls == 1

    assert batch.conditional_dict is not None
    expected_video_null = torch.tensor([True, False]) | null_video
    expected_text_null = torch.tensor([False, True]) | null_text
    clip = batch.conditional_dict["clip_features"]
    sync = batch.conditional_dict["sync_features"]
    text = batch.conditional_dict["text_features"]
    torch.testing.assert_close(
        clip[expected_video_null],
        torch.full_like(clip[expected_video_null], 11.0),
    )
    torch.testing.assert_close(
        sync[expected_video_null],
        torch.full_like(sync[expected_video_null], 12.0),
    )
    torch.testing.assert_close(
        text[expected_text_null],
        torch.full_like(text[expected_text_null], 13.0),
    )

    assert transformer.latent_mean.requires_grad is False
    assert transformer.latent_std.requires_grad is False
    assert transformer.empty_string_feat.requires_grad is False
    assert transformer.empty_clip_feat.requires_grad is True
    assert transformer.empty_sync_feat.requires_grad is True


def test_flow_matching_method_runs_loss_and_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    monkeypatch.setattr(
        "fastvideo.dataset.mmaudio_feature_dataset."
        "build_mmaudio_feature_dataloader",
        lambda *args, **kwargs: None,
    )
    model = MMAudioModel(
        init_from="unused",
        training_config=config,
        transformer=_TinyMMAudioTransformer(),
    )
    cfg = RunConfig(
        models={},
        method={
            "_target_": (
                "fastvideo.train.methods.fine_tuning.flow_matching."
                "FlowMatchingFineTuneMethod"
            )
        },
        training=config,
        callbacks={},
        raw={},
    )
    method = FlowMatchingFineTuneMethod(
        cfg=cfg,
        role_models={"student": model},
    )
    method.cuda_generator = torch.Generator().manual_seed(17)

    eager_parameters = model.eager_optimizer_state_parameters()
    assert len(eager_parameters) == 2
    for parameter in eager_parameters:
        assert set(method._student_optimizer.state[parameter]) == {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }

    loss_map, outputs, metrics = method.single_train_step(_raw_batch(), 0)

    assert torch.isfinite(loss_map["total_loss"])
    assert loss_map["flow_matching_loss"] is loss_map["total_loss"]
    assert metrics == {}
    method.backward(loss_map, outputs)
    assert model.transformer.proj.weight.grad is not None
    assert torch.count_nonzero(model.transformer.proj.weight.grad) > 0
    assert model.transformer.empty_clip_feat.grad is not None
    assert model.transformer.empty_sync_feat.grad is not None


def test_v2_training_is_rejected_by_default() -> None:
    with pytest.raises(ValueError, match="does not support `_v2`"):
        MMAudioModel(
            init_from="unused",
            training_config=_training_config(),
            transformer=_TinyMMAudioTransformer(v2=True),
        )
