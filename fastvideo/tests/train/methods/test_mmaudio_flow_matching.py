# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MMAudio flow-matching training adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fastvideo.configs.models.dits import MMAUDIO_VARIANT_ARCHITECTURES
from fastvideo.configs.pipelines.mmaudio import MMAudioSmall44kV2AConfig
from fastvideo.models.dits.mmaudio import MMAudioTransformer
from fastvideo.train.methods.fine_tuning.flow_matching import (
    FlowMatchingFineTuneMethod,
)
from fastvideo.train.models.mmaudio import MMAudioModel
from fastvideo.train.models.mmaudio.mmaudio import _load_or_compute_latent_stats
from fastvideo.train.utils.config import RunConfig
from fastvideo.train.utils.training_config import (
    DataConfig,
    DistributedConfig,
    ModelTrainingConfig,
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


def test_sequence_length_update_in_inference_mode_keeps_normal_buffers() -> None:
    owner = SimpleNamespace()
    inference_modes: list[bool] = []

    def initialize_rotations() -> None:
        inference_modes.append(torch.is_inference_mode_enabled())
        owner.latent_rot = torch.ones(1)
        owner.clip_rot = torch.ones(1)

    owner.initialize_rotations = initialize_rotations

    with torch.inference_mode():
        MMAudioTransformer.update_seq_lengths(owner, 4, 6, 9)

    assert owner._latent_seq_len == 4
    assert owner._clip_seq_len == 6
    assert owner._sync_seq_len == 9
    assert inference_modes == [False]
    assert not torch.is_inference(owner.latent_rot)
    assert not torch.is_inference(owner.clip_rot)


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


def test_flow_matching_method_compiles_inner_transformer_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    config.model = ModelTrainingConfig(
        compile_train_fn=True,
        torch_compile_kwargs={"backend": "eager"},
    )
    monkeypatch.setattr(
        "fastvideo.dataset.mmaudio_feature_dataset."
        "build_mmaudio_feature_dataloader",
        lambda *args, **kwargs: None,
    )
    compile_calls: list[tuple[object, dict[str, object]]] = []

    def _compile(fn, **kwargs):
        compile_calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", _compile)
    model = MMAudioModel(
        init_from="unused",
        training_config=config,
        transformer=_TinyMMAudioTransformer(),
    )
    config.distributed = DistributedConfig(strategy="ddp")
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
    method.cuda_generator = torch.Generator().manual_seed(19)

    loss_map, _, _ = method.single_train_step(_raw_batch(), 0)

    assert len(compile_calls) == 1
    assert compile_calls[0][1] == {"backend": "eager", "fullgraph": True}
    assert torch.isfinite(loss_map["total_loss"])


def test_compile_train_fn_rejects_fsdp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    config.model = ModelTrainingConfig(compile_train_fn=True)
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

    with pytest.raises(ValueError, match="strategy=ddp"):
        FlowMatchingFineTuneMethod(
            cfg=cfg,
            role_models={"student": model},
        )


def test_v2_training_is_rejected_by_default() -> None:
    with pytest.raises(ValueError, match="does not support `_v2`"):
        MMAudioModel(
            init_from="unused",
            training_config=_training_config(),
            transformer=_TinyMMAudioTransformer(v2=True),
        )


def test_official_mmaudio_variant_architectures() -> None:
    expected = {
        "small_16k": (20, 448, 12, 8, 7, 250, False),
        "small_44k": (40, 448, 12, 8, 7, 345, False),
        "medium_44k": (40, 896, 12, 8, 14, 345, False),
        "large_44k": (40, 896, 21, 14, 14, 345, False),
        "large_44k_v2": (40, 896, 21, 14, 14, 345, True),
    }
    actual = {
        name: (
            values["latent_dim"],
            values["hidden_dim"],
            values["depth"],
            values["fused_depth"],
            values["num_heads"],
            values["latent_seq_len"],
            values["v2"],
        )
        for name, values in MMAUDIO_VARIANT_ARCHITECTURES.items()
    }
    assert actual == expected


def test_from_scratch_builds_variant_without_loading_dit_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "fastvideo.train.models.mmaudio.mmaudio._load_or_compute_latent_stats",
        lambda *args, **kwargs: (
            torch.zeros(1, 1, 40),
            torch.ones(1, 1, 40),
        ),
    )
    monkeypatch.setattr(
        "fastvideo.train.models.mmaudio.mmaudio._load_empty_string_features",
        lambda *args, **kwargs: torch.zeros(77, 1024),
    )

    def _build(**kwargs):
        captured.update(kwargs)
        return _TinyMMAudioTransformer()

    monkeypatch.setattr(
        "fastvideo.train.models.mmaudio.mmaudio.build_fsdp_model_from_scratch",
        _build,
    )
    model = MMAudioModel(
        training_config=_training_config(),
        variant="small_44k",
        from_scratch=True,
        empty_string_features_path="unused.safetensors",
    )

    assert model._init_from == "scratch:small_44k"
    assert isinstance(model.training_config.pipeline_config, MMAudioSmall44kV2AConfig)
    assert captured["model_cls"].__name__ == "MMAudioTransformer"
    assert captured["init_params"]["hf_config"]["v2"] is False


def test_from_scratch_rejects_v2_variant_before_loading_assets() -> None:
    with pytest.raises(ValueError, match="does not train 'large_44k_v2'"):
        MMAudioModel(
            training_config=_training_config(),
            variant="large_44k_v2",
            from_scratch=True,
            empty_string_features_path="unused.safetensors",
        )


def test_44k_variants_share_cached_latent_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "variant": "small_44k",
        "latent_mean": torch.zeros(1, 1, 40),
        "latent_std": torch.ones(1, 1, 40),
    }

    class _ReceiverGroup:
        is_first_rank = False

        @staticmethod
        def broadcast_object(value, src=0):
            del value, src
            return payload

    monkeypatch.setattr(
        "fastvideo.train.models.mmaudio.mmaudio.get_world_group",
        lambda: _ReceiverGroup(),
    )
    mean, std = _load_or_compute_latent_stats(
        _training_config(),
        variant="medium_44k",
        cache_path=None,
        chunk_size=1,
    )

    torch.testing.assert_close(mean, payload["latent_mean"])
    torch.testing.assert_close(std, payload["latent_std"])


def test_16k_rejects_44k_cached_latent_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "variant": "small_44k",
        "latent_mean": torch.zeros(1, 1, 40),
        "latent_std": torch.ones(1, 1, 40),
    }

    class _ReceiverGroup:
        is_first_rank = False

        @staticmethod
        def broadcast_object(value, src=0):
            del value, src
            return payload

    monkeypatch.setattr(
        "fastvideo.train.models.mmaudio.mmaudio.get_world_group",
        lambda: _ReceiverGroup(),
    )
    with pytest.raises(ValueError, match="small_44k.*small_16k"):
        _load_or_compute_latent_stats(
            _training_config(),
            variant="small_16k",
            cache_path=None,
            chunk_size=1,
        )
