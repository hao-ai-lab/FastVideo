# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the shipped H3 joint video+audio TDM example config."""

from __future__ import annotations

from pathlib import Path

from fastvideo.train.methods.distribution_matching.tdm import TDMMethod
from fastvideo.train.models.minimax_h3 import MiniMaxH3Model
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.instantiate import resolve_target


def test_tdm_h3_config_resolves_without_loading_weights() -> None:
    config_path = Path("examples/train/configs/distribution_matching/overfit_minimax_h3_t2va_tdm.yaml")

    cfg = load_run_config(str(config_path))

    assert resolve_target(str(cfg.method["_target_"])) is TDMMethod
    for role in ("student", "teacher", "critic"):
        assert resolve_target(str(cfg.models[role]["_target_"])) is MiniMaxH3Model

    assert cfg.models["student"]["trainable"] is True
    assert cfg.models["teacher"]["trainable"] is False
    assert cfg.models["critic"]["trainable"] is True
    assert cfg.models["student"]["lora"]["enable"] is True
    assert cfg.models["critic"]["lora"]["enable"] is True
    assert "lora" not in cfg.models["teacher"]

    # H3 is guidance-distilled: guidance 1.0 and no unconditional branch.
    assert cfg.method["real_score_guidance_scale"] == 1.0
    assert "cfg_uncond" not in cfg.method

    assert cfg.method["tdm_denoising_steps"] == [1000, 750, 500, 250]
    assert cfg.method["warmup_steps"] == 50
    assert cfg.method["generator_update_interval"] == 1
    assert cfg.method["fake_score_learning_rate"] == 1.0e-4
    assert cfg.training.optimizer.learning_rate == 2.0e-5

    assert cfg.training.data.preprocessed_data_type == "t2va"
    assert cfg.training.data.train_batch_size == 1
    assert cfg.training.data.training_cfg_rate == 0.0
    assert cfg.training.data.num_latent_t == 37
    assert cfg.training.data.num_frames == 124
