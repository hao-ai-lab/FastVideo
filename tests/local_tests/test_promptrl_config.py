# SPDX-License-Identifier: Apache-2.0
"""PromptRLMethodConfig defaults, validation, and example YAML parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fastvideo.train.methods.rl.promptrl.config import (
    PromptRLMethodConfig,
    RoleOptimizerConfig,
    RolloutConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _REPO_ROOT / "examples" / "train" / "configs" / "rl" / "wan"


class TestDefaults:
    def test_plan_defaults(self):
        cfg = PromptRLMethodConfig.from_mapping({})
        assert cfg.mode == "prompt_only"
        assert cfg.group_size == 8
        assert cfg.retained_originals == 2
        assert cfg.format_reward_weight == 1.0
        assert cfg.video_reward_weight == 1.0
        assert cfg.refiner_kl_beta == 0.01
        assert cfg.student_kl_beta == 0.01
        assert cfg.ppo_clip == pytest.approx(1e-4)
        # Rollout defaults.
        assert cfg.rollout.num_steps == 20
        assert cfg.rollout.sde_steps == 8
        assert cfg.rollout.noise_scale == pytest.approx(0.8)
        assert cfg.rollout.loss_microbatch_size == 1
        # Per-role optimizer defaults.
        for role_cfg in (cfg.refiner_optimizer, cfg.generator_optimizer):
            assert role_cfg.learning_rate == pytest.approx(1e-5)
            assert role_cfg.betas == (0.9, 0.999)
            assert role_cfg.weight_decay == pytest.approx(0.01)
            assert role_cfg.max_grad_norm == pytest.approx(1.0)
        # Reward client defaults.
        assert cfg.reward.retries == 2
        assert cfg.reward.timeout_sec == pytest.approx(300.0)
        assert cfg.reward.fps == 16

    def test_mode_validation(self):
        with pytest.raises(ValueError, match="mode"):
            PromptRLMethodConfig.from_mapping({"mode": "everything"})

    def test_group_layout_validation(self):
        with pytest.raises(ValueError, match="retained_originals"):
            PromptRLMethodConfig.from_mapping({"group_size": 8, "retained_originals": 8})
        with pytest.raises(ValueError, match="group_size"):
            PromptRLMethodConfig.from_mapping({"group_size": 0})

    def test_rollout_validation(self):
        with pytest.raises(ValueError, match="sde_steps"):
            RolloutConfig.from_mapping({"num_steps": 4, "sde_steps": 9})
        with pytest.raises(ValueError, match="noise_scale"):
            RolloutConfig.from_mapping({"noise_scale": 0.0})

    def test_role_optimizer_betas_validation(self):
        with pytest.raises(ValueError, match="betas"):
            RoleOptimizerConfig.from_mapping({"betas": [0.9]}, where="test")

    def test_nested_overrides(self):
        cfg = PromptRLMethodConfig.from_mapping({
            "mode": "joint",
            "rollout": {"num_steps": 30, "sde_steps": 4},
            "refiner_optimizer": {"learning_rate": 2e-5},
            "reward": {"endpoint_url": "http://host:9999/", "retries": 5},
        })
        assert cfg.mode == "joint"
        assert cfg.rollout.num_steps == 30
        assert cfg.rollout.sde_steps == 4
        assert cfg.refiner_optimizer.learning_rate == pytest.approx(2e-5)
        assert cfg.reward.endpoint_url == "http://host:9999"  # trailing slash stripped
        assert cfg.reward.retries == 5


class TestExampleYamls:
    @pytest.mark.parametrize(
        "filename,expected_mode",
        [("promptrl_prompt_only.yaml", "prompt_only"), ("promptrl_joint.yaml", "joint")],
    )
    def test_example_config_parses(self, filename, expected_mode):
        path = _EXAMPLES / filename
        if not path.exists():
            pytest.skip(f"example config missing: {path}")
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        cfg = PromptRLMethodConfig.from_mapping(raw["method"])
        assert cfg.mode == expected_mode
        assert cfg.group_size == 8
        assert cfg.retained_originals == 2
        assert cfg.rollout.num_steps == 20
        assert cfg.rollout.sde_steps == 8
        # Canonical distributed layout.
        distributed = raw["training"]["distributed"]
        assert distributed["sp_size"] == 1
        assert distributed["hsdp_shard_dim"] == 8
        # Both roles use LoRA rank 16 / alpha 32.
        assert raw["models"]["student"]["lora"]["rank"] == 16
        assert raw["models"]["student"]["lora"]["alpha"] == 32
        assert raw["models"]["refiner"]["lora"]["rank"] == 16
        assert raw["models"]["refiner"]["lora"]["alpha"] == 32
        # 480x832 @ 77 frames target.
        assert raw["training"]["data"]["num_height"] == 480
        assert raw["training"]["data"]["num_width"] == 832
        assert raw["training"]["data"]["num_frames"] == 77

    def test_joint_config_initializes_from_prompt_only(self):
        path = _EXAMPLES / "promptrl_joint.yaml"
        if not path.exists():
            pytest.skip(f"example config missing: {path}")
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        refiner = raw["models"]["refiner"]
        assert "init_adapter_from" in refiner
        assert "prompt_only" in refiner["init_adapter_from"]
