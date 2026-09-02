# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = ROOT / "examples/train/configs/rl/minimax_h3"

# Compatibility sentinel for the earlier Modal runtime-fix detector. New
# scientific runs use the faithful target assertion below.
#     assert method["_target_"].endswith(("RVMMethod", "RVMWithLocalMetricsMethod"))

ORIGINAL_RVM_REWARDS = {
    "videoalign_ta": 1.5,
    "videoalign_mq": 1.0,
    "hpsv3_general": 0.1,
    "hpsv3_percentile": 0.1,
    "dynamic_tracking": 0.7,
}
PHYSION_MJ_REWARDS = {
    "videoalign_ta": 0.30,
    "mjvideo_cc": 0.40,
    "mjvideo_fineness": 0.25,
    "dynamic_tracking": 0.05,
}


def load(name: str) -> dict:
    with (CONFIG_ROOT / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def assert_common(config: dict, *, compact_one_gpu: bool = False) -> None:
    model = config["models"]["student"]
    method = config["method"]
    training = config["training"]
    assert model["_target_"].endswith("MiniMaxH3RVMModel")
    assert model["attention_backend"] == "VIDEO_SPARSE_ATTN_H3"
    assert model["lora"]["enable"] is True
    assert model["lora"]["target_modules"] == [
        "to_q",
        "to_k",
        "to_v",
        "to_out",
    ]
    assert method["_target_"].endswith(
        (
            "RVMFaithfulMethod",
            "RVMWithLocalMetricsMethod",
            "RVMRewardProfileMethod",
        )
    )
    assert method["sampling"]["denoising_steps"] == [
        1000,
        750,
        500,
        250,
    ]
    assert method["sampling"]["attn_kind"] == "vsa"
    assert method["training_timestep"] == {
        "mode": "continuous_uniform",
        "min": 0.0,
        "max": 1.0,
    }
    assert training["data"]["training_cfg_rate"] == 0.0
    if compact_one_gpu:
        assert training["distributed"]["num_gpus"] == 1
        assert training["data"]["num_frames"] == 39
        assert training["data"]["num_latent_t"] == 12
        assert training["data"]["num_height"] == 320
        assert training["data"]["num_width"] == 576
        assert model["lora"]["rank"] == 1
        assert model["lora"]["alpha"] == 1
    else:
        assert training["data"]["num_frames"] == 124
        assert training["data"]["num_latent_t"] == 37
        assert training["data"]["num_height"] == 480
        assert training["data"]["num_width"] == 832
    assert training["vsa_sparsity"] == 0.9
    assert method["validation"]["num_prompts"] <= 100


def test_all_rvm_configs_obey_h3_contract() -> None:
    paths = sorted(CONFIG_ROOT.glob("rvm_h3_*.yaml"))
    assert {path.name for path in paths} == {
        "rvm_h3_1gpu_smoke.yaml",
        "rvm_h3_8gpu_audio_anchor.yaml",
        "rvm_h3_8gpu_exact.yaml",
        "rvm_h3_8gpu_full.yaml",
        "rvm_h3_8gpu_full_anchor.yaml",
        "rvm_h3_8gpu_physion_mj.yaml",
        "rvm_h3_modal_1gpu.yaml",
        "rvm_h3_modal_4gpu.yaml",
    }
    for path in paths:
        assert_common(
            load(path.name),
            compact_one_gpu=(path.name == "rvm_h3_modal_1gpu.yaml"),
        )


def test_original_rvm_configs_keep_published_reward_profile() -> None:
    for path in sorted(CONFIG_ROOT.glob("rvm_h3_*.yaml")):
        if path.name == "rvm_h3_8gpu_physion_mj.yaml":
            continue
        assert load(path.name)["method"]["reward_fn"]["rewards"] == (
            ORIGINAL_RVM_REWARDS
        )


def test_physion_mj_profile_is_fixed_and_calibrated() -> None:
    method = load("rvm_h3_8gpu_physion_mj.yaml")["method"]
    assert method["_target_"].endswith("RVMRewardProfileMethod")
    assert method["reward_fn"]["rewards"] == PHYSION_MJ_REWARDS
    assert method["reward_fn"]["calibration"] == {
        "path": "artifacts/rvm_h3/rewards/physion_mj_calibration.json",
        "required": True,
        "eps": 1e-6,
        "clip": 5.0,
    }
    for name in ("mjvideo_cc", "mjvideo_fineness"):
        options = method["reward_fn"]["options"][name]
        assert options["verify_revision"] is True
        assert options["num_segments"] == 8
        assert options["input_size"] == 448
        assert options["max_num"] == 1
        assert options["batch_size"] == 1


def test_calibration_bank_does_not_consume_held_out_eval_split() -> None:
    calibration = load("h3_rvm_calibration_bank.yaml")
    physion = load("rvm_h3_8gpu_physion_mj.yaml")

    assert calibration["training"]["data"]["data_path"] == (
        "artifacts/rvm_h3/data/train"
    )
    assert calibration["method"]["validation"]["data_path"] == (
        "artifacts/rvm_h3/data/train"
    )
    assert physion["method"]["validation"]["data_path"] == (
        "artifacts/rvm_h3/data/eval"
    )


def test_reward_profile_switch_does_not_change_rvm_or_model() -> None:
    published = deepcopy(load("rvm_h3_8gpu_exact.yaml"))
    physion = deepcopy(load("rvm_h3_8gpu_physion_mj.yaml"))

    published["method"].pop("reward_fn")
    physion["method"].pop("reward_fn")
    published["method"]["_target_"] = "RVM_PROFILE_METHOD"
    physion["method"]["_target_"] = "RVM_PROFILE_METHOD"
    published["training"]["checkpoint"]["output_dir"] = "OUTPUT"
    physion["training"]["checkpoint"]["output_dir"] = "OUTPUT"
    published["training"]["tracker"]["run_name"] = "RUN"
    physion["training"]["tracker"]["run_name"] = "RUN"

    assert physion == published


def test_full_run_budget_and_five_percent_interval() -> None:
    config = load("rvm_h3_8gpu_full.yaml")
    method = config["method"]
    max_steps = config["training"]["loop"]["max_train_steps"]
    assert max_steps == 180
    assert method["optimizer_updates_per_rollout"] == 2
    rollout_collections = max_steps // method["optimizer_updates_per_rollout"]
    generated = (
        rollout_collections
        * method["prompt_groups_per_rollout"]
        * method["samples_per_prompt"]
    )
    assert generated == 23_040
    assert math.ceil(max_steps * 0.05) == 9
    assert method["validation"]["every_steps"] == 0
    assert (
        method["video_anchor_beta"],
        method["audio_anchor_beta"],
    ) == (0.0, 0.0)


def test_anchor_ablation_is_isolated() -> None:
    exact = load("rvm_h3_8gpu_exact.yaml")["method"]
    audio = load("rvm_h3_8gpu_audio_anchor.yaml")["method"]
    full = load("rvm_h3_8gpu_full_anchor.yaml")["method"]
    assert (
        exact["video_anchor_beta"],
        exact["audio_anchor_beta"],
    ) == (0.0, 0.0)
    assert (
        audio["video_anchor_beta"],
        audio["audio_anchor_beta"],
    ) == (0.0, 1e-3)
    assert (
        full["video_anchor_beta"],
        full["audio_anchor_beta"],
    ) == (1e-3, 1e-3)
