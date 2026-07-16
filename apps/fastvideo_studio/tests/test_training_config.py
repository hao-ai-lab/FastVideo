# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the studio -> fastvideo/train config mapping.

Every studio workload must produce a config dict that round-trips through
the real modular-trainer schema loader (``load_run_config``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from fastvideo_studio.training_config import (
    SUPPORTED_WORKLOADS,
    build_training_config,
    get_training_env,
    is_ltx2_model,
)

WAN = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _job(workload_type: str, **overrides: Any) -> dict[str, Any]:
    job: dict[str, Any] = {
        "model_id": WAN,
        "data_path": "outputs/datasets/my_dataset",
        "workload_type": workload_type,
        "num_gpus": 2,
        "max_train_steps": 1234,
        "train_batch_size": 1,
        "learning_rate": 5e-5,
        "num_latent_t": 20,
        "num_height": 480,
        "num_width": 832,
        "num_frames": 77,
        "validation_dataset_file": "",
        "lora_rank": 64,
    }
    if workload_type in ("dmd_t2v", "self_forcing_t2v"):
        job.update({
            "dmd_use_vsa": False,
            "dmd_vsa_sparsity": 0.8,
            "dmd_denoising_steps": "1000,757,522",
            "real_score_guidance_scale": 3.5,
            "generator_update_interval": 5,
            "real_score_model_path": "",
            "fake_score_model_path": "",
        })
    job.update(overrides)
    return job


@pytest.mark.parametrize("workload_type", sorted(SUPPORTED_WORKLOADS))
def test_config_round_trips_through_train_schema(workload_type: str, tmp_path: Path) -> None:
    """The generated config must parse with the real fastvideo/train loader."""
    # The loader transitively imports torch; skip (don't fail) where it's absent.
    config_mod = pytest.importorskip("fastvideo.train.utils.config")
    load_run_config = config_mod.load_run_config

    config = build_training_config(_job(workload_type), str(tmp_path / "out"))
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    cfg = load_run_config(str(path))
    assert cfg.training.distributed.num_gpus == 2
    assert cfg.training.distributed.hsdp_shard_dim == 2
    assert cfg.training.data.data_path == "outputs/datasets/my_dataset"
    assert cfg.training.loop.max_train_steps == 1234
    assert cfg.training.optimizer.learning_rate == pytest.approx(5e-5)
    if workload_type == "vsa_t2v":
        # Guards against training.vsa.sparsity schema drift (the loader
        # silently drops a mis-keyed sparsity value).
        assert cfg.training.vsa_sparsity == pytest.approx(0.8)
    assert cfg.training.checkpoint.output_dir == str(tmp_path / "out")
    # model_path derives from models.student.init_from.
    assert cfg.training.model_path == WAN
    assert "student" in cfg.models
    assert "grad_clip" in cfg.callbacks


def test_finetune_uses_finetune_method() -> None:
    config = build_training_config(_job("full_t2v"), "out")
    assert config["method"]["_target_"].endswith("FineTuneMethod")
    assert config["models"]["student"]["_target_"].endswith(".WanModel")
    assert "teacher" not in config["models"]
    assert config["training"]["vsa"]["sparsity"] == 0.0
    assert config["training"]["data"]["training_cfg_rate"] == 0.1


def test_vsa_finetune_sets_sparsity() -> None:
    config = build_training_config(_job("vsa_t2v"), "out")
    assert config["method"]["_target_"].endswith("FineTuneMethod")
    assert config["training"]["vsa"]["sparsity"] == 0.8


def test_lora_adds_lora_block() -> None:
    config = build_training_config(_job("lora_t2v"), "out")
    lora = config["models"]["student"]["lora"]
    assert lora["enable"] is True
    assert lora["rank"] == 64
    assert lora["alpha"] == 128
    assert lora["target_modules"] == ["to_q", "to_k", "to_v", "to_out"]


def test_dmd_builds_three_role_models_and_method_knobs() -> None:
    config = build_training_config(
        _job(
            "dmd_t2v",
            real_score_model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
            dmd_denoising_steps="1000, 757, 522",
        ),
        "out",
    )
    models = config["models"]
    assert models["student"]["_target_"].endswith(".WanModel")
    assert models["teacher"]["init_from"] == "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    assert models["teacher"]["trainable"] is False
    assert models["critic"]["init_from"] == WAN  # falls back to model_id
    method = config["method"]
    assert method["_target_"].endswith("DMD2Method")
    assert method["dmd_denoising_steps"] == [1000, 757, 522]
    assert method["fake_score_learning_rate"] == pytest.approx(5e-5)
    assert config["training"]["optimizer"]["betas"] == [0.0, 0.999]
    assert config["training"]["data"]["training_cfg_rate"] == 0.0
    assert config["pipeline"]["flow_shift"] == 8
    assert "ema" in config["callbacks"]


def test_dmd_vsa_maps_to_training_vsa_sparsity() -> None:
    config = build_training_config(
        _job("dmd_t2v", dmd_use_vsa=True, dmd_vsa_sparsity=0.9), "out")
    assert config["training"]["vsa"]["sparsity"] == 0.9


def test_self_forcing_uses_causal_student_and_rollout_knobs() -> None:
    config = build_training_config(_job("self_forcing_t2v"), "out")
    assert config["models"]["student"]["_target_"].endswith(".WanCausalModel")
    method = config["method"]
    assert method["_target_"].endswith("SelfForcingMethod")
    assert method["warp_denoising_step"] is True
    assert method["same_step_across_blocks"] is True
    assert config["pipeline"]["dit_config"] == {
        "local_attn_size": -1,
        "sink_size": 0,
    }


def test_ode_init_maps_to_kd_causal() -> None:
    config = build_training_config(_job("ode_init"), "out/job1")
    assert config["models"]["student"]["_target_"].endswith(".WanCausalModel")
    assert config["models"]["teacher"]["init_from"] == WAN
    method = config["method"]
    assert method["_target_"].endswith("KDCausalMethod")
    assert method["teacher_path_cache"] == "out/job1/kd_cache"


def test_validation_callback_only_when_file_given() -> None:
    without = build_training_config(_job("full_t2v"), "out")
    assert "validation" not in without["callbacks"]

    with_file = build_training_config(
        _job("full_t2v", validation_dataset_file="val.json"), "out")
    validation = with_file["callbacks"]["validation"]
    assert validation["dataset_file"] == "val.json"
    assert validation["pipeline_target"].endswith(".WanPipeline")
    assert validation["sampling_steps"] == [50]

    dmd = build_training_config(
        _job("dmd_t2v", validation_dataset_file="val.json"), "out")
    validation = dmd["callbacks"]["validation"]
    assert validation["pipeline_target"].endswith(".WanDMDPipeline")
    assert validation["sampling_steps"] == [3]
    assert validation["sampling_timesteps"] == [1000, 757, 522]

    # KD/ODE-init has no sampling-based validation pipeline.
    ode = build_training_config(
        _job("ode_init", validation_dataset_file="val.json"), "out")
    assert "validation" not in ode["callbacks"]


def test_ltx2_models_are_rejected() -> None:
    assert is_ltx2_model("Lightricks/LTX-2-19B")
    with pytest.raises(ValueError, match="LTX-2 training is not supported"):
        build_training_config(
            _job("full_t2v", model_id="Lightricks/LTX-2-19B"), "out")


def test_unknown_workload_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown workload type"):
        build_training_config(_job("full_i2v"), "out")


def test_invalid_denoising_steps_are_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid DMD denoising steps"):
        build_training_config(
            _job("dmd_t2v", dmd_denoising_steps="1000,abc"), "out")


def test_training_env_has_no_backend_override() -> None:
    env = get_training_env()
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["WANDB_MODE"] == "offline"
    assert "FASTVIDEO_ATTENTION_BACKEND" not in env


def test_workloads_match_frontend_job_config() -> None:
    """SUPPORTED_WORKLOADS must stay in sync with the UI's workload menu
    (src/lib/jobConfig.ts) — drift means creatable-but-unrunnable jobs."""
    import re

    job_config = (Path(__file__).resolve().parents[1] / "src" / "lib" /
                  "jobConfig.ts").read_text(encoding="utf-8")
    all_types = set(re.findall(r'type:\s*"([^"]+)"', job_config))
    inference_types = {"t2v", "i2v", "t2i"}
    assert inference_types <= all_types, "jobConfig.ts parse failed"
    assert all_types - inference_types == set(SUPPORTED_WORKLOADS)
