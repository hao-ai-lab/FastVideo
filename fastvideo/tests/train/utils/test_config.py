# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for :func:`load_run_config`."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from fastvideo.train.utils.config import RunConfig, load_run_config
from fastvideo.train.utils.training_config import TrainingConfig


def _write_yaml(tmp_path: Path, data: dict[str, Any]) -> str:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def _minimal_yaml() -> dict[str, Any]:
    return {
        "models": {
            "student": {
                "_target_": "fastvideo.train.models.wan.WanModel",
                "init_from": "fake/model",
            },
        },
        "method": {
            "_target_":
            "fastvideo.train.methods.fine_tuning.finetune.FineTuneMethod",
        },
        "training": {},
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_minimal_yaml_loads_happy_path(tmp_path: Path) -> None:
    cfg = load_run_config(_write_yaml(tmp_path, _minimal_yaml()))

    assert isinstance(cfg, RunConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert cfg.method["_target_"] == (
        "fastvideo.train.methods.fine_tuning.finetune.FineTuneMethod")
    assert "student" in cfg.models
    assert cfg.callbacks == {}
    # raw retains the original YAML dict for downstream logging.
    assert "models" in cfg.raw and "method" in cfg.raw


def test_minimal_yaml_applies_all_defaults(tmp_path: Path) -> None:
    cfg = load_run_config(_write_yaml(tmp_path, _minimal_yaml()))
    t = cfg.training

    assert t.distributed.num_gpus == 1
    assert t.distributed.tp_size == 1
    assert t.distributed.sp_size == 1
    assert t.distributed.hsdp_replicate_dim == 1
    assert t.distributed.pin_cpu_memory is False

    assert t.data.train_batch_size == 1
    assert t.data.dataloader_num_workers == 0
    assert t.data.training_cfg_rate == 0.0
    assert t.data.seed == 0

    assert t.optimizer.learning_rate == 0.0
    assert t.optimizer.betas == (0.9, 0.999)
    assert t.optimizer.weight_decay == 0.0
    assert t.optimizer.lr_scheduler == "constant"
    assert t.optimizer.min_lr_ratio == 0.5

    assert t.loop.max_train_steps == 0
    assert t.loop.gradient_accumulation_steps == 1

    assert t.checkpoint.output_dir == ""
    assert t.checkpoint.checkpoints_total_limit == 0

    assert t.tracker.trackers == []
    assert t.tracker.project_name == "fastvideo"
    assert t.tracker.run_name == ""

    assert t.model.weighting_scheme == "uniform"
    assert t.model.precondition_outputs is False
    assert t.model.moba_config == {}

    assert t.dit_precision == "fp32"
    assert t.vsa_sparsity == 0.0
    assert t.pipeline_config is None


def test_full_yaml_populates_all_training_fields(tmp_path: Path) -> None:
    data = _minimal_yaml()
    data["training"] = {
        "distributed": {
            "num_gpus": 4,
            "tp_size": 2,
            "sp_size": 2,
            "hsdp_replicate_dim": 2,
            "hsdp_shard_dim": 2,
            "pin_cpu_memory": True,
        },
        "data": {
            "data_path": "/some/path",
            "train_batch_size": 2,
            "dataloader_num_workers": 4,
            "training_cfg_rate": 0.1,
            "seed": 42,
            "num_height": 256,
            "num_width": 512,
            "num_latent_t": 8,
            "num_frames": 33,
        },
        "optimizer": {
            "learning_rate": 1e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 100,
            "min_lr_ratio": 0.1,
        },
        "loop": {
            "max_train_steps": 1000,
            "gradient_accumulation_steps": 4,
        },
        "checkpoint": {
            "output_dir": "/out",
            "training_state_checkpointing_steps": 50,
            "checkpoints_total_limit": 3,
        },
        "tracker": {
            "trackers": ["wandb"],
            "project_name": "myproj",
            "run_name": "myrun",
        },
        "vsa": {
            "sparsity": 0.5
        },
        "model": {
            "weighting_scheme": "logit_normal",
            "logit_mean": 0.5,
            "logit_std": 1.5,
            "precondition_outputs": True,
        },
        "dit_precision": "bf16",
    }
    cfg = load_run_config(_write_yaml(tmp_path, data))
    t = cfg.training

    assert t.distributed.num_gpus == 4
    assert t.distributed.tp_size == 2
    assert t.distributed.pin_cpu_memory is True

    assert t.data.train_batch_size == 2
    assert t.data.num_frames == 33
    assert t.data.seed == 42

    assert t.optimizer.learning_rate == pytest.approx(1e-4)
    assert t.optimizer.betas == (0.9, 0.95)
    assert t.optimizer.lr_scheduler == "cosine"
    assert t.optimizer.lr_warmup_steps == 100
    assert t.optimizer.min_lr_ratio == pytest.approx(0.1)

    assert t.loop.max_train_steps == 1000
    assert t.loop.gradient_accumulation_steps == 4

    assert t.checkpoint.output_dir == "/out"
    assert t.checkpoint.checkpoints_total_limit == 3

    assert t.tracker.trackers == ["wandb"]
    assert t.tracker.project_name == "myproj"

    assert t.vsa_sparsity == pytest.approx(0.5)
    assert t.model.weighting_scheme == "logit_normal"
    assert t.model.precondition_outputs is True
    assert t.dit_precision == "bf16"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_missing_models_raises(tmp_path: Path) -> None:
    data = _minimal_yaml()
    del data["models"]
    with pytest.raises(ValueError, match="models"):
        load_run_config(_write_yaml(tmp_path, data))


def test_missing_method_raises(tmp_path: Path) -> None:
    data = _minimal_yaml()
    del data["method"]
    with pytest.raises(ValueError, match="method"):
        load_run_config(_write_yaml(tmp_path, data))


def test_missing_training_raises(tmp_path: Path) -> None:
    data = _minimal_yaml()
    del data["training"]
    with pytest.raises(ValueError, match="training"):
        load_run_config(_write_yaml(tmp_path, data))


def test_models_role_without_target_raises(tmp_path: Path) -> None:
    data = _minimal_yaml()
    data["models"]["student"] = {"init_from": "fake/model"}
    with pytest.raises(ValueError, match="_target_"):
        load_run_config(_write_yaml(tmp_path, data))


def test_method_without_target_raises(tmp_path: Path) -> None:
    data = _minimal_yaml()
    data["method"] = {"some_param": 5}
    with pytest.raises(ValueError, match="_target_"):
        load_run_config(_write_yaml(tmp_path, data))


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run_config(str(tmp_path / "does_not_exist.yaml"))


# ---------------------------------------------------------------------------
# Special parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("betas_value, expected", [
    ([0.8, 0.9], (0.8, 0.9)),
    ("0.9,0.999", (0.9, 0.999)),
])
def test_betas_parses_list_and_string_forms(
    tmp_path: Path,
    betas_value: Any,
    expected: tuple[float, float],
) -> None:
    data = _minimal_yaml()
    data["training"] = {"optimizer": {"betas": betas_value}}
    cfg = load_run_config(_write_yaml(tmp_path, data))
    assert cfg.training.optimizer.betas == expected


def test_dotted_overrides_apply_with_type_coercion(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, _minimal_yaml())
    overrides = [
        "--training.distributed.num_gpus=4",
        "--training.optimizer.learning_rate=1e-3",
        "--training.distributed.pin_cpu_memory=true",
        "--training.tracker.project_name=overridden",
    ]
    cfg = load_run_config(path, overrides=overrides)

    assert cfg.training.distributed.num_gpus == 4
    assert cfg.training.optimizer.learning_rate == pytest.approx(1e-3)
    assert cfg.training.distributed.pin_cpu_memory is True
    assert cfg.training.tracker.project_name == "overridden"


def test_dotted_overrides_accept_separate_value_token(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, _minimal_yaml())
    cfg = load_run_config(
        path,
        overrides=["--training.distributed.num_gpus", "8"],
    )
    assert cfg.training.distributed.num_gpus == 8


def test_overrides_create_intermediate_keys(tmp_path: Path) -> None:
    """Overrides into a nested key absent from YAML should still apply."""
    data = _minimal_yaml()
    # No `training.checkpoint` block in the minimal YAML.
    path = _write_yaml(tmp_path, data)
    cfg = load_run_config(
        path,
        overrides=["--training.checkpoint.checkpoints_total_limit=5"],
    )
    assert cfg.training.checkpoint.checkpoints_total_limit == 5


def test_hsdp_shard_dim_defaults_to_num_gpus(tmp_path: Path) -> None:
    """When unset, hsdp_shard_dim and sp_size fall back to num_gpus."""
    data = _minimal_yaml()
    data["training"] = {"distributed": {"num_gpus": 4}}
    cfg = load_run_config(_write_yaml(tmp_path, data))
    assert cfg.training.distributed.hsdp_shard_dim == 4
    assert cfg.training.distributed.sp_size == 4


def test_model_path_derived_from_student_init_from(tmp_path: Path) -> None:
    cfg = load_run_config(_write_yaml(tmp_path, _minimal_yaml()))
    assert cfg.training.model_path == "fake/model"


def test_callbacks_default_to_empty_when_absent(tmp_path: Path) -> None:
    cfg = load_run_config(_write_yaml(tmp_path, _minimal_yaml()))
    assert cfg.callbacks == {}


def test_callbacks_passed_through_when_present(tmp_path: Path) -> None:
    data = _minimal_yaml()
    data["callbacks"] = {
        "grad_clip": {
            "_target_":
            "fastvideo.train.callbacks.grad_clip.GradNormClipCallback",
            "max_grad_norm": 1.0,
        },
    }
    cfg = load_run_config(_write_yaml(tmp_path, data))
    assert "grad_clip" in cfg.callbacks
    assert cfg.callbacks["grad_clip"]["max_grad_norm"] == 1.0
