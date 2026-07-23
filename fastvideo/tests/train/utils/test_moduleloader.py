# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fastvideo.fastvideo_args import WorkloadType
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.moduleloader import _make_training_args


SFWAN_I2V_MODEL_ID = "FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers"


def _write_yaml(tmp_path: Path, data: dict[str, Any]) -> str:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def test_make_training_args_uses_inferred_i2v_workload(tmp_path: Path) -> None:
    cfg = load_run_config(_write_yaml(tmp_path, {
        "models": {
            "student": {
                "_target_": "fastvideo.train.models.wan.WanModel",
                "init_from": SFWAN_I2V_MODEL_ID,
            },
        },
        "method": {
            "_target_": "fastvideo.train.methods.fine_tuning.finetune.FineTuneMethod",
        },
        "training": {},
    }))

    args = _make_training_args(cfg.training, model_path=cfg.training.model_path)

    assert args.workload_type is WorkloadType.I2V
