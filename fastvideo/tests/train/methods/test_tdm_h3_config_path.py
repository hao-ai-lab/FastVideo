# SPDX-License-Identifier: Apache-2.0
"""Regression for the H3 TDM validation grid.

``MiniMaxH3DenoisingStage`` rebuilds its schedule from
``callbacks.validation.sampling_steps`` (it ignores the caller's
``batch.timesteps``), and ``MiniMaxH3Scheduler`` turns N sigma points into
N-1 denoising forwards. Validation therefore lands on the trained ladder
only when ``sampling_steps`` selects the sigma grid whose forward
timesteps equal the model times the student is trained against.
"""

from __future__ import annotations

import torch
from torch.testing import assert_close

from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.train.models.minimax_h3.minimax_h3 import (
    tdm_h3_model_timestep,
    tdm_h3_shift,
    tdm_h3_sigma_grid,
)
from fastvideo.train.utils.config import load_run_config


_H3_TDM_CONFIG = "examples/train/configs/distribution_matching/overfit_minimax_h3_t2va_tdm.yaml"


def test_h3_tdm_validation_grid_reproduces_trained_ladder() -> None:
    cfg = load_run_config(_H3_TDM_CONFIG)

    ladder = cfg.method["tdm_denoising_steps"]
    validation = cfg.callbacks["validation"]
    sampling_steps = validation["sampling_steps"]

    assert sampling_steps == [5]
    assert validation["sampling_timesteps"] == ladder

    labels = torch.tensor([float(step) for step in ladder])
    for modality in ("video", "audio"):
        trained_model_timesteps = tdm_h3_model_timestep(tdm_h3_sigma_grid(labels, modality))

        scheduler = MiniMaxH3Scheduler(shift=tdm_h3_shift(modality))
        scheduler.set_timesteps(int(sampling_steps[0]))

        assert_close(
            scheduler.timesteps.cpu(),
            trained_model_timesteps,
            msg=f"H3 {modality} validation grid is off the trained ladder",
        )
