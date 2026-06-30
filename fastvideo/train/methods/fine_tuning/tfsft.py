# SPDX-License-Identifier: Apache-2.0
"""Teacher-forcing SFT method (TFSFT; algorithm layer)."""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.train.methods.fine_tuning.dfsft import (
    DiffusionForcingSFTMethod, )


class TeacherForcingSFTMethod(DiffusionForcingSFTMethod):

    def _predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        training_batch: Any,
        clean_latents: torch.Tensor,
    ) -> torch.Tensor:
        return self.student.predict_noise(
            noisy_latents,
            timestep,
            training_batch,
            conditional=True,
            attn_kind=self._attn_kind,
            clean_x=clean_latents,
        )
