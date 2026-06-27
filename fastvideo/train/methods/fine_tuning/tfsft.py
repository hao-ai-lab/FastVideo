# SPDX-License-Identifier: Apache-2.0
"""Teacher-forcing SFT method (TFSFT; algorithm layer).

Stage-1 AR-diffusion training from Causal-Forcing. Identical to
:class:`DiffusionForcingSFTMethod` (inhomogeneous per-chunk timesteps,
flow loss, bsmntw weighting) except the causal transformer denoises the
current block while attending to *clean* history (``clean_x``) instead of
its own noisy rollout. This is the "Causal Forcing" namesake mechanism and
the recommended initialization for the downstream ODE / consistency stages.
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.train.methods.fine_tuning.dfsft import (
    DiffusionForcingSFTMethod, )


class TeacherForcingSFTMethod(DiffusionForcingSFTMethod):
    """AR-diffusion SFT with clean-history teacher forcing."""

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
