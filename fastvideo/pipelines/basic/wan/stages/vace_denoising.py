# SPDX-License-Identifier: Apache-2.0
"""VACE denoising hooks on top of the shared Wan dense loop."""

from dataclasses import dataclass

import torch

from fastvideo.pipelines.basic.wan.stages.denoising import WanDenoisingStage, WanDenoisingState
from fastvideo.pipelines.stages.denoising import DenoisingState


@dataclass
class WanVACEDenoisingState(WanDenoisingState):
    control_hidden_states: torch.Tensor | None = None
    control_hidden_states_scale: float | list[float] | torch.Tensor | None = None


class WanVACEDenoisingStage(WanDenoisingStage):

    def prepare_model_input(self, latents, batch, target_dtype, state):
        # VACE passes pixel control video via ``batch.video_latent`` and packs
        # DiT conditioning through ``control_hidden_states`` instead of channel concat.
        return latents.to(target_dtype)

    def prepare_denoising(self, batch, fastvideo_args, target_dtype) -> WanVACEDenoisingState:
        return WanVACEDenoisingState(
            latents=batch.latents,
            control_hidden_states=batch.vace_control_latents,
            control_hidden_states_scale=batch.conditioning_scale,
        )

    def prepare_family_transformer_kwargs(self, batch, state: DenoisingState, target_dtype) -> dict:
        if not isinstance(state, WanVACEDenoisingState) or state.control_hidden_states is None:
            return {}
        return {
            "control_hidden_states": state.control_hidden_states.to(target_dtype),
            "control_hidden_states_scale": state.control_hidden_states_scale,
        }
