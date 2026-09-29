# SPDX-License-Identifier: Apache-2.0
"""VACE denoising hooks on top of the shared Wan dense loop."""

from fastvideo.pipelines.basic.wan.stages.denoising import WanDenoisingStage
from fastvideo.pipelines.stages.denoising import DenoisingState


class WanVACEDenoisingStage(WanDenoisingStage):

    def prepare_model_input(self, latents, batch, target_dtype, state):
        # VACE passes pixel control video via ``batch.video_latent`` and packs
        # DiT conditioning through ``control_hidden_states`` instead of channel concat.
        return latents.to(target_dtype)

    def prepare_family_transformer_kwargs(self, batch, state: DenoisingState, target_dtype) -> dict:
        if batch.vace_control_latents is None:
            return {}
        # The transformer validates and broadcasts the per-VACE-layer scale.
        return {
            "control_hidden_states": batch.vace_control_latents.to(target_dtype),
            "control_hidden_states_scale": batch.conditioning_scale,
        }
