# SPDX-License-Identifier: Apache-2.0
"""Wan conditioning and expert selection around the shared dense sampling loop."""

from dataclasses import dataclass

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.logger import init_logger
from fastvideo.pipelines.stages.denoising import DenoisingStage, DenoisingState
from fastvideo.utils import masks_like

logger = init_logger(__name__)


@dataclass
class WanDenoisingState(DenoisingState):
    boundary_timestep: float | None = None
    first_frame: torch.Tensor | None = None
    first_frame_mask: torch.Tensor | None = None
    timestep_sequence_length: int | None = None
    ti2v_task: bool = False
    lucy_edit_task: bool = False


class WanDenoisingStage(DenoisingStage):
    """Dense Wan sampling, including Wan2.2 experts, TI2V, and Lucy Edit.

    All conditioning state is per request. The stage receives normalized
    first-frame latents; it does not load or execute a VAE.
    """

    def prepare_denoising(self, batch, fastvideo_args, target_dtype) -> WanDenoisingState:
        config = fastvideo_args.pipeline_config
        latents = batch.latents
        assert latents is not None
        assert latents.shape[0] == 1, "only support batch size 1"
        boundary_ratio = config.dit_config.boundary_ratio
        if batch.boundary_ratio is not None:
            logger.info("Overriding boundary ratio from %s to %s", boundary_ratio, batch.boundary_ratio)
            boundary_ratio = batch.boundary_ratio
        state = WanDenoisingState(
            latents=latents,
            boundary_timestep=boundary_ratio *
            self.scheduler.num_train_timesteps if boundary_ratio is not None else None,
            ti2v_task=config.ti2v_task,
            lucy_edit_task=config.lucy_edit_task,
        )
        if config.ti2v_task and batch.pil_image is not None:
            assert batch.image_latent is None, "TI2V task should not have image latents"
            assert batch.first_frame_latent is not None, "Run WanFirstFrameEncodingStage before TI2V denoising"
            state.first_frame = batch.first_frame_latent
            latent_model_input = latents.to(target_dtype).squeeze(0)
            _, masks = masks_like([latent_model_input], zero=True)
            state.first_frame_mask = masks[0]
            state.latents = ((1. - masks[0]) * state.first_frame + masks[0] * latent_model_input).to(
                get_local_torch_device())
            temporal_scale = config.vae_config.arch_config.scale_factor_temporal
            spatial_scale = config.vae_config.arch_config.scale_factor_spatial
            patch_size = config.dit_config.arch_config.patch_size
            state.timestep_sequence_length = (((batch.num_frames - 1) // temporal_scale + 1) *
                                              (batch.height // spatial_scale) * (batch.width // spatial_scale) //
                                              (patch_size[1] * patch_size[2]))
        if state.lucy_edit_task:
            patch_size = config.dit_config.arch_config.patch_size
            assert patch_size[0] == 1, "Lucy Edit timestep expansion assumes temporal patch size 1"
            state.timestep_sequence_length = (state.latents.shape[2] * (state.latents.shape[3] // patch_size[1]) *
                                              (state.latents.shape[4] // patch_size[2]))
        if batch.video_latent is not None:
            state.video_padding = torch.zeros_like(state.latents)
        return state

    def select_model(self, timestep, batch, fastvideo_args, state):
        if state.boundary_timestep is None or timestep >= state.boundary_timestep:
            model, inactive, guidance = self.transformer, self.transformer_2, batch.guidance_scale
        else:
            model, inactive, guidance = self.transformer_2, self.transformer, batch.guidance_scale_2
        self.activate_transformer(model, inactive, fastvideo_args)
        return model, guidance

    def prepare_model_input(self, latents, batch, target_dtype, state):
        if batch.video_latent is not None and state.lucy_edit_task:
            return torch.cat([latents.to(target_dtype), batch.video_latent], dim=1).to(target_dtype)
        if batch.image_latent is not None:
            assert not state.ti2v_task, "image latents should not be provided for TI2V task"
        return super().prepare_model_input(latents, batch, target_dtype, state)

    def prepare_timestep(self, timestep, latent_model_input, batch, state):
        if state.lucy_edit_task:
            return timestep.repeat(latent_model_input.shape[0], state.timestep_sequence_length)
        if state.first_frame_mask is not None:
            step = torch.stack([timestep]).to(get_local_torch_device())
            expanded = (state.first_frame_mask[0][:, ::2, ::2] * step).flatten()
            expanded = torch.cat([
                expanded,
                expanded.new_ones(state.timestep_sequence_length - expanded.size(0)) * step,
            ])
            return expanded.unsqueeze(0).repeat(latent_model_input.shape[0], 1)
        return super().prepare_timestep(timestep, latent_model_input, batch, state)

    def finish_step(self, latents, state):
        if state.first_frame_mask is not None:
            latents = latents.squeeze(0)
            latents = (1. - state.first_frame_mask) * state.first_frame + state.first_frame_mask * latents
        return latents
