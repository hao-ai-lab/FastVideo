# SPDX-License-Identifier: Apache-2.0
"""One-forward joint audio/video denoising loop for MiniMax H3."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.hooks.activation_trace import trace_step
from fastvideo.pipelines.basic.minimax_h3.stages._module_lifecycle import (
    maybe_offload_module,
    move_module_to_local_device,
)
from fastvideo.pipelines.basic.minimax_h3.types import get_minimax_h3_state
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE


class MiniMaxH3DenoisingStage(PipelineStage):
    """Denoise both modalities with one Transformer call per sigma interval."""

    def __init__(self, transformer: Any, scheduler: Any, audio_scheduler: Any) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.audio_scheduler = audio_scheduler

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("layout", state.layout, V.not_none)
        result.add_check("prompt_embeds", state.prompt_embeds, V.with_dims(3))
        result.add_check("video_latents", state.video_latents, V.with_dims(2))
        result.add_check("audio_latents", state.audio_latents, V.with_dims(2))
        result.add_check("video_timesteps", state.video_timesteps, V.with_dims(1))
        result.add_check("audio_timesteps", state.audio_timesteps, V.with_dims(1))
        result.add_check("row_timestep_plan", state.row_timestep_plan, V.list_not_empty)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("video_latents", state.video_latents, V.with_dims(2))
        result.add_check("audio_latents", state.audio_latents, V.with_dims(2))
        result.add_check("step_index", batch.step_index, V.non_negative_int)
        return result

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        state = get_minimax_h3_state(batch)
        if (state.layout is None or state.prompt_embeds is None or state.video_latents is None
                or state.audio_latents is None or state.video_timesteps is None or state.audio_timesteps is None):
            raise ValueError("MiniMax-H3 conditioning, layout, latents, and timesteps must precede denoising.")
        if len(state.row_timestep_plan) != len(state.video_timesteps):
            raise ValueError("MiniMax-H3 row timestep plan does not match the schedule.")

        full_cpu_offload = (bool(getattr(fastvideo_args, "dit_cpu_offload", False))
                            and not bool(getattr(fastvideo_args, "dit_layerwise_offload", False))
                            and not bool(getattr(fastvideo_args, "use_fsdp_inference", False)))
        moved_for_forward = False
        if full_cpu_offload:
            self.transformer, device, moved_for_forward = move_module_to_local_device(self.transformer)
            state.video_latents = state.video_latents.to(device)
            state.audio_latents = state.audio_latents.to(device)
        else:
            device = state.video_latents.device

        layout = state.layout
        position_ids = layout.position_ids.to(device)
        token_tags = layout.token_tags.to(device)
        video_indices = layout.video_indices.to(device)
        audio_indices = layout.audio_indices.to(device)
        text_indices = layout.text_indices.to(device)
        prompt_embeds = state.prompt_embeds.to(device)

        precision = getattr(fastvideo_args.pipeline_config, "dit_precision", "bf16")
        target_dtype = PRECISION_TO_TYPE.get(precision, torch.bfloat16)
        autocast_enabled = device.type == "cuda" and target_dtype != torch.float32 and not fastvideo_args.disable_autocast

        try:
            for index, (video_timestep,
                        audio_timestep) in enumerate(zip(state.video_timesteps, state.audio_timesteps, strict=False)):
                unique_timesteps, timestep_indices = state.row_timestep_plan[index]
                autocast = (torch.autocast(device_type="cuda", dtype=target_dtype, enabled=True)
                            if autocast_enabled else nullcontext())
                with trace_step(index), set_forward_context(
                        current_timestep=index,
                        attn_metadata=None,
                        forward_batch=batch,
                ), autocast:
                    video_velocity, audio_velocity = self.transformer(
                        hidden_states=state.video_latents[None],
                        audio_hidden_states=state.audio_latents[None],
                        encoder_hidden_states=prompt_embeds,
                        timestep=unique_timesteps.to(device),
                        timestep_indices=timestep_indices.to(device),
                        token_tags=token_tags,
                        position_ids=position_ids,
                        video_indices=video_indices,
                        audio_indices=audio_indices,
                        text_indices=text_indices,
                    )

                video_start = layout.num_condition_video_rows
                audio_start = layout.num_condition_audio_rows
                state.video_latents[video_start:] = self.scheduler.step(
                    video_velocity[0, video_start:].float(),
                    video_timestep,
                    state.video_latents[video_start:],
                    return_dict=False,
                )[0]
                state.audio_latents[audio_start:] = self.audio_scheduler.step(
                    audio_velocity[0, audio_start:].float(),
                    audio_timestep,
                    state.audio_latents[audio_start:],
                    return_dict=False,
                )[0]
                batch.step_index = index
                batch.timestep = video_timestep
        finally:
            if bool(getattr(fastvideo_args, "dit_layerwise_offload", False)):
                manager = getattr(self.transformer, "_layerwise_offload_manager", None)
                if manager is not None and getattr(manager, "enabled", False):
                    manager.release_all()
            if moved_for_forward:
                self.transformer = maybe_offload_module(self.transformer, enabled=True)
        return batch


__all__ = ["MiniMaxH3DenoisingStage"]
