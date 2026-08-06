# SPDX-License-Identifier: Apache-2.0
"""One-forward joint audio/video denoising for MiniMax H3."""

from __future__ import annotations

from typing import Any

import json
import os

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.hooks.activation_trace import trace_step
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MiniMaxH3PackedLayout,
    build_row_timesteps,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


class MiniMaxH3DenoisingStage(PipelineStage):
    """Build both schedules and denoise both modalities in one transformer call."""

    performance_component_metric = "dit_time_s"

    def __init__(self, transformer: Any, scheduler: Any, audio_scheduler: Any) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.audio_scheduler = audio_scheduler

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("layout", batch.extra.get(MINIMAX_H3_LAYOUT_KEY), V.not_none)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_of_tensors_dims(3))
        result.add_check("latents", batch.latents, V.with_dims(2))
        result.add_check("audio_latents", batch.audio_latents, V.with_dims(2))
        result.add_check("num_inference_steps", batch.num_inference_steps, V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, V.with_dims(2))
        result.add_check("audio_latents", batch.audio_latents, V.with_dims(2))
        result.add_check("timesteps", batch.timesteps, V.with_dims(1))
        result.add_check("step_index", batch.step_index, V.non_negative_int)
        return result

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        layout = batch.extra.get(MINIMAX_H3_LAYOUT_KEY)
        if not isinstance(layout, MiniMaxH3PackedLayout):
            raise ValueError("MiniMax-H3 packed layout is missing before denoising.")
        if not batch.prompt_embeds or batch.latents is None or batch.audio_latents is None:
            raise ValueError("MiniMax-H3 conditioning and packed latents must precede denoising.")

        full_cpu_offload = (bool(getattr(fastvideo_args, "dit_cpu_offload", False))
                            and not bool(getattr(fastvideo_args, "dit_layerwise_offload", False))
                            and not bool(getattr(fastvideo_args, "use_fsdp_inference", False)))
        device = get_local_torch_device()
        if full_cpu_offload:
            self.transformer.to(device)
            batch.latents = batch.latents.to(device)
            batch.audio_latents = batch.audio_latents.to(device)

        self.scheduler.set_timesteps(batch.num_inference_steps, device=device)
        self.audio_scheduler.set_timesteps(batch.num_inference_steps, device=device)
        video_timesteps = self.scheduler.timesteps
        audio_timesteps = self.audio_scheduler.timesteps
        if video_timesteps is None or audio_timesteps is None:
            raise ValueError("MiniMax-H3 schedulers did not produce timesteps.")
        if len(video_timesteps) != len(audio_timesteps):
            raise ValueError("MiniMax-H3 video and audio schedules must have the same number of intervals.")

        row_timestep_plan = []
        for video_timestep, audio_timestep in zip(video_timesteps, audio_timesteps, strict=True):
            video_value = float(video_timestep.item())
            audio_value = float(audio_timestep.item())
            unique, inverse = build_row_timesteps(
                layout,
                video_timestep=video_value,
                audio_timestep=audio_value,
                condition_video_timestep=max(video_value, MINIMAX_H3_KEYFRAME_NOISE_AUG),
                condition_audio_timestep=1.0,
            )
            row_timestep_plan.append((unique.to(device), inverse.to(device)))
        batch.timesteps = video_timesteps

        position_ids = layout.position_ids.to(device)
        token_tags = layout.token_tags.to(device)
        video_indices = layout.video_indices.to(device)
        audio_indices = layout.audio_indices.to(device)
        text_indices = layout.text_indices.to(device)
        prompt_embeds = batch.prompt_embeds[0].to(device)

        profile_steps = int(os.environ.get("FASTVIDEO_PROFILE_STEPS", "0"))
        profiler = None
        if profile_steps:
            # skip step 0 (allocator/lazy-init noise), capture the next N steps,
            # then stop — generation continues unprofiled. record_shapes lets the
            # analyzer attribute gemms to attention vs MLP by their K/N dims.
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
            )
        try:
            for index, (video_timestep, audio_timestep) in enumerate(zip(video_timesteps, audio_timesteps,
                                                                         strict=True)):
                if profiler is not None:
                    if index == 1:
                        profiler.start()
                    elif index == 1 + profile_steps:
                        profiler.stop()
                        rank = os.environ.get("RANK", "0")
                        out_dir = os.environ.get("FASTVIDEO_PROFILE_DIR", "/tmp")
                        rows = [{
                            "name": e.key,
                            "shapes": str(e.input_shapes),
                            "cuda_us": e.device_time_total,
                            "count": e.count,
                        } for e in profiler.key_averages(group_by_input_shape=True)]
                        with open(f"{out_dir}/h3_profile_rank{rank}.json", "w") as fh:
                            json.dump({"profiled_steps": profile_steps, "rows": rows}, fh)
                        with open(f"{out_dir}/h3_profile_rank{rank}.txt", "w") as fh:
                            fh.write(profiler.key_averages(group_by_input_shape=True).table(
                                sort_by="cuda_time_total", row_limit=60))
                        profiler = None
                unique_timesteps, timestep_indices = row_timestep_plan[index]
                # Under torch.compile(mode="reduce-overhead") each denoising
                # step must be marked, or cudagraph trees flag cross-step
                # reuse of pooled outputs as "accessing tensor output of
                # CUDAGraphs that has been overwritten" (surfaces at sp=1;
                # sp>1 is masked by collective-induced graph breaks).
                torch.compiler.cudagraph_mark_step_begin()
                with trace_step(index), set_forward_context(
                        current_timestep=index,
                        attn_metadata=None,
                        forward_batch=batch,
                ):
                    video_velocity, audio_velocity = self.transformer(
                        hidden_states=batch.latents[None],
                        audio_hidden_states=batch.audio_latents[None],
                        encoder_hidden_states=prompt_embeds,
                        timestep=unique_timesteps,
                        timestep_indices=timestep_indices,
                        token_tags=token_tags,
                        position_ids=position_ids,
                        video_indices=video_indices,
                        audio_indices=audio_indices,
                        text_indices=text_indices,
                    )

                video_start = layout.num_condition_video_rows
                audio_start = layout.num_condition_audio_rows
                batch.latents[video_start:] = self.scheduler.step(
                    video_velocity[0, video_start:].float(),
                    video_timestep,
                    batch.latents[video_start:],
                    return_dict=False,
                )[0]
                batch.audio_latents[audio_start:] = self.audio_scheduler.step(
                    audio_velocity[0, audio_start:].float(),
                    audio_timestep,
                    batch.audio_latents[audio_start:],
                    return_dict=False,
                )[0]
                batch.step_index = index
                batch.timestep = video_timestep
        finally:
            if bool(getattr(fastvideo_args, "dit_layerwise_offload", False)):
                manager = getattr(self.transformer, "_layerwise_offload_manager", None)
                if manager is not None and getattr(manager, "enabled", False):
                    manager.release_all()
            if full_cpu_offload:
                self.transformer.to("cpu")
        return batch


__all__ = ["MiniMaxH3DenoisingStage"]
