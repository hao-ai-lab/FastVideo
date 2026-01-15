# SPDX-License-Identifier: Apache-2.0
"""
DMD denoising strategy (FlowMatch-based).
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising_strategies import (
    DenoisingStrategy,
    ModelInputs,
    StrategyState,
)
from fastvideo.utils import dict_to_3d_list

try:
    from fastvideo.attention.backends.sliding_tile_attn import (
        SlidingTileAttentionBackend)
    st_attn_available = True
except ImportError:
    st_attn_available = False
    SlidingTileAttentionBackend = None  # type: ignore

try:
    from fastvideo.attention.backends.video_sparse_attn import (
        VideoSparseAttentionBackend)
    vsa_available = True
except ImportError:
    vsa_available = False
    VideoSparseAttentionBackend = None  # type: ignore

logger = init_logger(__name__)


class DmdStrategy(DenoisingStrategy):

    def __init__(self, stage: Any) -> None:
        self.stage = stage

    def prepare(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> StrategyState:
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        timesteps = batch.timesteps
        if timesteps is None:
            raise ValueError("Timesteps must be provided")
        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.stage.scheduler.order

        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            assert torch.isnan(image_embeds[0]).sum() == 0
            image_embeds = [
                image_embed.to(target_dtype) for image_embed in image_embeds
            ]

        image_kwargs = self.stage.prepare_extra_func_kwargs(
            self.stage.transformer.forward,
            {
                "encoder_hidden_states_image": image_embeds,
                "mask_strategy": dict_to_3d_list(
                    None, t_max=50, l_max=60, h_max=24)
            },
        )

        pos_cond_kwargs = self.stage.prepare_extra_func_kwargs(
            self.stage.transformer.forward,
            {
                "encoder_hidden_states_2": batch.clip_embedding_pos,
                "encoder_attention_mask": batch.prompt_attention_mask,
            },
        )

        if (st_attn_available
                and self.stage.attn_backend == SlidingTileAttentionBackend):
            self.stage.prepare_sta_param(batch, fastvideo_args)

        assert batch.latents is not None, "latents must be provided"
        latents = batch.latents

        prompt_embeds = batch.prompt_embeds
        assert not torch.isnan(
            prompt_embeds[0]).any(), "prompt_embeds contains nan"

        timesteps = torch.tensor(
            fastvideo_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long,
            device=get_local_torch_device())

        progress_bar = self.stage.progress_bar(total=len(timesteps))

        extra: dict[str, Any] = {
            "batch": batch,
            "fastvideo_args": fastvideo_args,
            "target_dtype": target_dtype,
            "autocast_enabled": autocast_enabled,
            "num_warmup_steps": num_warmup_steps,
            "image_kwargs": image_kwargs,
            "pos_cond_kwargs": pos_cond_kwargs,
            "progress_bar": progress_bar,
            "video_raw_latent_shape": latents.shape,
        }

        return StrategyState(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=batch.negative_prompt_embeds,
            prompt_attention_mask=batch.prompt_attention_mask,
            negative_attention_mask=batch.negative_attention_mask,
            image_embeds=image_embeds,
            guidance_scale=batch.guidance_scale,
            guidance_scale_2=batch.guidance_scale_2,
            guidance_rescale=batch.guidance_rescale,
            do_cfg=batch.do_classifier_free_guidance,
            extra=extra,
        )

    def make_model_inputs(self, state: StrategyState, t: torch.Tensor,
                          step_idx: int) -> ModelInputs:
        state.extra["step_idx"] = step_idx
        return ModelInputs(
            latent_model_input=state.latents,
            timestep=t,
            prompt_embeds=state.prompt_embeds,
            prompt_attention_mask=state.prompt_attention_mask,
        )

    def forward(self, state: StrategyState,
                model_inputs: ModelInputs) -> torch.Tensor:
        batch = state.extra["batch"]
        fastvideo_args = state.extra["fastvideo_args"]
        target_dtype = state.extra["target_dtype"]
        autocast_enabled = state.extra["autocast_enabled"]
        step_idx = state.extra["step_idx"]

        if getattr(self.stage, "interrupt", False):
            state.extra["pred_latents"] = state.latents
            return state.latents

        latents = state.latents
        noise_latents = latents.clone()
        latent_model_input = latents.to(target_dtype)

        if batch.image_latent is not None:
            latent_model_input = torch.cat(
                [latent_model_input,
                 batch.image_latent.permute(0, 2, 1, 3, 4)],
                dim=2).to(target_dtype)

        t_expand = model_inputs.timestep.repeat(latent_model_input.shape[0])
        guidance_expand = None
        if fastvideo_args.pipeline_config.embedded_cfg_scale is not None:
            guidance_expand = (torch.tensor(
                [fastvideo_args.pipeline_config.embedded_cfg_scale] *
                latent_model_input.shape[0],
                dtype=torch.float32,
                device=get_local_torch_device(),
            ).to(target_dtype) * 1000.0)

        with torch.autocast(device_type="cuda",
                            dtype=target_dtype,
                            enabled=autocast_enabled):
            if (vsa_available
                    and self.stage.attn_backend == VideoSparseAttentionBackend):
                self.attn_metadata_builder_cls = (
                    self.stage.attn_backend.get_builder_cls())

                if self.attn_metadata_builder_cls is not None:
                    self.attn_metadata_builder = (
                        self.attn_metadata_builder_cls())
                    attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                        current_timestep=step_idx,  # type: ignore
                        raw_latent_shape=batch.
                        raw_latent_shape[2:5],  # type: ignore
                        patch_size=fastvideo_args.pipeline_config.dit_config.
                        patch_size,  # type: ignore
                        STA_param=batch.STA_param,  # type: ignore
                        VSA_sparsity=fastvideo_args.
                        VSA_sparsity,  # type: ignore
                        device=get_local_torch_device(),  # type: ignore
                    )
                    assert attn_metadata is not None, (
                        "attn_metadata cannot be None")
                else:
                    attn_metadata = None
            else:
                attn_metadata = None

            with set_forward_context(
                    current_timestep=step_idx,
                    attn_metadata=attn_metadata,
                    forward_batch=batch,
            ):
                pred_noise = self.stage.transformer(
                    latent_model_input.permute(0, 2, 1, 3, 4),
                    state.prompt_embeds,
                    t_expand,
                    guidance=guidance_expand,
                    **state.extra["image_kwargs"],
                    **state.extra["pos_cond_kwargs"],
                ).permute(0, 2, 1, 3, 4)

        pred_video = pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noise_latents.flatten(0, 1),
            timestep=t_expand,
            scheduler=self.stage.scheduler).unflatten(0, pred_noise.shape[:2])

        if step_idx < len(state.timesteps) - 1:
            next_timestep = state.timesteps[step_idx + 1] * torch.ones(
                [1], dtype=torch.long, device=pred_video.device)
            generator = batch.generator
            if isinstance(generator, list):
                generator = generator[0] if generator else None
            noise = torch.randn(state.extra["video_raw_latent_shape"],
                                dtype=pred_video.dtype,
                                generator=generator).to(self.stage.device)
            latents = self.stage.scheduler.add_noise(pred_video.flatten(0, 1),
                                                     noise.flatten(0, 1),
                                                     next_timestep).unflatten(
                                                         0,
                                                         pred_video.shape[:2])
        else:
            latents = pred_video

        state.extra["pred_latents"] = latents
        return latents

    def cfg_combine(self, state: StrategyState,
                    noise_pred: torch.Tensor) -> torch.Tensor:
        return noise_pred

    def scheduler_step(self, state: StrategyState, noise_pred: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
        progress_bar = state.extra["progress_bar"]
        step_idx = state.extra["step_idx"]
        num_warmup_steps = state.extra["num_warmup_steps"]

        if step_idx == len(state.timesteps) - 1 or (
            (step_idx + 1) > num_warmup_steps and
            (step_idx + 1) % self.stage.scheduler.order == 0
                and progress_bar is not None):
            progress_bar.update()

        return state.extra.get("pred_latents", state.latents)

    def postprocess(self, state: StrategyState) -> ForwardBatch:
        progress_bar = state.extra.get("progress_bar")
        if progress_bar is not None:
            progress_bar.close()

        batch = state.extra["batch"]
        latents = state.extra["pred_latents"].permute(0, 2, 1, 3, 4)
        batch.latents = latents
        return batch
