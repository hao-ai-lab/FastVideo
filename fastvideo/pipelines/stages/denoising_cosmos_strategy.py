# SPDX-License-Identifier: Apache-2.0
"""
Cosmos denoising strategy using FlowMatchEulerDiscreteScheduler.
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising_strategies import (
    DenoisingStrategy,
    ModelInputs,
    StrategyState,
)

logger = init_logger(__name__)


class CosmosStrategy(DenoisingStrategy):

    def __init__(self, stage: Any) -> None:
        self.stage = stage

    def prepare(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> StrategyState:
        pipeline = self.stage.pipeline() if self.stage.pipeline else None
        if not fastvideo_args.model_loaded["transformer"]:
            loader = TransformerLoader()
            self.stage.transformer = loader.load(
                fastvideo_args.model_paths["transformer"], fastvideo_args)
            if pipeline:
                pipeline.add_module("transformer", self.stage.transformer)
            fastvideo_args.model_loaded["transformer"] = True

        extra_step_kwargs = self.stage.prepare_extra_func_kwargs(
            self.stage.scheduler.step,
            {
                "generator": batch.generator,
                "eta": batch.eta
            },
        )

        if hasattr(self.stage.transformer, "module"):
            transformer_dtype = next(
                self.stage.transformer.module.parameters()).dtype
        else:
            transformer_dtype = next(self.stage.transformer.parameters()).dtype
        target_dtype = transformer_dtype
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        latents = batch.latents
        num_inference_steps = batch.num_inference_steps

        sigma_max = 80.0
        sigma_min = 0.002
        sigma_data = 1.0
        final_sigmas_type = "sigma_min"

        if self.stage.scheduler is not None:
            self.stage.scheduler.register_to_config(
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                sigma_data=sigma_data,
                final_sigmas_type=final_sigmas_type,
            )

        self.stage.scheduler.set_timesteps(num_inference_steps,
                                           device=latents.device)
        timesteps = self.stage.scheduler.timesteps

        if (hasattr(self.stage.scheduler.config, "final_sigmas_type")
                and self.stage.scheduler.config.final_sigmas_type == "sigma_min"
                and len(self.stage.scheduler.sigmas) > 1):
            self.stage.scheduler.sigmas[-1] = self.stage.scheduler.sigmas[-2]

        conditioning_latents = getattr(batch, "conditioning_latents", None)
        unconditioning_latents = conditioning_latents

        progress_bar = self.stage.progress_bar(total=num_inference_steps)

        extra: dict[str, Any] = {
            "batch": batch,
            "fastvideo_args": fastvideo_args,
            "extra_step_kwargs": extra_step_kwargs,
            "target_dtype": target_dtype,
            "autocast_enabled": autocast_enabled,
            "progress_bar": progress_bar,
            "conditioning_latents": conditioning_latents,
            "unconditioning_latents": unconditioning_latents,
        }

        return StrategyState(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            prompt_embeds=batch.prompt_embeds,
            negative_prompt_embeds=batch.negative_prompt_embeds,
            prompt_attention_mask=batch.prompt_attention_mask,
            negative_attention_mask=batch.negative_attention_mask,
            image_embeds=batch.image_embeds,
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
        target_dtype = state.extra["target_dtype"]
        autocast_enabled = state.extra["autocast_enabled"]
        conditioning_latents = state.extra["conditioning_latents"]
        unconditioning_latents = state.extra["unconditioning_latents"]
        step_idx = state.extra["step_idx"]

        if getattr(self.stage, "interrupt", False):
            return state.latents

        current_sigma = self.stage.scheduler.sigmas[step_idx]
        current_t = current_sigma / (current_sigma + 1)
        c_in = 1 - current_t
        c_skip = 1 - current_t
        c_out = -current_t

        timestep = current_t.view(1, 1, 1, 1,
                                  1).expand(state.latents.size(0), -1,
                                            state.latents.size(2), -1, -1)

        with torch.autocast(device_type="cuda",
                            dtype=target_dtype,
                            enabled=autocast_enabled):
            cond_latent = state.latents * c_in

            if (hasattr(batch, "cond_indicator")
                    and batch.cond_indicator is not None
                    and conditioning_latents is not None):
                cond_latent = (batch.cond_indicator * conditioning_latents +
                               (1 - batch.cond_indicator) * cond_latent)
            else:
                logger.warning(
                    "Step %s: Missing conditioning data - "
                    "cond_indicator: %s, conditioning_latents: %s", step_idx,
                    hasattr(batch, "cond_indicator"), conditioning_latents
                    is not None)

            cond_latent = cond_latent.to(target_dtype)

            cond_timestep = timestep
            if hasattr(batch,
                       "cond_indicator") and batch.cond_indicator is not None:
                sigma_conditioning = 0.0001
                t_conditioning = sigma_conditioning / (sigma_conditioning + 1)
                cond_timestep = (batch.cond_indicator * t_conditioning +
                                 (1 - batch.cond_indicator) * timestep)
                cond_timestep = cond_timestep.to(target_dtype)

            with set_forward_context(
                    current_timestep=step_idx,
                    attn_metadata=None,
                    forward_batch=batch,
            ):
                condition_mask = (batch.cond_mask.to(target_dtype) if hasattr(
                    batch, "cond_mask") else None)
                padding_mask = torch.zeros(1,
                                           1,
                                           batch.height,
                                           batch.width,
                                           device=cond_latent.device,
                                           dtype=target_dtype)

                if condition_mask is None:
                    batch_size, _, num_frames, height, width = cond_latent.shape
                    condition_mask = torch.zeros(batch_size,
                                                 1,
                                                 num_frames,
                                                 height,
                                                 width,
                                                 device=cond_latent.device,
                                                 dtype=target_dtype)

                noise_pred = self.stage.transformer(
                    hidden_states=cond_latent,
                    timestep=cond_timestep.to(target_dtype),
                    encoder_hidden_states=batch.prompt_embeds[0].to(
                        target_dtype),
                    fps=24,
                    condition_mask=condition_mask,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]

            cond_pred = (c_skip * state.latents +
                         c_out * noise_pred.float()).to(target_dtype)

            if (hasattr(batch, "cond_indicator")
                    and batch.cond_indicator is not None
                    and conditioning_latents is not None):
                cond_pred = (batch.cond_indicator * conditioning_latents +
                             (1 - batch.cond_indicator) * cond_pred)

            if (state.do_cfg and batch.negative_prompt_embeds is not None):
                uncond_latent = state.latents * c_in

                if (hasattr(batch, "uncond_indicator")
                        and batch.uncond_indicator is not None
                        and unconditioning_latents is not None):
                    uncond_latent = (
                        batch.uncond_indicator * unconditioning_latents +
                        (1 - batch.uncond_indicator) * uncond_latent)

                with set_forward_context(
                        current_timestep=step_idx,
                        attn_metadata=None,
                        forward_batch=batch,
                ):
                    uncond_condition_mask = (
                        batch.uncond_mask.to(target_dtype) if
                        (hasattr(batch, "uncond_mask")
                         and batch.uncond_mask is not None) else condition_mask)

                    uncond_timestep = timestep
                    if (hasattr(batch, "uncond_indicator")
                            and batch.uncond_indicator is not None):
                        sigma_conditioning = 0.0001
                        t_conditioning = sigma_conditioning / (
                            sigma_conditioning + 1)
                        uncond_timestep = (
                            batch.uncond_indicator * t_conditioning +
                            (1 - batch.uncond_indicator) * timestep)
                        uncond_timestep = uncond_timestep.to(target_dtype)

                    noise_pred_uncond = self.stage.transformer(
                        hidden_states=uncond_latent.to(target_dtype),
                        timestep=uncond_timestep.to(target_dtype),
                        encoder_hidden_states=batch.negative_prompt_embeds[0].
                        to(target_dtype),
                        fps=24,
                        condition_mask=uncond_condition_mask,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0]

                uncond_pred = (
                    c_skip * state.latents +
                    c_out * noise_pred_uncond.float()).to(target_dtype)

                if (hasattr(batch, "uncond_indicator")
                        and batch.uncond_indicator is not None
                        and unconditioning_latents is not None):
                    uncond_pred = (
                        batch.uncond_indicator * unconditioning_latents +
                        (1 - batch.uncond_indicator) * uncond_pred)

                guidance_diff = cond_pred - uncond_pred
                final_pred = cond_pred + state.guidance_scale * guidance_diff
            else:
                final_pred = cond_pred

        if current_sigma > 1e-8:
            noise_for_scheduler = (state.latents - final_pred) / current_sigma
        else:
            logger.warning(
                "Step %s: current_sigma too small (%s), using final_pred directly",
                step_idx, current_sigma)
            noise_for_scheduler = final_pred

        if torch.isnan(noise_for_scheduler).sum() > 0:
            logger.error(
                "Step %s: NaN detected in noise_for_scheduler, sum: %s",
                step_idx,
                noise_for_scheduler.float().sum().item())
            logger.error(
                "Step %s: latents sum: %s, final_pred sum: %s, current_sigma: %s",
                step_idx,
                state.latents.float().sum().item(),
                final_pred.float().sum().item(), current_sigma)

        return noise_for_scheduler

    def cfg_combine(self, state: StrategyState,
                    noise_pred: torch.Tensor) -> torch.Tensor:
        return noise_pred

    def scheduler_step(self, state: StrategyState, noise_pred: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
        latents = self.stage.scheduler.step(
            noise_pred,
            t,
            state.latents,
            **state.extra["extra_step_kwargs"],
            return_dict=False,
        )[0]

        progress_bar = state.extra["progress_bar"]
        if progress_bar is not None:
            progress_bar.update()

        return latents

    def postprocess(self, state: StrategyState) -> ForwardBatch:
        progress_bar = state.extra.get("progress_bar")
        if progress_bar is not None:
            progress_bar.close()
        batch = state.extra["batch"]
        batch.latents = state.latents
        return batch
