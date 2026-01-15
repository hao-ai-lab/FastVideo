# SPDX-License-Identifier: Apache-2.0
"""
LongCat denoising strategies (base, I2V, VC).
"""

from __future__ import annotations

import time
from typing import Any

import torch
from tqdm import tqdm

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


class _BaseLongCatStrategy(DenoisingStrategy):

    def __init__(self, stage: Any) -> None:
        self.stage = stage

    def _load_transformer(self, batch: ForwardBatch,
                          fastvideo_args: FastVideoArgs,
                          attach_pipeline: bool) -> None:
        if fastvideo_args.model_loaded["transformer"]:
            return
        loader = TransformerLoader()
        self.stage.transformer = loader.load(
            fastvideo_args.model_paths["transformer"], fastvideo_args)
        if attach_pipeline:
            pipeline = self.stage.pipeline() if self.stage.pipeline else None
            if pipeline:
                pipeline.add_module("transformer", self.stage.transformer)
        fastvideo_args.model_loaded["transformer"] = True

    def _build_prompt_inputs(self, batch: ForwardBatch):
        prompt_embeds = batch.prompt_embeds[0]
        prompt_attention_mask = (batch.prompt_attention_mask[0]
                                 if batch.prompt_attention_mask else None)
        do_cfg = batch.do_classifier_free_guidance

        if do_cfg:
            negative_prompt_embeds = batch.negative_prompt_embeds[0]
            negative_prompt_attention_mask = (batch.negative_attention_mask[0]
                                              if batch.negative_attention_mask
                                              else None)
            prompt_embeds_combined = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0)
            if prompt_attention_mask is not None:
                prompt_attention_mask_combined = torch.cat(
                    [negative_prompt_attention_mask, prompt_attention_mask],
                    dim=0)
            else:
                prompt_attention_mask_combined = None
        else:
            prompt_embeds_combined = prompt_embeds
            prompt_attention_mask_combined = prompt_attention_mask

        return prompt_embeds, prompt_attention_mask, prompt_embeds_combined, \
            prompt_attention_mask_combined

    def optimized_scale(self, positive_flat: torch.Tensor,
                        negative_flat: torch.Tensor) -> torch.Tensor:
        """
        Calculate optimized scale from CFG-zero paper.

        st_star = (v_cond^T * v_uncond) / ||v_uncond||^2
        """
        dot_product = torch.sum(positive_flat * negative_flat,
                                dim=1,
                                keepdim=True)
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        return dot_product / squared_norm

    def cfg_combine(self, state: StrategyState,
                    noise_pred: torch.Tensor) -> torch.Tensor:
        return noise_pred


class LongCatStrategy(_BaseLongCatStrategy):

    def prepare(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> StrategyState:
        self._load_transformer(batch, fastvideo_args, attach_pipeline=True)

        if hasattr(self.stage.transformer, "module"):
            transformer_dtype = next(
                self.stage.transformer.module.parameters()).dtype
        else:
            transformer_dtype = next(self.stage.transformer.parameters()).dtype

        target_dtype = transformer_dtype
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        prompt_embeds, prompt_attention_mask, prompt_embeds_combined, \
            prompt_attention_mask_combined = self._build_prompt_inputs(batch)

        timesteps = batch.timesteps
        num_inference_steps = len(timesteps)
        progress_bar = tqdm(total=num_inference_steps, desc="LongCat Denoising")

        extra: dict[str, Any] = {
            "batch": batch,
            "target_dtype": target_dtype,
            "autocast_enabled": autocast_enabled,
            "prompt_embeds_combined": prompt_embeds_combined,
            "prompt_attention_mask_combined": prompt_attention_mask_combined,
            "progress_bar": progress_bar,
        }

        return StrategyState(
            latents=batch.latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            prompt_embeds=[prompt_embeds],
            negative_prompt_embeds=batch.negative_prompt_embeds,
            prompt_attention_mask=[prompt_attention_mask]
            if prompt_attention_mask is not None else None,
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
        target_dtype = state.extra["target_dtype"]
        latents = state.latents

        if state.do_cfg:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents

        latent_model_input = latent_model_input.to(target_dtype)
        timestep = t.expand(latent_model_input.shape[0]).to(target_dtype)

        state.extra["step_idx"] = step_idx
        return ModelInputs(
            latent_model_input=latent_model_input,
            timestep=timestep,
            prompt_embeds=state.prompt_embeds,
            prompt_attention_mask=state.prompt_attention_mask,
        )

    def forward(self, state: StrategyState,
                model_inputs: ModelInputs) -> torch.Tensor:
        batch = state.extra["batch"]
        target_dtype = state.extra["target_dtype"]
        autocast_enabled = state.extra["autocast_enabled"]
        prompt_embeds_combined = state.extra["prompt_embeds_combined"]
        prompt_attention_mask_combined = state.extra[
            "prompt_attention_mask_combined"]
        step_idx = state.extra["step_idx"]

        batch.is_cfg_negative = False
        with set_forward_context(
                current_timestep=step_idx,
                attn_metadata=None,
                forward_batch=batch,
        ), torch.autocast(device_type="cuda",
                          dtype=target_dtype,
                          enabled=autocast_enabled):
            noise_pred = self.stage.transformer(
                hidden_states=model_inputs.latent_model_input,
                encoder_hidden_states=prompt_embeds_combined,
                timestep=model_inputs.timestep,
                encoder_attention_mask=prompt_attention_mask_combined,
            )

        if state.do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            B = noise_pred_cond.shape[0]
            positive = noise_pred_cond.reshape(B, -1)
            negative = noise_pred_uncond.reshape(B, -1)
            st_star = self.optimized_scale(positive, negative)
            st_star = st_star.view(B, 1, 1, 1, 1)
            noise_pred = (noise_pred_uncond * st_star + state.guidance_scale *
                          (noise_pred_cond - noise_pred_uncond * st_star))

        noise_pred = -noise_pred
        return noise_pred

    def scheduler_step(self, state: StrategyState, noise_pred: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
        latents = self.stage.scheduler.step(noise_pred,
                                            t,
                                            state.latents,
                                            return_dict=False)[0]

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


class LongCatI2VStrategy(_BaseLongCatStrategy):

    def prepare(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> StrategyState:
        self._load_transformer(batch, fastvideo_args, attach_pipeline=False)

        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        prompt_embeds, prompt_attention_mask, prompt_embeds_combined, \
            prompt_attention_mask_combined = self._build_prompt_inputs(batch)

        timesteps = batch.timesteps
        num_inference_steps = len(timesteps)
        progress_bar = tqdm(total=num_inference_steps, desc="I2V Denoising")

        num_cond_latents = getattr(batch, "num_cond_latents", 0)
        if num_cond_latents > 0:
            logger.info("I2V Denoising: num_cond_latents=%s, latent_shape=%s",
                        num_cond_latents, batch.latents.shape)

        extra: dict[str, Any] = {
            "batch": batch,
            "target_dtype": target_dtype,
            "autocast_enabled": autocast_enabled,
            "prompt_embeds_combined": prompt_embeds_combined,
            "prompt_attention_mask_combined": prompt_attention_mask_combined,
            "num_cond_latents": num_cond_latents,
            "progress_bar": progress_bar,
        }

        return StrategyState(
            latents=batch.latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            prompt_embeds=[prompt_embeds],
            negative_prompt_embeds=batch.negative_prompt_embeds,
            prompt_attention_mask=[prompt_attention_mask]
            if prompt_attention_mask is not None else None,
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
        target_dtype = state.extra["target_dtype"]
        num_cond_latents = state.extra["num_cond_latents"]
        latents = state.latents

        if state.do_cfg:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents

        latent_model_input = latent_model_input.to(target_dtype)
        timestep = t.expand(latent_model_input.shape[0]).to(target_dtype)
        timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
        if num_cond_latents > 0:
            timestep[:, :num_cond_latents] = 0

        state.extra["step_idx"] = step_idx
        return ModelInputs(
            latent_model_input=latent_model_input,
            timestep=timestep,
            prompt_embeds=state.prompt_embeds,
            prompt_attention_mask=state.prompt_attention_mask,
            extra_kwargs={"num_cond_latents": num_cond_latents},
        )

    def forward(self, state: StrategyState,
                model_inputs: ModelInputs) -> torch.Tensor:
        batch = state.extra["batch"]
        target_dtype = state.extra["target_dtype"]
        autocast_enabled = state.extra["autocast_enabled"]
        prompt_embeds_combined = state.extra["prompt_embeds_combined"]
        prompt_attention_mask_combined = state.extra[
            "prompt_attention_mask_combined"]
        step_idx = state.extra["step_idx"]
        num_cond_latents = state.extra["num_cond_latents"]

        batch.is_cfg_negative = False
        with set_forward_context(
                current_timestep=step_idx,
                attn_metadata=None,
                forward_batch=batch,
        ), torch.autocast(device_type="cuda",
                          dtype=target_dtype,
                          enabled=autocast_enabled):
            noise_pred = self.stage.transformer(
                hidden_states=model_inputs.latent_model_input,
                encoder_hidden_states=prompt_embeds_combined,
                timestep=model_inputs.timestep,
                encoder_attention_mask=prompt_attention_mask_combined,
                num_cond_latents=num_cond_latents,
            )

        if state.do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            B = noise_pred_cond.shape[0]
            positive = noise_pred_cond.reshape(B, -1)
            negative = noise_pred_uncond.reshape(B, -1)
            st_star = self.optimized_scale(positive, negative)
            st_star = st_star.view(B, 1, 1, 1, 1)
            noise_pred = (noise_pred_uncond * st_star + state.guidance_scale *
                          (noise_pred_cond - noise_pred_uncond * st_star))

        noise_pred = -noise_pred
        return noise_pred

    def scheduler_step(self, state: StrategyState, noise_pred: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
        num_cond_latents = state.extra["num_cond_latents"]
        latents = state.latents
        if num_cond_latents > 0:
            latents[:, :, num_cond_latents:] = self.stage.scheduler.step(
                noise_pred[:, :, num_cond_latents:],
                t,
                latents[:, :, num_cond_latents:],
                return_dict=False)[0]
        else:
            latents = self.stage.scheduler.step(noise_pred,
                                                t,
                                                latents,
                                                return_dict=False)[0]

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


class LongCatVCStrategy(_BaseLongCatStrategy):

    def prepare(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> StrategyState:
        self._load_transformer(batch, fastvideo_args, attach_pipeline=False)

        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        prompt_embeds, prompt_attention_mask, prompt_embeds_combined, \
            prompt_attention_mask_combined = self._build_prompt_inputs(batch)

        timesteps = batch.timesteps
        num_inference_steps = len(timesteps)
        progress_bar = tqdm(total=num_inference_steps, desc="VC Denoising")

        num_cond_latents = getattr(batch, "num_cond_latents", 0)
        use_kv_cache = getattr(batch, "use_kv_cache", False)
        kv_cache_dict = getattr(batch, "kv_cache_dict", {})

        logger.info(
            "VC Denoising: num_cond_latents=%d, use_kv_cache=%s, latent_shape=%s",
            num_cond_latents, use_kv_cache, batch.latents.shape)

        extra: dict[str, Any] = {
            "batch": batch,
            "target_dtype": target_dtype,
            "autocast_enabled": autocast_enabled,
            "prompt_embeds_combined": prompt_embeds_combined,
            "prompt_attention_mask_combined": prompt_attention_mask_combined,
            "num_cond_latents": num_cond_latents,
            "use_kv_cache": use_kv_cache,
            "kv_cache_dict": kv_cache_dict,
            "progress_bar": progress_bar,
            "step_times": [],
        }

        return StrategyState(
            latents=batch.latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            prompt_embeds=[prompt_embeds],
            negative_prompt_embeds=batch.negative_prompt_embeds,
            prompt_attention_mask=[prompt_attention_mask]
            if prompt_attention_mask is not None else None,
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
        target_dtype = state.extra["target_dtype"]
        num_cond_latents = state.extra["num_cond_latents"]
        use_kv_cache = state.extra["use_kv_cache"]
        latents = state.latents

        if state.do_cfg:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents

        latent_model_input = latent_model_input.to(target_dtype)
        timestep = t.expand(latent_model_input.shape[0]).to(target_dtype)
        timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
        if not use_kv_cache and num_cond_latents > 0:
            timestep[:, :num_cond_latents] = 0

        state.extra["step_idx"] = step_idx
        state.extra["step_start"] = time.time()

        extra_kwargs = {"num_cond_latents": num_cond_latents}
        if use_kv_cache:
            extra_kwargs["kv_cache_dict"] = state.extra["kv_cache_dict"]

        return ModelInputs(
            latent_model_input=latent_model_input,
            timestep=timestep,
            prompt_embeds=state.prompt_embeds,
            prompt_attention_mask=state.prompt_attention_mask,
            extra_kwargs=extra_kwargs,
        )

    def forward(self, state: StrategyState,
                model_inputs: ModelInputs) -> torch.Tensor:
        batch = state.extra["batch"]
        target_dtype = state.extra["target_dtype"]
        autocast_enabled = state.extra["autocast_enabled"]
        prompt_embeds_combined = state.extra["prompt_embeds_combined"]
        prompt_attention_mask_combined = state.extra[
            "prompt_attention_mask_combined"]
        step_idx = state.extra["step_idx"]

        batch.is_cfg_negative = False
        with set_forward_context(
                current_timestep=step_idx,
                attn_metadata=None,
                forward_batch=batch,
        ), torch.autocast(device_type="cuda",
                          dtype=target_dtype,
                          enabled=autocast_enabled):
            noise_pred = self.stage.transformer(
                hidden_states=model_inputs.latent_model_input,
                encoder_hidden_states=prompt_embeds_combined,
                timestep=model_inputs.timestep,
                encoder_attention_mask=prompt_attention_mask_combined,
                **model_inputs.extra_kwargs,
            )

        if state.do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            B = noise_pred_cond.shape[0]
            positive = noise_pred_cond.reshape(B, -1)
            negative = noise_pred_uncond.reshape(B, -1)
            st_star = self.optimized_scale(positive, negative)
            st_star = st_star.view(B, 1, 1, 1, 1)
            noise_pred = (noise_pred_uncond * st_star + state.guidance_scale *
                          (noise_pred_cond - noise_pred_uncond * st_star))

        noise_pred = -noise_pred
        return noise_pred

    def scheduler_step(self, state: StrategyState, noise_pred: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
        num_cond_latents = state.extra["num_cond_latents"]
        use_kv_cache = state.extra["use_kv_cache"]
        latents = state.latents

        if use_kv_cache:
            latents = self.stage.scheduler.step(noise_pred,
                                                t,
                                                latents,
                                                return_dict=False)[0]
        else:
            if num_cond_latents > 0:
                latents[:, :, num_cond_latents:] = self.stage.scheduler.step(
                    noise_pred[:, :, num_cond_latents:],
                    t,
                    latents[:, :, num_cond_latents:],
                    return_dict=False,
                )[0]
            else:
                latents = self.stage.scheduler.step(noise_pred,
                                                    t,
                                                    latents,
                                                    return_dict=False)[0]

        step_time = time.time() - state.extra["step_start"]
        state.extra["step_times"].append(step_time)
        if state.extra["step_idx"] < 3:
            logger.info("Step %d: %.2fs", state.extra["step_idx"], step_time)

        progress_bar = state.extra["progress_bar"]
        if progress_bar is not None:
            progress_bar.update()

        return latents

    def postprocess(self, state: StrategyState) -> ForwardBatch:
        progress_bar = state.extra.get("progress_bar")
        if progress_bar is not None:
            progress_bar.close()

        batch = state.extra["batch"]
        use_kv_cache = state.extra["use_kv_cache"]
        step_times = state.extra["step_times"]

        latents = state.latents
        if use_kv_cache and hasattr(
                batch, "cond_latents") and batch.cond_latents is not None:
            latents = torch.cat([batch.cond_latents, latents], dim=2)
            logger.info(
                "Concatenated conditioning latents back, final shape: %s",
                latents.shape)

        if step_times:
            avg_time = sum(step_times) / len(step_times)
            logger.info("Average step time: %.2fs (total: %.1fs)", avg_time,
                        sum(step_times))

        batch.latents = latents
        return batch
