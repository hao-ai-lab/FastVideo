# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 denoising stage using the native sigma schedule.
"""

from __future__ import annotations

import math
import os

import torch
from tqdm.auto import tqdm

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.logger import init_logger
from fastvideo.models.dits.ltx2 import (
    AudioLatentShape, DEFAULT_LTX2_AUDIO_CHANNELS,
    DEFAULT_LTX2_AUDIO_DOWNSAMPLE, DEFAULT_LTX2_AUDIO_HOP_LENGTH,
    DEFAULT_LTX2_AUDIO_MEL_BINS, DEFAULT_LTX2_AUDIO_SAMPLE_RATE,
    VideoLatentShape)
from fastvideo.utils import PRECISION_TO_TYPE

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096

# Official distilled sigma schedule (8 denoising steps)
# From LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
DISTILLED_SIGMA_VALUES = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0
]

logger = init_logger(__name__)


def _ltx2_sigmas(
    steps: int,
    latent: torch.Tensor | None,
    device: torch.device,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> torch.Tensor:
    tokens = math.prod(
        latent.shape[2:]) if latent is not None else MAX_SHIFT_ANCHOR
    sigmas = torch.linspace(1.0,
                            0.0,
                            steps + 1,
                            device=device,
                            dtype=torch.float32)

    mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)
    b = base_shift - mm * BASE_SHIFT_ANCHOR
    sigma_shift = tokens * mm + b

    numerator = math.exp(sigma_shift)
    sigmas = torch.where(
        sigmas != 0,
        numerator / (numerator + (1 / sigmas - 1)),
        torch.zeros_like(sigmas),
    )

    if stretch:
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas = sigmas.clone()
        sigmas[non_zero_mask] = stretched

    return sigmas


class LTX2DenoisingStage(PipelineStage):
    """Run the LTX-2 denoising loop over the sigma schedule."""

    def __init__(self, transformer) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        if batch.latents is None:
            raise ValueError("Latents must be provided before denoising.")

        latents = batch.latents
        prompt_embeds = batch.prompt_embeds[0]
        prompt_mask = None

        neg_prompt_embeds = None
        neg_prompt_mask = None
        # Only load negative prompts if CFG is actually enabled
        if batch.do_classifier_free_guidance:
            assert batch.negative_prompt_embeds is not None, (
                "CFG is enabled but negative_prompt_embeds is None")
            neg_prompt_embeds = batch.negative_prompt_embeds[0]

        # Ensure text conditioning is on the same device as latents.
        if prompt_embeds.device != latents.device:
            prompt_embeds = prompt_embeds.to(latents.device)
        if prompt_mask is not None and prompt_mask.device != latents.device:
            prompt_mask = prompt_mask.to(latents.device)
        if neg_prompt_embeds is not None and neg_prompt_embeds.device != latents.device:
            neg_prompt_embeds = neg_prompt_embeds.to(latents.device)
        if neg_prompt_mask is not None and neg_prompt_mask.device != latents.device:
            neg_prompt_mask = neg_prompt_mask.to(latents.device)

        target_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.dit_precision]
        disable_autocast = os.getenv("LTX2_DISABLE_AUTOCAST", "1") == "1"
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast and (
                                not disable_autocast)

        # Use official distilled sigma schedule for 8 steps (distilled models)
        use_distilled_sigmas = os.getenv("LTX2_USE_DISTILLED_SIGMAS",
                                         "1") == "1"
        if use_distilled_sigmas and batch.num_inference_steps == 8:
            sigmas = torch.tensor(
                DISTILLED_SIGMA_VALUES,
                device=latents.device,
                dtype=torch.float32,
            )
            logger.info("[LTX2] Using official distilled sigma schedule")
        else:
            sigmas = _ltx2_sigmas(
                steps=batch.num_inference_steps,
                latent=None,
                device=latents.device,
            )
        if hasattr(self.transformer, "patchifier"):
            video_shape = VideoLatentShape.from_torch_shape(latents.shape)
            token_count = self.transformer.patchifier.get_token_count(
                video_shape)
        else:
            token_count = 1
        timestep_template = torch.ones(
            (latents.shape[0], token_count),
            device=latents.device,
            dtype=torch.float32,
        )
        audio_prompt_embeds = batch.extra.get("ltx2_audio_prompt_embeds")
        audio_neg_embeds = batch.extra.get("ltx2_audio_negative_embeds")
        audio_context_p = audio_prompt_embeds[0] if audio_prompt_embeds else None
        audio_context_n = audio_neg_embeds[0] if audio_neg_embeds else None
        audio_latents = None
        audio_timestep_template = None
        if audio_context_p is not None:
            fps_value = batch.fps
            if isinstance(fps_value, list):
                fps_value = fps_value[0] if fps_value else None
            if fps_value is None:
                fps_value = 1.0
            duration = float(batch.num_frames) / float(fps_value)
            audio_shape = AudioLatentShape.from_duration(
                batch=latents.shape[0],
                duration=duration,
                channels=DEFAULT_LTX2_AUDIO_CHANNELS,
                mel_bins=DEFAULT_LTX2_AUDIO_MEL_BINS,
                sample_rate=DEFAULT_LTX2_AUDIO_SAMPLE_RATE,
                hop_length=DEFAULT_LTX2_AUDIO_HOP_LENGTH,
                audio_latent_downsample_factor=DEFAULT_LTX2_AUDIO_DOWNSAMPLE,
            )
            audio_generator = None
            if fastvideo_args.ltx2_initial_latent_path and batch.seed is not None:
                audio_generator = torch.Generator(
                    device=latents.device).manual_seed(batch.seed)
            elif batch.generator is not None:
                if isinstance(batch.generator, list):
                    audio_generator = batch.generator[0]
                else:
                    audio_generator = batch.generator
            if audio_generator is not None and audio_generator.device.type != latents.device.type:
                if batch.seed is None:
                    audio_generator = torch.Generator(device=latents.device)
                else:
                    audio_generator = torch.Generator(
                        device=latents.device).manual_seed(batch.seed)
            audio_patch_shape = (
                audio_shape.batch,
                audio_shape.frames,
                audio_shape.channels * audio_shape.mel_bins,
            )
            audio_latents_patch = torch.randn(
                audio_patch_shape,
                generator=audio_generator,
                device=latents.device,
                dtype=latents.dtype,
            )
            if hasattr(self.transformer, "audio_patchifier"):
                audio_latents = self.transformer.audio_patchifier.unpatchify(
                    audio_latents_patch, audio_shape)
            else:
                audio_latents = audio_latents_patch.view(
                    audio_shape.batch,
                    audio_shape.frames,
                    audio_shape.channels,
                    audio_shape.mel_bins,
                ).permute(0, 2, 1, 3).contiguous()
            audio_timestep_template = torch.ones(
                (latents.shape[0], audio_shape.frames),
                device=latents.device,
                dtype=torch.float32,
            )
        logger.info(
            "[LTX2] Denoising start: steps=%d dtype=%s guidance=%s "
            "sigmas_shape=%s latents_shape=%s",
            batch.num_inference_steps,
            target_dtype,
            batch.guidance_scale,
            tuple(sigmas.shape),
            tuple(latents.shape),
        )

        for step_index in tqdm(range(len(sigmas) - 1)):
            sigma = sigmas[step_index]
            sigma_next = sigmas[step_index + 1]
            timestep = timestep_template * sigma
            audio_timestep = (audio_timestep_template * sigma
                              if audio_timestep_template is not None else None)

            with torch.autocast(
                    device_type="cuda",
                    dtype=target_dtype,
                    enabled=autocast_enabled,
            ), set_forward_context(
                    current_timestep=sigma.item(),
                    attn_metadata=None,
                    forward_batch=batch,
            ):
                pos_outputs = self.transformer(
                    hidden_states=latents.to(target_dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    timestep=timestep,
                    audio_hidden_states=audio_latents,
                    audio_encoder_hidden_states=audio_context_p,
                    audio_timestep=audio_timestep,
                )
                if isinstance(pos_outputs, tuple):
                    pos_denoised, pos_audio = pos_outputs
                else:
                    pos_denoised = pos_outputs
                    pos_audio = None

                # Only run negative pass if CFG is enabled
                if batch.do_classifier_free_guidance:
                    neg_outputs = self.transformer(
                        hidden_states=latents.to(target_dtype),
                        encoder_hidden_states=neg_prompt_embeds,
                        encoder_attention_mask=neg_prompt_mask,
                        timestep=timestep,
                        audio_hidden_states=audio_latents,
                        audio_encoder_hidden_states=audio_context_n,
                        audio_timestep=audio_timestep,
                    )
                    if isinstance(neg_outputs, tuple):
                        neg_denoised, neg_audio = neg_outputs
                    else:
                        neg_denoised = neg_outputs
                        neg_audio = None
                    pos_denoised = pos_denoised + (batch.guidance_scale - 1) * (
                        pos_denoised - neg_denoised)
                    if pos_audio is not None and neg_audio is not None:
                        pos_audio = pos_audio + (batch.guidance_scale -
                                                 1) * (pos_audio - neg_audio)

            sigma_value = sigma.to(torch.float32) if isinstance(
                sigma, torch.Tensor) else torch.tensor(
                    float(sigma),
                    device=latents.device,
                    dtype=torch.float32,
                )
            dt = sigma_next - sigma
            velocity = ((latents.float() - pos_denoised.float()) /
                        sigma_value).to(latents.dtype)
            latents = (latents.float() + velocity.float() * dt).to(
                latents.dtype)
            if pos_audio is not None and audio_latents is not None:
                audio_velocity = ((audio_latents.float() - pos_audio.float()) /
                                  sigma_value).to(audio_latents.dtype)
                audio_latents = (audio_latents.float() +
                                 audio_velocity.float() * dt).to(
                                     audio_latents.dtype)

        batch.latents = latents
        batch.extra["ltx2_audio_latents"] = audio_latents
        logger.info("[LTX2] Denoising done.")
        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("num_inference_steps", batch.num_inference_steps,
                         V.positive_int)
        return result
