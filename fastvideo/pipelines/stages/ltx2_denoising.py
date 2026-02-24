# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 denoising stage using the native sigma schedule.
"""

from __future__ import annotations

import math
import os

import torch
from tqdm.auto import tqdm

import fastvideo.envs as envs
from fastvideo.attention.backends.video_sparse_attn import (
    VideoSparseAttentionMetadataBuilder)
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
from fastvideo.utils import is_vsa_available

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096

# Official distilled sigma schedule (8 denoising steps)
# From LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
DISTILLED_SIGMA_VALUES = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0
]

logger = init_logger(__name__)

try:
    vsa_available = is_vsa_available()
except ImportError:
    vsa_available = False


def _ltx2_sigmas(
    steps: int,
    latent: torch.Tensor | None,
    device: torch.device,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> torch.Tensor:
    # Copied/following official LTX-2 scheduler (LTX2Scheduler.execute).
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
            if not batch.negative_prompt_embeds:
                raise ValueError(
                    "CFG is enabled but negative_prompt_embeds is empty")
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

        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

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
            logger.info("[LTX2] Using computed sigma schedule")
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
        # Multi-modal CFG parameters (per-stream scales).
        cfg_scale_video = batch.ltx2_cfg_scale_video
        cfg_scale_audio = batch.ltx2_cfg_scale_audio
        modality_scale_video = batch.ltx2_modality_scale_video
        modality_scale_audio = batch.ltx2_modality_scale_audio
        rescale_scale = batch.ltx2_rescale_scale
        # STG (Spatio-Temporal Guidance) parameters.
        stg_scale_video = batch.ltx2_stg_scale_video
        stg_scale_audio = batch.ltx2_stg_scale_audio
        stg_blocks_video = batch.ltx2_stg_blocks_video
        stg_blocks_audio = batch.ltx2_stg_blocks_audio
        do_stg_video = stg_scale_video != 0.0
        do_stg_audio = stg_scale_audio != 0.0
        do_stg = do_stg_video or do_stg_audio
        do_cfg_text = (cfg_scale_video != 1.0 or cfg_scale_audio != 1.0)
        do_mod = (modality_scale_video != 1.0 or modality_scale_audio != 1.0)
        do_guidance = do_cfg_text or do_mod or do_stg

        if do_cfg_text and neg_prompt_embeds is None:
            raise ValueError("LTX-2 text CFG is enabled "
                             "(ltx2_cfg_scale_video/audio != 1.0), "
                             "but negative prompt embeddings are missing")

        logger.info(
            "[LTX2] Denoising start: steps=%d dtype=%s "
            "cfg_video=%.1f cfg_audio=%.1f mod_video=%.1f "
            "mod_audio=%.1f rescale=%.2f "
            "stg_video=%.1f stg_audio=%.1f "
            "sigmas_shape=%s latents_shape=%s",
            batch.num_inference_steps,
            target_dtype,
            cfg_scale_video,
            cfg_scale_audio,
            modality_scale_video,
            modality_scale_audio,
            rescale_scale,
            stg_scale_video,
            stg_scale_audio,
            tuple(sigmas.shape),
            tuple(latents.shape),
        )
        use_vsa = (vsa_available
                   and envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN")
        vsa_metadata_builder = (VideoSparseAttentionMetadataBuilder()
                                if use_vsa else None)

        for step_index in tqdm(range(len(sigmas) - 1)):
            sigma = sigmas[step_index]
            sigma_next = sigmas[step_index + 1]
            timestep = timestep_template * sigma
            audio_timestep = (audio_timestep_template * sigma
                              if audio_timestep_template is not None else None)
            attn_metadata = None
            if vsa_metadata_builder is not None:
                attn_metadata = vsa_metadata_builder.build(
                    current_timestep=step_index,
                    raw_latent_shape=latents.shape[2:5],
                    patch_size=fastvideo_args.pipeline_config.dit_config.
                    patch_size,
                    VSA_sparsity=fastvideo_args.VSA_sparsity,
                    device=latents.device,
                )

            with torch.autocast(
                    device_type="cuda",
                    dtype=target_dtype,
                    enabled=autocast_enabled,
            ), set_forward_context(
                    current_timestep=sigma.item(),
                    attn_metadata=attn_metadata,
                    forward_batch=batch,
            ):
                # Pass 1: Full conditioning (text + cross-modal)
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

                if do_guidance:
                    # Defaults: (pos - pos) = 0 under each scale.
                    neg_denoised = pos_denoised
                    neg_audio = pos_audio
                    mod_denoised = pos_denoised
                    mod_audio = pos_audio
                    ptb_denoised = pos_denoised
                    ptb_audio = pos_audio

                    # Pass 2: text CFG (negative prompt)
                    if do_cfg_text:
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

                    # Pass 3: Modality-isolated (skip cross-modal
                    # attn)
                    if do_mod:
                        mod_outputs = self.transformer(
                            hidden_states=latents.to(target_dtype),
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_mask,
                            timestep=timestep,
                            audio_hidden_states=audio_latents,
                            audio_encoder_hidden_states=audio_context_p,
                            audio_timestep=audio_timestep,
                            skip_cross_modal_attn=True,
                        )
                        if isinstance(mod_outputs, tuple):
                            mod_denoised, mod_audio = mod_outputs
                        else:
                            mod_denoised = mod_outputs

                    # Pass 4: STG perturbed (skip self-attn in
                    # specified blocks)
                    if do_stg:
                        ptb_outputs = self.transformer(
                            hidden_states=latents.to(target_dtype),
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_mask,
                            timestep=timestep,
                            audio_hidden_states=audio_latents,
                            audio_encoder_hidden_states=(audio_context_p),
                            audio_timestep=audio_timestep,
                            skip_video_self_attn_blocks=(
                                stg_blocks_video if do_stg_video else None),
                            skip_audio_self_attn_blocks=(
                                stg_blocks_audio if do_stg_audio else None),
                        )
                        if isinstance(ptb_outputs, tuple):
                            ptb_denoised, ptb_audio = ptb_outputs
                        else:
                            ptb_denoised = ptb_outputs

                    # Multi-modal guidance formula per stream.
                    vid = (pos_denoised + (cfg_scale_video - 1) *
                           (pos_denoised - neg_denoised) +
                           (modality_scale_video - 1) *
                           (pos_denoised - mod_denoised) + stg_scale_video *
                           (pos_denoised - ptb_denoised))
                    aud = None
                    if pos_audio is not None:
                        aud = (pos_audio + (cfg_scale_audio - 1) *
                               (pos_audio - neg_audio) +
                               (modality_scale_audio - 1) *
                               (pos_audio - mod_audio) + stg_scale_audio *
                               (pos_audio - ptb_audio))

                    # Guidance rescaling (prevents saturation).
                    if rescale_scale > 0:
                        f_v = pos_denoised.std() / vid.std()
                        f_v = rescale_scale * f_v + (1 - rescale_scale)
                        vid = vid * f_v
                        if aud is not None:
                            f_a = pos_audio.std() / aud.std()
                            f_a = rescale_scale * f_a + (1 - rescale_scale)
                            aud = aud * f_a

                    pos_denoised = vid
                    pos_audio = aud

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
