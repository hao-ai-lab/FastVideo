# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 denoising stage using the native sigma schedule.
"""

from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations
import math
import os
from pathlib import Path

import torch
from tqdm.auto import tqdm

import fastvideo.envs as envs
from fastvideo.attention.backends.video_sparse_attn import (VideoSparseAttentionMetadataBuilder)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.basic.ltx2.stages.ltx2_image_conditioning import (LTX2_CONTINUATION_STAGE2_LAST_LATENT_KEY,
                                                                           LTX2_VIDEO_CLEAN_LATENT_KEY,
                                                                           LTX2_VIDEO_DENOISE_MASK_KEY,
                                                                           apply_ltx2_gaussian_noiser,
                                                                           post_process_ltx2_denoised)
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.logger import init_logger
from fastvideo.models.dits.ltx2 import (AudioLatentShape, DEFAULT_LTX2_AUDIO_CHANNELS, DEFAULT_LTX2_AUDIO_DOWNSAMPLE,
                                        DEFAULT_LTX2_AUDIO_HOP_LENGTH, DEFAULT_LTX2_AUDIO_MEL_BINS,
                                        DEFAULT_LTX2_AUDIO_SAMPLE_RATE, VideoLatentShape)
from fastvideo.utils import is_vsa_available

LTX2_AUDIO_CLEAN_LATENT_KEY = "ltx2_audio_clean_latent"
LTX2_AUDIO_DENOISE_MASK_KEY = "ltx2_audio_denoise_mask"

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096

# Official distilled sigma schedule (8 denoising steps)
# From LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

logger = init_logger(__name__)

try:
    vsa_available = is_vsa_available()
except ImportError:
    vsa_available = False


@contextmanager
def _nvtx_range(name: str):
    if os.getenv("FASTVIDEO_NVTX_PROFILE", "0") == "1" and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


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
    tokens = math.prod(latent.shape[2:]) if latent is not None else MAX_SHIFT_ANCHOR
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)

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


def _distilled_subset_sigmas(
    steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """Select distilled sigma values for a requested denoising step count.

    For steps < 8, choose indices that minimize the largest adjacent sigma
    drop while always preserving both endpoints.
    """
    max_steps = len(DISTILLED_SIGMA_VALUES) - 1
    if steps < 1 or steps > max_steps:
        raise ValueError(f"Distilled subset supports steps in [1, {max_steps}], got {steps}.")

    max_index = len(DISTILLED_SIGMA_VALUES) - 1
    if steps == max_steps:
        full_indices = list(range(max_index + 1))
        full_sigmas = torch.tensor(
            DISTILLED_SIGMA_VALUES,
            dtype=torch.float32,
            device=device,
        )
        return full_sigmas, full_indices

    interior_count = steps - 1
    best_key: tuple[float, float, float] | None = None
    best_indices: tuple[int, ...] | None = None

    for interior in combinations(range(1, max_index), interior_count):
        candidate = (0, *interior, max_index)
        gaps = [
            DISTILLED_SIGMA_VALUES[lo] - DISTILLED_SIGMA_VALUES[hi]
            for lo, hi in zip(candidate, candidate[1:], strict=False)
        ]
        key = (max(gaps), gaps[-1], sum(gap * gap for gap in gaps))
        if best_key is None or key < best_key:
            best_key = key
            best_indices = candidate

    if best_indices is None:
        raise RuntimeError("Failed to construct distilled subset schedule.")

    base_sigmas = torch.tensor(
        DISTILLED_SIGMA_VALUES,
        dtype=torch.float32,
        device=device,
    )
    subset_indices = torch.tensor(best_indices, dtype=torch.long, device=device)
    subset_sigmas = base_sigmas.index_select(0, subset_indices)
    return subset_sigmas, list(best_indices)


class LTX2DenoisingStage(PipelineStage):
    """Run the LTX-2 denoising loop over the sigma schedule."""

    def __init__(
        self,
        transformer,
        *,
        sigmas_override: list[float] | None = None,
        num_inference_steps_override: int | None = None,
        force_guidance_scale: float | None = None,
        initial_audio_latents_key: str | None = "ltx2_audio_latents",
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.sigmas_override = sigmas_override
        self.num_inference_steps_override = num_inference_steps_override
        self.force_guidance_scale = force_guidance_scale
        self.initial_audio_latents_key = initial_audio_latents_key

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        if batch.latents is None:
            raise ValueError("Latents must be provided before denoising.")

        latents = batch.latents
        video_clean_latent = batch.extra.get(LTX2_VIDEO_CLEAN_LATENT_KEY)
        video_denoise_mask = batch.extra.get(LTX2_VIDEO_DENOISE_MASK_KEY)
        if (video_clean_latent is None) != (video_denoise_mask is None):
            raise ValueError("LTX-2 i2v conditioning state is inconsistent: clean_latent/mask "
                             "must both be set or both be unset.")
        if video_clean_latent is not None and video_denoise_mask is not None:
            if not torch.is_tensor(video_clean_latent) or not torch.is_tensor(video_denoise_mask):
                raise TypeError("LTX-2 i2v conditioning tensors must be Tensors.")
            video_clean_latent = video_clean_latent.to(
                device=latents.device,
                dtype=latents.dtype,
            )
            video_denoise_mask = video_denoise_mask.to(
                device=latents.device,
                dtype=torch.float32,
            )

        prompt_embeds = batch.prompt_embeds[0]
        prompt_mask = None

        num_inference_steps = (self.num_inference_steps_override
                               if self.num_inference_steps_override is not None else batch.num_inference_steps)

        cfg_scale_video = batch.ltx2_cfg_scale_video
        cfg_scale_audio = batch.ltx2_cfg_scale_audio
        effective_guidance_scale = batch.guidance_scale
        use_cfg = batch.do_classifier_free_guidance
        if self.force_guidance_scale is not None:
            effective_guidance_scale = float(self.force_guidance_scale)
            cfg_scale_video = effective_guidance_scale
            cfg_scale_audio = effective_guidance_scale
            use_cfg = effective_guidance_scale > 1.0

        neg_prompt_embeds = None
        neg_prompt_mask = None
        if use_cfg:
            if batch.negative_prompt_embeds is not None and batch.negative_prompt_embeds:
                neg_prompt_embeds = batch.negative_prompt_embeds[0]
            else:
                logger.warning("[LTX2] CFG requested but negative_prompt_embeds missing; "
                               "falling back to no-CFG for this stage.")
                use_cfg = False

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
        autocast_enabled = (target_dtype != torch.float32) and not fastvideo_args.disable_autocast

        if self.sigmas_override is not None:
            sigmas = torch.tensor(
                self.sigmas_override,
                device=latents.device,
                dtype=torch.float32,
            )
            logger.info("[LTX2] Using override sigma schedule, %s", self.sigmas_override)
        else:
            # Use distilled hardcoded schedule (or subsets) when enabled.
            use_distilled_sigmas = os.getenv("LTX2_USE_DISTILLED_SIGMAS", "1") == "1"
            max_distilled_steps = len(DISTILLED_SIGMA_VALUES) - 1
            if use_distilled_sigmas and num_inference_steps <= max_distilled_steps:
                sigmas, distilled_indices = _distilled_subset_sigmas(
                    steps=num_inference_steps,
                    device=latents.device,
                )
                if num_inference_steps == max_distilled_steps:
                    logger.info("[LTX2] Using official distilled sigma schedule")
                else:
                    gaps = sigmas[:-1] - sigmas[1:]
                    logger.info(
                        "[LTX2] Using distilled sigma subset for %d steps "
                        "(indices=%s max_gap=%.6f tail_gap=%.6f)",
                        num_inference_steps,
                        distilled_indices,
                        float(gaps.max().item()),
                        float(gaps[-1].item()),
                    )
            else:
                sigmas = _ltx2_sigmas(
                    steps=num_inference_steps,
                    latent=None,
                    device=latents.device,
                )
                logger.info("[LTX2] Using computed sigma schedule, "
                            "num_inference_steps=%s", num_inference_steps)
        if hasattr(self.transformer, "patchifier"):
            video_shape = VideoLatentShape.from_torch_shape(latents.shape)
            token_count = self.transformer.patchifier.get_token_count(video_shape)
        else:
            token_count = 1

        timestep_template = torch.ones(
            (latents.shape[0], token_count, 1),
            device=latents.device,
            dtype=torch.float32,
        )
        if video_denoise_mask is not None:
            patchifier = getattr(self.transformer, "patchifier", None)
            if patchifier is not None:
                mask_patch = patchifier.patchify(video_denoise_mask.to(dtype=latents.dtype))
                timestep_template = mask_patch.mean(dim=-1).to(torch.float32)
            else:
                flat_mask = video_denoise_mask.reshape(video_denoise_mask.shape[0], -1).to(torch.float32)
                if flat_mask.shape[1] != token_count:
                    raise ValueError("LTX-2 i2v timestep mask token count mismatch: "
                                     f"expected {token_count}, got {flat_mask.shape[1]}")
                timestep_template = flat_mask
        audio_prompt_embeds = batch.extra.get("ltx2_audio_prompt_embeds")
        audio_neg_embeds = batch.extra.get("ltx2_audio_negative_embeds")
        audio_context_p = audio_prompt_embeds[0] if audio_prompt_embeds else None
        audio_context_n = audio_neg_embeds[0] if audio_neg_embeds else None
        audio_latents = batch.extra.get(self.initial_audio_latents_key)
        if isinstance(audio_latents, torch.Tensor):
            audio_latents = audio_latents.to(device=latents.device, dtype=latents.dtype)
        # Audio conditioning: mirror video i2v mask approach.
        audio_clean_latent = batch.extra.get(LTX2_AUDIO_CLEAN_LATENT_KEY)
        audio_denoise_mask = batch.extra.get(LTX2_AUDIO_DENOISE_MASK_KEY)
        if isinstance(audio_clean_latent, torch.Tensor):
            audio_clean_latent = audio_clean_latent.to(device=latents.device, dtype=latents.dtype)
        if isinstance(audio_denoise_mask, torch.Tensor):
            audio_denoise_mask = audio_denoise_mask.to(device=latents.device, dtype=torch.float32)
        audio_timestep_template = None
        if audio_latents is not None:
            audio_timestep_template = torch.ones(
                (latents.shape[0], audio_latents.shape[2], 1),
                device=latents.device,
                dtype=torch.float32,
            )
        if audio_context_p is not None and audio_latents is None:
            fps_value = batch.fps
            if isinstance(fps_value, list):
                fps_value = fps_value[0] if fps_value else None
            if fps_value is None:
                fps_value = 1.0
            # Allow audio to span a longer duration than video so
            # audio conditioning can overlap more without shrinking
            # the newly-generated audio region.
            audio_num_frames = batch.extra.get("audio_num_frames", batch.num_frames)
            duration = float(audio_num_frames) / float(fps_value)
            audio_shape = AudioLatentShape.from_duration(
                batch=latents.shape[0],
                duration=duration,
                channels=DEFAULT_LTX2_AUDIO_CHANNELS,
                mel_bins=DEFAULT_LTX2_AUDIO_MEL_BINS,
                sample_rate=DEFAULT_LTX2_AUDIO_SAMPLE_RATE,
                hop_length=DEFAULT_LTX2_AUDIO_HOP_LENGTH,
                audio_latent_downsample_factor=DEFAULT_LTX2_AUDIO_DOWNSAMPLE,
            )
            expected_shape = (
                audio_shape.batch,
                audio_shape.channels,
                audio_shape.frames,
                audio_shape.mel_bins,
            )
            audio_latent_path = fastvideo_args.ltx2_audio_latent_path
            audio_latents = self._load_audio_latents(
                audio_latent_path,
                device=latents.device,
                dtype=latents.dtype,
                expected_shape=expected_shape,
            ) if audio_latent_path else None
            if audio_latents is None:
                audio_generator = None
                if fastvideo_args.ltx2_initial_latent_path and batch.seed is not None:
                    audio_generator = torch.Generator(device=latents.device).manual_seed(batch.seed)
                elif batch.generator is not None:
                    audio_generator = (batch.generator[0] if isinstance(batch.generator, list) else batch.generator)
                if audio_generator is not None and audio_generator.device.type != latents.device.type:
                    if batch.seed is None:
                        audio_generator = torch.Generator(device=latents.device)
                    else:
                        audio_generator = torch.Generator(device=latents.device).manual_seed(batch.seed)
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
                    audio_latents = self.transformer.audio_patchifier.unpatchify(audio_latents_patch, audio_shape)
                else:
                    audio_latents = audio_latents_patch.view(
                        audio_shape.batch,
                        audio_shape.frames,
                        audio_shape.channels,
                        audio_shape.mel_bins,
                    ).permute(0, 2, 1, 3).contiguous()
                if audio_latent_path:
                    self._save_audio_latents(audio_latent_path, audio_latents)
            audio_timestep_template = torch.ones(
                (latents.shape[0], audio_shape.frames, 1),
                device=latents.device,
                dtype=torch.float32,
            )
        # Apply audio conditioning mask (mirrors video i2v approach).
        if (audio_latents is not None and audio_clean_latent is not None and audio_denoise_mask is not None):
            audio_T = audio_latents.shape[2]
            cond_T = audio_clean_latent.shape[2]
            if audio_T != cond_T:
                logger.warning(
                    "[LTX2] Audio conditioning T mismatch: "
                    "latents=%d, clean=%d; skipping audio "
                    "conditioning.", audio_T, cond_T)
                audio_clean_latent = None
                audio_denoise_mask = None
            else:
                # audio_denoise_mask: [B, 1, T, 1] → [B, T]
                # for timestep template.
                audio_timestep_template = (audio_denoise_mask[:, 0, :, 0])
                audio_latents = apply_ltx2_gaussian_noiser(
                    noise=audio_latents,
                    clean_latent=audio_clean_latent,
                    denoise_mask=audio_denoise_mask,
                    noise_scale=1.0,
                )

        # Video position offset: when audio is longer than video for
        # conditioning, shift video RoPE positions forward so the
        # audio prefix sits at t>=0 and video aligns with the later
        # portion of audio.
        video_position_offset_sec = float(batch.extra.get("video_position_offset_sec", 0.0))

        # Multi-modal CFG parameters (per-stream scales).
        modality_scale_video = batch.ltx2_modality_scale_video
        modality_scale_audio = batch.ltx2_modality_scale_audio
        rescale_scale = batch.ltx2_rescale_scale
        # STG (Spatio-Temporal Guidance) parameters.
        stg_scale_video = batch.ltx2_stg_scale_video
        stg_scale_audio = batch.ltx2_stg_scale_audio
        stg_blocks_video = batch.ltx2_stg_blocks_video
        stg_blocks_audio = batch.ltx2_stg_blocks_audio
        do_stg_video = not math.isclose(float(stg_scale_video), 0.0)
        do_stg_audio = not math.isclose(float(stg_scale_audio), 0.0)
        do_stg = do_stg_video or do_stg_audio
        do_cfg_text = use_cfg and (cfg_scale_video != 1.0 or cfg_scale_audio != 1.0)
        do_modality_video = not math.isclose(float(modality_scale_video), 1.0)
        do_modality_audio = not math.isclose(float(modality_scale_audio), 1.0)
        do_mod = do_modality_video or do_modality_audio
        do_guidance = do_cfg_text or do_mod or do_stg

        if do_cfg_text and neg_prompt_embeds is None:
            raise ValueError("LTX-2 text CFG is enabled "
                             "(ltx2_cfg_scale_video/audio != 1.0), "
                             "but negative prompt embeddings are missing")

        logger.info(
            "[LTX2] Denoising start: steps=%d dtype=%s "
            "cfg=%.1f "
            "cfg_video=%.1f cfg_audio=%.1f mod_video=%.1f "
            "mod_audio=%.1f rescale=%.2f "
            "stg_video=%.1f stg_audio=%.1f "
            "sigmas_shape=%s latents_shape=%s",
            num_inference_steps,
            target_dtype,
            effective_guidance_scale,
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
        # Hint runtime FP4 layer gating (single shared transformer path):
        # stage-1 denoising uses "base", stage-2 refine uses "refine".
        batch.extra["ltx2_fp4_stage_profile"] = ("refine" if self.sigmas_override is not None else "base")
        attention_backend = os.getenv("FASTVIDEO_ATTENTION_BACKEND", envs.FASTVIDEO_ATTENTION_BACKEND)
        wants_vsa_metadata = attention_backend in (
            "VIDEO_SPARSE_ATTN",
            "SAGE_ATTN_THREE",
        )
        # VIDEO_SPARSE_ATTN requires the fastvideo-kernel VSA op.
        # SAGE_ATTN_THREE VSA+QAT only needs VSA metadata and its own kernel path.
        use_vsa = wants_vsa_metadata and (attention_backend != "VIDEO_SPARSE_ATTN" or vsa_available)
        if attention_backend == "VIDEO_SPARSE_ATTN" and not vsa_available:
            logger.warning("FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN but VSA kernel "
                           "is unavailable; disabling VSA metadata for this run.")
        vsa_metadata_builder = (VideoSparseAttentionMetadataBuilder() if use_vsa else None)

        for step_index in tqdm(range(len(sigmas) - 1)):
            sigma = sigmas[step_index]
            sigma_next = sigmas[step_index + 1]
            timestep = timestep_template * sigma
            audio_timestep = (audio_timestep_template * sigma if audio_timestep_template is not None else None)
            latent_model_input = latents.to(target_dtype)
            attn_metadata = None
            if vsa_metadata_builder is not None:
                attn_metadata = vsa_metadata_builder.build(
                    current_timestep=step_index,
                    raw_latent_shape=latents.shape[2:5],
                    patch_size=fastvideo_args.pipeline_config.dit_config.patch_size,
                    VSA_sparsity=fastvideo_args.VSA_sparsity,
                    device=latents.device,
                )

            with torch.autocast(
                    device_type="cuda",
                    dtype=target_dtype,
                    enabled=autocast_enabled,
            ), set_forward_context(
                    current_timestep=sigma,
                    attn_metadata=attn_metadata,
                    forward_batch=batch,
            ):
                # Pass 1: Full conditioning (text + cross-modal)
                with _nvtx_range("ltx2.denoise.pass.pos"):
                    pos_outputs = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_mask,
                        timestep=timestep,
                        audio_hidden_states=audio_latents,
                        audio_encoder_hidden_states=audio_context_p,
                        audio_timestep=audio_timestep,
                        video_position_offset_sec=video_position_offset_sec,
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
                        with _nvtx_range("ltx2.denoise.pass.neg"):
                            neg_outputs = self.transformer(
                                hidden_states=latent_model_input,
                                encoder_hidden_states=neg_prompt_embeds,
                                encoder_attention_mask=neg_prompt_mask,
                                timestep=timestep,
                                audio_hidden_states=audio_latents,
                                audio_encoder_hidden_states=audio_context_n,
                                audio_timestep=audio_timestep,
                                video_position_offset_sec=video_position_offset_sec,
                            )
                        if isinstance(neg_outputs, tuple):
                            neg_denoised, neg_audio = neg_outputs
                        else:
                            neg_denoised = neg_outputs
                            neg_audio = None

                    # Pass 3: Modality-isolated (skip cross-modal attn)
                    if do_mod:
                        with _nvtx_range("ltx2.denoise.pass.modality"):
                            mod_outputs = self.transformer(
                                hidden_states=latent_model_input,
                                encoder_hidden_states=prompt_embeds,
                                encoder_attention_mask=prompt_mask,
                                timestep=timestep,
                                audio_hidden_states=audio_latents,
                                audio_encoder_hidden_states=audio_context_p,
                                audio_timestep=audio_timestep,
                                skip_cross_modal_attn=True,
                                video_position_offset_sec=video_position_offset_sec,
                            )
                        if isinstance(mod_outputs, tuple):
                            mod_denoised, mod_audio = mod_outputs
                        else:
                            mod_denoised = mod_outputs
                            mod_audio = None

                    # Pass 4: STG perturbed (skip self-attn in specified blocks)
                    if do_stg:
                        with _nvtx_range("ltx2.denoise.pass.stg"):
                            ptb_outputs = self.transformer(
                                hidden_states=latent_model_input,
                                encoder_hidden_states=prompt_embeds,
                                encoder_attention_mask=prompt_mask,
                                timestep=timestep,
                                audio_hidden_states=audio_latents,
                                audio_encoder_hidden_states=audio_context_p,
                                audio_timestep=audio_timestep,
                                skip_video_self_attn_blocks=(stg_blocks_video if do_stg_video else None),
                                skip_audio_self_attn_blocks=(stg_blocks_audio if do_stg_audio else None),
                                video_position_offset_sec=video_position_offset_sec,
                            )
                        if isinstance(ptb_outputs, tuple):
                            ptb_denoised, ptb_audio = ptb_outputs
                        else:
                            ptb_denoised = ptb_outputs
                            ptb_audio = None

                    # Multi-modal guidance formula per stream.
                    vid = (pos_denoised + (cfg_scale_video - 1) * (pos_denoised - neg_denoised) +
                           (modality_scale_video - 1) * (pos_denoised - mod_denoised) + stg_scale_video *
                           (pos_denoised - ptb_denoised))
                    aud = None
                    if pos_audio is not None:
                        aud = (pos_audio + (cfg_scale_audio - 1) * (pos_audio - neg_audio) +
                               (modality_scale_audio - 1) * (pos_audio - mod_audio) + stg_scale_audio *
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

            if video_clean_latent is not None and video_denoise_mask is not None:
                pos_denoised = post_process_ltx2_denoised(
                    denoised=pos_denoised,
                    denoise_mask=video_denoise_mask,
                    clean_latent=video_clean_latent,
                )
            if (audio_clean_latent is not None and audio_denoise_mask is not None and pos_audio is not None):
                pos_audio = post_process_ltx2_denoised(
                    denoised=pos_audio,
                    denoise_mask=audio_denoise_mask,
                    clean_latent=audio_clean_latent,
                )

            sigma_value = sigma.to(torch.float32) if isinstance(sigma, torch.Tensor) else torch.tensor(
                float(sigma),
                device=latents.device,
                dtype=torch.float32,
            )
            dt = sigma_next - sigma
            with _nvtx_range("ltx2.denoise.scheduler_update"):
                velocity = ((latents.float() - pos_denoised.float()) / sigma_value).to(latents.dtype)
                latents = (latents.float() + velocity.float() * dt).to(latents.dtype)
                if pos_audio is not None and audio_latents is not None:
                    audio_velocity = ((audio_latents.float() - pos_audio.float()) / sigma_value).to(audio_latents.dtype)
                    audio_latents = (audio_latents.float() + audio_velocity.float() * dt).to(audio_latents.dtype)

        batch.latents = latents
        if (batch.return_continuation_state and self.sigmas_override is not None):
            batch.extra[LTX2_CONTINUATION_STAGE2_LAST_LATENT_KEY] = (latents[:, :, -1:, :, :].detach().clone())
        batch.extra[self.initial_audio_latents_key] = audio_latents
        if self.initial_audio_latents_key != "ltx2_audio_latents":
            batch.extra["ltx2_audio_latents"] = audio_latents
        logger.info("[LTX2] Denoising done.")
        return batch

    def _load_audio_latents(
        self,
        latent_path: str | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
        expected_shape: tuple[int, ...],
    ) -> torch.Tensor | None:
        if not latent_path:
            return None
        path = Path(latent_path)
        if not path.exists():
            return None
        payload = torch.load(path, map_location=device)
        if isinstance(payload, dict):
            latent = (payload.get("audio_latent") or payload.get("latent") or payload.get("audio"))
        else:
            latent = payload
        if not torch.is_tensor(latent):
            raise TypeError(f"Expected tensor audio latent in {path}")
        if tuple(latent.shape) != tuple(expected_shape):
            raise ValueError(
                f"Audio latent shape mismatch for {path}: expected {expected_shape}, got {tuple(latent.shape)}")
        logger.info("[LTX2] Loaded audio latent from %s", path)
        return latent.to(device=device, dtype=dtype)

    def _save_audio_latents(self, latent_path: str, latents: torch.Tensor) -> None:
        path = Path(latent_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        torch.save({"audio_latent": latents.detach().cpu()}, path)
        logger.info("[LTX2] Saved audio latent to %s", path)

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("num_inference_steps", batch.num_inference_steps, V.positive_int)
        return result
