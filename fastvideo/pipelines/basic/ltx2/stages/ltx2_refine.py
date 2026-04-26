# SPDX-License-Identifier: Apache-2.0
"""LTX-2 refinement stages for 2x spatial upscaling + distilled denoising.

Public-side port of ``FastVideo-internal/.../stages/ltx2_refine.py``.
The three stages run between the stage-1 denoising pass and the stage-2
denoising pass:

* :class:`LTX2RefineInitStage` — halves the requested resolution so the
  first denoise runs at ½× and stashes the original target resolution
  on ``batch.extra`` so the upsample stage can recover it.
* :class:`LTX2UpsampleStage` — upsamples the stage-1 latents through
  the LTX-2 latent upsampler, optionally re-applies image conditioning,
  and mixes in fresh noise scaled by the stage-2 sigma so the next
  denoise has something to refine.
* :class:`LTX2RefineLoRAStage` — swaps in a refinement LoRA before the
  stage-2 denoise (no-op when the path is unset).

Behaviour matches the internal version 1:1 for the text-to-video path;
the i2v / continuation branches inside ``build_ltx2_image_conditioning``
defer to a NotImplementedError until the rest of the i2v conditioning
module is ported.
"""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.dits.ltx2 import AudioLatentShape, VideoLatentShape
from fastvideo.models.upsamplers import upsample_video
from fastvideo.pipelines.basic.ltx2.stages.ltx2_image_conditioning import (
    LTX2_CONTINUATION_STAGE1_LAST_LATENT_KEY,
    LTX2_VIDEO_CLEAN_LATENT_KEY,
    LTX2_VIDEO_DENOISE_MASK_KEY,
    apply_ltx2_gaussian_noiser,
    build_ltx2_image_conditioning,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)

# Reduced schedule for super-resolution stage 2 (subset of distilled values).
# Lifted verbatim from LTX-2 upstream
# (packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py).
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


class LTX2RefineInitStage(PipelineStage):
    """Switch the request to half resolution before the stage-1 denoise.

    Stashes the original target resolution on ``batch.extra`` so
    :class:`LTX2UpsampleStage` can recover it after stage 1 runs. When
    the refine path is disabled the stage is a no-op.
    """

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        if not fastvideo_args.ltx2_refine_enabled:
            return batch

        height = batch.height
        width = batch.width
        if height is None or width is None:
            raise ValueError("Height and width must be provided for LTX-2 refinement.")
        if isinstance(height, list) or isinstance(width, list):
            raise ValueError("LTX-2 refinement expects scalar height/width.")

        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("LTX-2 refinement requires even height/width so stage1 can be "
                             "half resolution.")

        spatial_ratio = (fastvideo_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio)
        stage1_height = height // 2
        stage1_width = width // 2
        if stage1_height % spatial_ratio != 0 or stage1_width % spatial_ratio != 0:
            raise ValueError(f"LTX-2 refinement requires height/width divisible by "
                             f"{2 * spatial_ratio} (got {height}x{width}).")

        batch.extra["ltx2_refine_target_height"] = height
        batch.extra["ltx2_refine_target_width"] = width
        batch.height = stage1_height
        batch.width = stage1_width

        logger.info(
            "[LTX2] Refinement enabled: stage1=%dx%d stage2=%dx%d",
            stage1_width,
            stage1_height,
            width,
            height,
        )
        return batch

    def verify_output(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        # Only meaningful checks live downstream of the upsample stage;
        # the init stage just rewrites height/width which the existing
        # latent-prep stage already validates.
        return VerificationResult()


class LTX2UpsampleStage(PipelineStage):
    """Upsample stage-1 latents to stage-2 resolution and add refine noise."""

    def __init__(
        self,
        *,
        upsampler: Any,
        vae: Any,
        transformer: Any | None = None,
        sigmas: list[float] | None = None,
        add_noise: bool = True,
    ) -> None:
        super().__init__()
        self.upsampler = upsampler
        self.vae = vae
        self.transformer = transformer
        self.sigmas = sigmas or STAGE_2_DISTILLED_SIGMA_VALUES
        self.add_noise = add_noise

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        if not fastvideo_args.ltx2_refine_enabled:
            return batch

        if batch.latents is None:
            raise ValueError("Latents must be available before LTX-2 upsample stage.")

        latents = batch.latents
        if batch.return_continuation_state:
            batch.extra[LTX2_CONTINUATION_STAGE1_LAST_LATENT_KEY] = (latents[:, :, -1:, :, :].detach().clone())

        orig_dtype = latents.dtype
        orig_device = latents.device
        if isinstance(self.upsampler, torch.nn.Module):
            first_param = next(self.upsampler.parameters(), None)
            if first_param is not None:
                if first_param.device != orig_device:
                    latents = latents.to(device=first_param.device)
                if first_param.dtype != latents.dtype:
                    latents = latents.to(dtype=first_param.dtype)
                if (latents.dtype != orig_dtype or latents.device != orig_device):
                    logger.info(
                        "[LTX2] Cast latents to %s on %s for upsampler.",
                        latents.dtype,
                        latents.device,
                    )
        target_height = batch.extra.get("ltx2_refine_target_height")
        target_width = batch.extra.get("ltx2_refine_target_width")
        if target_height is None or target_width is None:
            raise ValueError("Missing target resolution for LTX-2 refinement.")

        video_encoder = getattr(self.vae, "encoder", None)
        if video_encoder is None:
            raise ValueError("LTX-2 VAE encoder is required for latent upsampling.")

        upsampler_module = getattr(self.upsampler, "model", self.upsampler)
        latents = upsample_video(latents, video_encoder, upsampler_module)
        if latents.dtype != orig_dtype or latents.device != orig_device:
            latents = latents.to(device=orig_device, dtype=orig_dtype)

        image_conditioning = build_ltx2_image_conditioning(
            batch=batch,
            latents=latents,
            vae=self.vae,
            height=target_height,
            width=target_width,
            base_clean_latent=latents,
        )
        if image_conditioning is None:
            batch.extra.pop(LTX2_VIDEO_CLEAN_LATENT_KEY, None)
            batch.extra.pop(LTX2_VIDEO_DENOISE_MASK_KEY, None)
            clean_latents = latents
            denoise_mask = torch.ones(
                (latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]),
                dtype=torch.float32,
                device=latents.device,
            )
        else:
            clean_latents = image_conditioning.clean_latent
            denoise_mask = image_conditioning.denoise_mask
            batch.extra[LTX2_VIDEO_CLEAN_LATENT_KEY] = clean_latents
            batch.extra[LTX2_VIDEO_DENOISE_MASK_KEY] = denoise_mask
            logger.info(
                "[LTX2] Applied conditioning for stage-2: images=%d latent=%s.",
                len(image_conditioning.images),
                image_conditioning.latent_conditioned,
            )

        sigma0 = float(self.sigmas[0]) if self.sigmas else 1.0
        if self.add_noise:
            patchifier = getattr(self.transformer, "patchifier", None)
            patch_noise_shape: torch.Size | None = None
            video_shape: VideoLatentShape | None = None
            if patchifier is not None:
                video_shape = VideoLatentShape.from_torch_shape(latents.shape)
                patch_noise_shape = patchifier.patchify(latents).shape
            noise_path = fastvideo_args.ltx2_refine_noise_path
            noise = self._load_noise(
                noise_path,
                device=latents.device,
                dtype=latents.dtype,
                expected_shape=latents.shape,
                alternate_shape=patch_noise_shape,
            ) if noise_path else None
            if noise is None:
                noise = randn_tensor(
                    latents.shape,
                    generator=batch.generator,
                    device=latents.device,
                    dtype=latents.dtype,
                )
                if noise_path:
                    self._save_noise(noise_path, noise)
            elif (patchifier is not None and patch_noise_shape is not None and video_shape is not None
                  and noise.shape == patch_noise_shape):
                noise = patchifier.unpatchify(noise, video_shape)

            latents = apply_ltx2_gaussian_noiser(
                noise=noise,
                clean_latent=clean_latents,
                denoise_mask=denoise_mask,
                noise_scale=sigma0,
            )
        else:
            latents = clean_latents

        audio_latents = batch.extra.get("ltx2_audio_latents")
        if audio_latents is not None:
            audio_latents = audio_latents.to(device=latents.device)
            if self.add_noise:
                audio_patchifier = getattr(self.transformer, "audio_patchifier", None)
                if audio_patchifier is not None:
                    audio_shape = AudioLatentShape.from_torch_shape(audio_latents.shape)
                    audio_patch = audio_patchifier.patchify(audio_latents)
                    audio_noise_shape = audio_patch.shape
                    audio_noise_path = (fastvideo_args.ltx2_refine_audio_noise_path)
                    audio_noise = self._load_noise(
                        audio_noise_path,
                        device=audio_latents.device,
                        dtype=audio_latents.dtype,
                        expected_shape=audio_noise_shape,
                        alternate_shape=audio_latents.shape,
                    ) if audio_noise_path else None
                    if audio_noise is None:
                        audio_noise = randn_tensor(
                            audio_noise_shape,
                            generator=batch.generator,
                            device=audio_latents.device,
                            dtype=audio_latents.dtype,
                        )
                        if audio_noise_path:
                            self._save_noise(audio_noise_path, audio_noise)
                    elif audio_noise.shape == audio_latents.shape:
                        audio_noise = audio_patchifier.patchify(audio_noise)
                    audio_noised_patch = audio_noise * sigma0 + audio_patch * (1.0 - sigma0)
                    audio_latents = audio_patchifier.unpatchify(audio_noised_patch, audio_shape)
                else:
                    audio_noise_path = (fastvideo_args.ltx2_refine_audio_noise_path)
                    audio_noise = self._load_noise(
                        audio_noise_path,
                        device=audio_latents.device,
                        dtype=audio_latents.dtype,
                        expected_shape=audio_latents.shape,
                    ) if audio_noise_path else None
                    if audio_noise is None:
                        audio_noise = randn_tensor(
                            audio_latents.shape,
                            generator=batch.generator,
                            device=audio_latents.device,
                            dtype=audio_latents.dtype,
                        )
                        if audio_noise_path:
                            self._save_noise(audio_noise_path, audio_noise)
                    # Same noise mixing as video latents for the
                    # distilled refinement schedule.
                    audio_latents = audio_noise * sigma0 + audio_latents * (1.0 - sigma0)
            batch.extra["ltx2_audio_latents"] = audio_latents

        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        batch.height = target_height
        batch.width = target_width
        return batch

    def verify_input(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        result = VerificationResult()
        if fastvideo_args.ltx2_refine_enabled:
            result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result

    def _load_noise(
        self,
        noise_path: str | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
        expected_shape: torch.Size | tuple[int, ...],
        alternate_shape: torch.Size | tuple[int, ...] | None = None,
    ) -> torch.Tensor | None:
        if not noise_path:
            return None
        path = Path(noise_path)
        if not path.exists():
            return None
        payload = torch.load(path, map_location=device)
        if isinstance(payload, dict):
            noise = (payload.get("noise") or payload.get("latent_noise") or payload.get("latent")
                     or payload.get("video_noise"))
        else:
            noise = payload
        if not torch.is_tensor(noise):
            raise TypeError(f"Expected tensor noise in {path}")
        noise_shape = tuple(noise.shape)
        if (noise_shape != tuple(expected_shape)
                and (alternate_shape is None or noise_shape != tuple(alternate_shape))):
            raise ValueError(f"Noise shape mismatch for {path}: expected "
                             f"{tuple(expected_shape)}, got {noise_shape}")
        logger.info("[LTX2] Loaded refine noise from %s", path)
        return noise.to(device=device, dtype=dtype)

    def _save_noise(self, noise_path: str, noise: torch.Tensor) -> None:
        path = Path(noise_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        torch.save({"noise": noise.detach().cpu()}, path)
        logger.info("[LTX2] Saved refine noise to %s", path)


class LTX2RefineLoRAStage(PipelineStage):
    """Apply a refinement-specific LoRA before stage-2 denoising."""

    def __init__(
        self,
        *,
        pipeline: Any,
        lora_path: str | None,
        lora_nickname: str = "ltx2_refine",
    ) -> None:
        super().__init__()
        self._pipeline_ref = (weakref.ref(pipeline) if pipeline is not None else None)
        self._lora_path = lora_path
        self._lora_nickname = lora_nickname
        self._applied = False

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        if not fastvideo_args.ltx2_refine_enabled:
            return batch
        lora_path = fastvideo_args.ltx2_refine_lora_path or self._lora_path
        if not lora_path or self._applied:
            return batch

        pipeline = (self._pipeline_ref() if self._pipeline_ref is not None else None)
        if pipeline is None or not hasattr(pipeline, "set_lora_adapter"):
            raise ValueError("LTX2 refinement LoRA requested but pipeline does not "
                             "support LoRA adapters.")

        pipeline.set_lora_adapter(self._lora_nickname, lora_path)
        self._applied = True
        logger.info("[LTX2] Applied refinement LoRA from %s", lora_path)
        return batch


__all__ = [
    "LTX2RefineInitStage",
    "LTX2RefineLoRAStage",
    "LTX2UpsampleStage",
    "STAGE_2_DISTILLED_SIGMA_VALUES",
]
