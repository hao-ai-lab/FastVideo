# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 refinement stages for 2x spatial upscaling + distilled denoising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import weakref
from diffusers.utils.torch_utils import randn_tensor
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.dits.ltx2 import AudioLatentShape, VideoLatentShape
from fastvideo.models.upsamplers import upsample_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)

# Reduced schedule for super-resolution stage 2 (subset of distilled values)
# From LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


class LTX2RefineInitStage(PipelineStage):
    """Prepare stage-1 resolution for LTX-2 2x spatial refinement."""

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
            raise ValueError(
                "Height and width must be provided for LTX-2 refinement.")
        if isinstance(height, list) or isinstance(width, list):
            raise ValueError("LTX-2 refinement expects scalar height/width.")

        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                "LTX-2 refinement requires even height/width so stage1 can be half resolution."
            )

        spatial_ratio = fastvideo_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        stage1_height = height // 2
        stage1_width = width // 2
        if stage1_height % spatial_ratio != 0 or stage1_width % spatial_ratio != 0:
            raise ValueError(
                f"LTX-2 refinement requires height/width divisible by {2 * spatial_ratio} "
                f"(got {height}x{width}).")

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

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        # if fastvideo_args.ltx2_refine_enabled:
        #     result.add_check(
        #         "height", batch.height, [V.is_int, V.positive_int])
        #     result.add_check(
        #         "width", batch.width, [V.is_int, V.positive_int])
        return result


class LTX2UpsampleStage(PipelineStage):
    """Upsample stage-1 latents to stage-2 resolution and add refinement noise."""

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
            raise ValueError(
                "Latents must be available before LTX-2 upsample stage.")

        # Ensure upsampler matches latent dtype/device to avoid conv mismatch.
        if isinstance(self.upsampler, torch.nn.Module):
            target_dtype = batch.latents.dtype
            target_device = batch.latents.device
            first_param = next(self.upsampler.parameters(), None)
            if first_param is not None and (first_param.dtype != target_dtype
                                            or first_param.device
                                            != target_device):
                self.upsampler.to(device=target_device, dtype=target_dtype)
                logger.info(
                    "[LTX2] Cast upsampler to %s on %s to match latents.",
                    target_dtype,
                    target_device,
                )

        target_height = batch.extra.get("ltx2_refine_target_height")
        target_width = batch.extra.get("ltx2_refine_target_width")
        if target_height is None or target_width is None:
            raise ValueError("Missing target resolution for LTX-2 refinement.")

        video_encoder = getattr(self.vae, "encoder", None)
        if video_encoder is None:
            raise ValueError(
                "LTX-2 VAE encoder is required for latent upsampling.")

        upsampler_module = getattr(self.upsampler, "model", self.upsampler)
        latents = upsample_video(batch.latents, video_encoder, upsampler_module)

        sigma0 = float(self.sigmas[0]) if self.sigmas else 1.0
        if self.add_noise:
            patchifier = getattr(self.transformer, "patchifier", None)
            if patchifier is not None:
                video_shape = VideoLatentShape.from_torch_shape(latents.shape)
                latents_patch = patchifier.patchify(latents)
                noise_shape = latents_patch.shape
                noise_path = fastvideo_args.ltx2_refine_noise_path
                noise = self._load_noise(
                    noise_path,
                    device=latents.device,
                    dtype=latents.dtype,
                    expected_shape=noise_shape,
                    alternate_shape=latents.shape,
                ) if noise_path else None
                if noise is None:
                    noise = randn_tensor(
                        noise_shape,
                        generator=batch.generator,
                        device=latents.device,
                        dtype=latents.dtype,
                    )
                    if noise_path:
                        self._save_noise(noise_path, noise)
                elif noise.shape == latents.shape:
                    noise = patchifier.patchify(noise)
                noised_patch = noise * sigma0 + latents_patch * (1.0 - sigma0)
                latents = patchifier.unpatchify(noised_patch, video_shape)
            else:
                noise_path = fastvideo_args.ltx2_refine_noise_path
                noise = self._load_noise(
                    noise_path,
                    device=latents.device,
                    dtype=latents.dtype,
                    expected_shape=latents.shape,
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
                # Match LTX-2 GaussianNoiser: noise * sigma + latent * (1 - sigma).
                latents = noise * sigma0 + latents * (1.0 - sigma0)

        audio_latents = batch.extra.get("ltx2_audio_latents")
        if audio_latents is not None:
            audio_latents = audio_latents.to(device=latents.device)
            if self.add_noise:
                audio_patchifier = getattr(self.transformer, "audio_patchifier",
                                           None)
                if audio_patchifier is not None:
                    audio_shape = AudioLatentShape.from_torch_shape(
                        audio_latents.shape)
                    audio_patch = audio_patchifier.patchify(audio_latents)
                    audio_noise_shape = audio_patch.shape
                    audio_noise_path = fastvideo_args.ltx2_refine_audio_noise_path
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
                    audio_noised_patch = audio_noise * sigma0 + audio_patch * (
                        1.0 - sigma0)
                    audio_latents = audio_patchifier.unpatchify(
                        audio_noised_patch, audio_shape)
                else:
                    audio_noise_path = fastvideo_args.ltx2_refine_audio_noise_path
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
                    # Same noise mixing as video latents for distilled refinement.
                    audio_latents = audio_noise * sigma0 + audio_latents * (
                        1.0 - sigma0)
            batch.extra["ltx2_audio_latents"] = audio_latents

        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        batch.height = target_height
        batch.width = target_width
        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        if fastvideo_args.ltx2_refine_enabled:
            result.add_check("latents", batch.latents,
                             [V.is_tensor, V.with_dims(5)])
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
            noise = (payload.get("noise") or payload.get("latent_noise")
                     or payload.get("latent") or payload.get("video_noise"))
        else:
            noise = payload
        if not torch.is_tensor(noise):
            raise TypeError(f"Expected tensor noise in {path}")
        noise_shape = tuple(noise.shape)
        if noise_shape != tuple(expected_shape) and (alternate_shape is None
                                                     or noise_shape
                                                     != tuple(alternate_shape)):
            raise ValueError(
                f"Noise shape mismatch for {path}: expected {tuple(expected_shape)}, got {noise_shape}"
            )
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
        self._pipeline_ref = weakref.ref(
            pipeline) if pipeline is not None else None
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

        pipeline = self._pipeline_ref(
        ) if self._pipeline_ref is not None else None
        if pipeline is None or not hasattr(pipeline, "set_lora_adapter"):
            raise ValueError(
                "LTX2 refinement LoRA requested but pipeline does not support LoRA adapters."
            )

        pipeline.set_lora_adapter(self._lora_nickname, lora_path)
        self._applied = True
        logger.info("[LTX2] Applied refinement LoRA from %s", lora_path)
        return batch


__all__ = [
    "STAGE_2_DISTILLED_SIGMA_VALUES",
    "LTX2RefineInitStage",
    "LTX2UpsampleStage",
    "LTX2RefineLoRAStage",
]
