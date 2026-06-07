# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 video denoising stage.

The Cosmos3 video path is monolithic by design: each CFG pass repacks the whole
text+vision sequence (the conditional pass carries prompt tokens, the
unconditional pass carries negative-prompt tokens), so the standard
encode/condition/denoise/decode stage split does not apply. This single stage
owns the full flow, delegating the framework-parity-tested math to
``fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline``:

  1. resolve mode (T2I / I2V / T2V) + per-mode defaults, set ``flow_shift``;
  2. tokenize the prompt + negative prompt with the Qwen2 chat template;
  3. VAE-encode the conditioning frame(s) for I2V / T2I (kept clean), build the
     initial noise (clean condition frames + pure noise elsewhere);
  4. run the UniPC denoise loop with sequential CFG
     (``Cosmos3DenoiseEngine.denoise``);
  5. VAE-decode + ``(1 + x) / 2`` clamp to [0, 1].

This mirrors the framework ``Cosmos3OmniDiffusersPipeline.__call__``.
"""
from __future__ import annotations

import weakref
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (
    Cosmos3DenoiseEngine,
    Cosmos3VisionSpec,
    _VaeNorm,
    cosmos3_special_tokens,
    cosmos3_tokenize_caption,
    cosmos3_vae_decode,
    cosmos3_vae_encode,
)
from fastvideo.pipelines.basic.cosmos3.presets import (
    COSMOS3_VIDEO_NEGATIVE_PROMPT, )
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)


class Cosmos3DenoisingStage(PipelineStage):
    """Full Cosmos3 video denoise: tokenize + encode + denoise + decode."""

    def __init__(self, *, transformer, scheduler, vae, tokenizer, pipeline=None) -> None:
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.tokenizer = tokenizer
        self.pipeline = weakref.ref(pipeline) if pipeline is not None else None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _latent_frames(num_frames: int, temporal_factor: int) -> int:
        return (int(num_frames) - 1) // int(temporal_factor) + 1

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        pipeline_config = fastvideo_args.pipeline_config
        arch = pipeline_config.dit_config.arch_config
        device = self.transformer.embed_tokens.weight.device
        dtype = self.transformer.embed_tokens.weight.dtype

        num_frames = int(batch.num_frames) if batch.num_frames is not None else 1
        height = int(batch.height)
        width = int(batch.width)
        fps = float(batch.fps) if batch.fps is not None else float(arch.base_fps)
        guidance = float(batch.guidance_scale)

        is_t2i = num_frames == 1 and batch.preprocessed_image is None and batch.pil_image is None
        is_i2v = (batch.preprocessed_image is not None or batch.pil_image is not None) and not is_t2i

        # Per-mode flow_shift, set on the owning pipeline (rebuilds scheduler).
        pipe = self.pipeline() if self.pipeline is not None else None
        engine_shift = float(getattr(pipe, "_engine_init_flow_shift", pipeline_config.flow_shift or 1.0))
        flow_shift = 3.0 if is_t2i else engine_shift
        if pipe is not None and hasattr(pipe, "_set_flow_shift"):
            pipe._set_flow_shift(flow_shift)
            scheduler = pipe.scheduler
        else:
            scheduler = self.scheduler

        # ---- Tokenize prompt + negative prompt ----
        prompt = batch.prompt if isinstance(batch.prompt, str) else (batch.prompt[0] if batch.prompt else "")
        negative_prompt = batch.negative_prompt
        if negative_prompt is None:
            negative_prompt = "" if is_t2i else COSMOS3_VIDEO_NEGATIVE_PROMPT
        if isinstance(negative_prompt, list):
            negative_prompt = negative_prompt[0] if negative_prompt else ""

        special_tokens = cosmos3_special_tokens(self.tokenizer)
        is_video = not is_t2i
        cond_ids = cosmos3_tokenize_caption(self.tokenizer, prompt, is_video=is_video, use_system_prompt=False)
        uncond_ids = cosmos3_tokenize_caption(self.tokenizer,
                                              negative_prompt,
                                              is_video=is_video,
                                              use_system_prompt=False)

        # ---- VAE normalization constants + geometry ----
        norm = _VaeNorm.from_vae(self.vae, dtype)
        temporal_factor = int(arch.temporal_compression_factor)
        spatial_factor = int(self.vae.config.scale_factor_spatial)
        latent_t = self._latent_frames(num_frames, temporal_factor)
        latent_h = height // spatial_factor
        latent_w = width // spatial_factor
        latent_channel = int(arch.latent_channel)
        latent_shape = (latent_channel, latent_t, latent_h, latent_w)

        generator = batch.generator
        if isinstance(generator, list):
            generator = generator[0] if generator else None

        # ---- Conditioning latent (I2V / T2I) + condition mask ----
        condition_frame_indexes: list[int] = []
        clean_latent: torch.Tensor | None = None
        if is_i2v or (is_t2i and (batch.preprocessed_image is not None or batch.pil_image is not None)):
            image = batch.preprocessed_image if batch.preprocessed_image is not None else batch.pil_image
            cond_pixels = self._image_to_video_tensor(image, num_frames, height, width, device, dtype)
            clean_latent = cosmos3_vae_encode(self.vae, cond_pixels, norm).squeeze(0).float()  # [C, T, H, W]
            condition_frame_indexes = [0]

        # ---- Initial noise (clean condition frames + pure noise elsewhere) ----
        pure_noise = randn_tensor(latent_shape, generator=generator, device=device, dtype=dtype).float()
        if clean_latent is not None:
            cond_mask = torch.zeros((latent_t, 1, 1), device=device, dtype=pure_noise.dtype)
            for idx in condition_frame_indexes:
                if 0 <= idx < latent_t:
                    cond_mask[idx, 0, 0] = 1.0
            clean = clean_latent.to(device=device, dtype=pure_noise.dtype)
            init_latent = cond_mask * clean + (1.0 - cond_mask) * pure_noise
        else:
            init_latent = pure_noise

        spec = Cosmos3VisionSpec(
            shape=latent_shape,
            condition_frame_indexes=condition_frame_indexes,
        )

        # ---- Scheduler timesteps ----
        scheduler.set_timesteps(int(batch.num_inference_steps), device=device)
        timesteps = scheduler.timesteps

        engine = Cosmos3DenoiseEngine(
            transformer=self.transformer,
            scheduler=scheduler,
            special_tokens=special_tokens,
            latent_patch_size=int(arch.latent_patch_size),
            temporal_modality_margin=int(arch.temporal_modality_margin),
            reset_spatial_ids=bool(arch.unified_3d_mrope_reset_spatial_ids),
            enable_fps_modulation=bool(arch.enable_fps_modulation),
            base_fps=float(arch.base_fps),
            temporal_compression_factor=temporal_factor,
            include_end_of_generation_token=False,
        )

        flat_latent = init_latent.reshape(-1)
        fps_per_item = [fps] if bool(arch.enable_fps_modulation) else None
        final_flat = engine.denoise(
            flat_latent=flat_latent,
            timesteps=timesteps,
            guidance=guidance,
            specs=[spec],
            cond_token_ids=cond_ids,
            uncond_token_ids=uncond_ids,
            fps_per_item=fps_per_item,
            progress_bar=lambda it: tqdm(it, desc="Cosmos3 denoising"),
        )

        # ---- Decode: [C, T, H, W] -> pixels [B, 3, T, H, W] in [0, 1] ----
        result_latent = final_flat.reshape(latent_shape).unsqueeze(0).to(device=device, dtype=dtype)
        decoded = cosmos3_vae_decode(self.vae, result_latent, norm)  # [B, 3, T, H, W] in [-1, 1]
        video = ((1.0 + decoded) / 2.0).clamp(0.0, 1.0)

        batch.latents = result_latent
        batch.output = video
        return batch

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _image_to_video_tensor(
        image: Any,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Coerce a conditioning image into a ``[1, 3, T, H, W]`` pixel tensor in [-1, 1].

        The first frame holds the (already preprocessed) conditioning image and
        the remaining frames are zero-filled; only frame 0 is kept clean by the
        condition mask, so the rest of the clip is denoised from noise.
        """
        import torchvision.transforms.functional as TF

        if isinstance(image, torch.Tensor):
            img = image.float()
            # Accept [B,3,T,H,W], [3,T,H,W], [3,H,W], or [B,3,H,W].
            if img.dim() == 5:
                img = img[0]
            if img.dim() == 4 and img.shape[1] == 1:
                img = img[:, 0]  # [3, H, W]
            elif img.dim() == 4:
                img = img[:, 0]  # take first frame -> [3, H, W]
            # Normalize to [-1, 1] if it looks like [0, 1] or [0, 255].
            if img.max() > 1.5:
                img = img / 127.5 - 1.0
        elif hasattr(image, "convert"):  # PIL.Image
            import numpy as np
            arr = np.array(image.convert("RGB"))
            img = torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0
        else:
            raise TypeError(f"Unsupported conditioning image type: {type(image)}")

        if img.shape[-2:] != (height, width):
            img = TF.resize(img.unsqueeze(0), [height, width]).squeeze(0)

        video = torch.zeros((1, 3, num_frames, height, width), device=device, dtype=dtype)
        video[0, :, 0] = img.to(device=device, dtype=dtype)
        return video
