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

import os
import weakref
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (
    Cosmos3DenoiseEngine,
    Cosmos3SoundSpec,
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

    @staticmethod
    def _flow_shift_for_resolution(height: int, width: int) -> float:
        """UniPC ``flow_shift`` for a given pixel resolution.

        Mirrors the framework's ``_RESOLUTION_SHIFT_DEFAULTS`` (8B VLM backbone,
        which Cosmos3-Nano uses): the shift is keyed by the named resolution
        bucket the (H, W) belongs to, regardless of task (T2V/I2V/T2I):

            "256" -> 3.0,  "480" -> 5.0,  "704"/"720"/"768" -> 10.0

        We invert the framework's ``{IMAGE,VIDEO}_RES_SIZE_INFO`` tables by the
        longest side: <=320 is the 256 bucket, 640-832 the 480 bucket, and
        960-1360 the 704/720/768 buckets.
        """
        long_side = max(int(height), int(width))
        if long_side <= 480:
            return 3.0
        if long_side <= 896:
            return 5.0
        return 10.0

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

        # Resolution-based flow_shift, set on the owning pipeline (rebuilds the
        # scheduler). The framework picks the UniPC shift purely from the named
        # resolution bucket (``_RESOLUTION_SHIFT_DEFAULTS``), NOT from the task,
        # so T2V/I2V/T2I at the same resolution share a shift.
        pipe = self.pipeline() if self.pipeline is not None else None
        flow_shift = self._flow_shift_for_resolution(height, width)
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

        # ---- t2vs: jointly generate sound (combined [vision | sound] latent) ----
        # Mirrors the framework: a placeholder audio sized to the video duration
        # sets the sound latent length; sound shares the denoise/CFG with vision.
        with_audio = is_video and os.environ.get("COSMOS3_T2VS", "") not in ("", "0")
        sound_specs = None
        sound_fps_per_item = None
        sound_vae = None
        sound_shape: tuple[int, int] | None = None
        if with_audio:
            sound_vae = self._get_sound_vae(pipe, device, dtype)
            sound_dim = int(arch.sound_dim)
            sound_latent_fps = float(arch.sound_latent_fps)
            # Framework ``create_placeholder_audio`` + ``get_latent_num_samples``.
            num_audio_samples = int(num_frames / fps * sound_vae.sample_rate)
            sound_latent_t = max(1, sound_vae.get_latent_num_samples(num_audio_samples))
            sound_shape = (sound_dim, sound_latent_t)
            sound_noise = randn_tensor((sound_dim, sound_latent_t), generator=generator, device=device,
                                       dtype=dtype).float()
            flat_latent = torch.cat([flat_latent, sound_noise.reshape(-1)])
            sound_specs = [Cosmos3SoundSpec(shape=sound_shape, condition_frame_indexes=[], fps=sound_latent_fps)]
            sound_fps_per_item = [sound_latent_fps] if bool(arch.enable_fps_modulation) else None

        final_flat = engine.denoise(
            flat_latent=flat_latent,
            timesteps=timesteps,
            guidance=guidance,
            specs=[spec],
            cond_token_ids=cond_ids,
            uncond_token_ids=uncond_ids,
            fps_per_item=fps_per_item,
            progress_bar=lambda it: tqdm(it, desc="Cosmos3 denoising"),
            sound_specs=sound_specs,
            sound_fps_per_item=sound_fps_per_item,
        )

        # ---- Decode vision: [C, T, H, W] -> pixels [B, 3, T, H, W] in [0, 1] ----
        vision_flat = final_flat[:spec.numel]
        result_latent = vision_flat.reshape(latent_shape).unsqueeze(0).to(device=device, dtype=dtype)
        decoded = cosmos3_vae_decode(self.vae, result_latent, norm)  # [B, 3, T, H, W] in [-1, 1]
        video = ((1.0 + decoded) / 2.0).clamp(0.0, 1.0)

        batch.latents = result_latent
        batch.output = video

        # ---- Decode sound: AVAE latent [C, T] -> waveform [C, N] in [-1, 1] ----
        if with_audio and sound_vae is not None and sound_shape is not None:
            sound_latent = final_flat[spec.numel:].reshape(sound_shape).unsqueeze(0).to(device=device, dtype=dtype)
            waveform = sound_vae.decode(sound_latent)  # [1, C_audio, N]
            batch.extra["audio"] = waveform[0].detach().float().cpu()  # [C_audio, N]
            batch.extra["audio_sample_rate"] = int(sound_vae.sample_rate)
        return batch

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _resize_and_center_crop(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Aspect-ratio-preserving resize + center crop, matching the framework
        (``cosmos_framework.inference.vision._resize_and_center_crop``)."""
        import math

        import torchvision.transforms.functional as TF
        orig_h, orig_w = img.shape[-2], img.shape[-1]
        scaling_ratio = max(target_w / orig_w, target_h / orig_h)
        resize_h = int(math.ceil(scaling_ratio * orig_h))
        resize_w = int(math.ceil(scaling_ratio * orig_w))
        img = TF.resize(img, [resize_h, resize_w])
        return TF.center_crop(img, [target_h, target_w])

    @staticmethod
    def _get_sound_vae(pipe: Any, device: torch.device, dtype: torch.dtype) -> Any:
        """Lazily load + cache the Cosmos3 sound AVAE decoder from the checkpoint.

        The video path does not load ``sound_tokenizer``; t2vs needs only its
        decoder, so we load it on first use from ``<model_path>/sound_tokenizer``.
        """
        cached = getattr(pipe, "_sound_vae", None) if pipe is not None else None
        if cached is not None:
            return cached
        from fastvideo.models.audio.cosmos3_avae import Cosmos3SoundVAE
        model_path = pipe.model_path
        sound_dir = os.path.join(model_path, "sound_tokenizer")
        sound_vae = Cosmos3SoundVAE.from_pretrained(sound_dir, torch_dtype=dtype).to(device)
        if pipe is not None:
            pipe._sound_vae = sound_vae
        return sound_vae

    @classmethod
    def _image_to_video_tensor(
        cls,
        image: Any,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build the I2V conditioning pixel video ``[1, 3, T, H, W]`` in [-1, 1].

        Faithful to the framework (``cosmos_framework.inference.vision``):
        ``load_conditioning_image`` (aspect-preserving resize + center crop +
        uint8 quantization, then ``/127.5 - 1``) followed by
        ``build_conditioned_video_batch``, which fills frame 0 with the image and
        **repeats the last conditioning frame** for the rest of the clip (a static
        video), NOT zeros. The whole clip is VAE-encoded by the caller; only the
        latent condition frame(s) are kept clean by the condition mask, but the
        VAE is temporal, so the repeated (not zeroed) frames change the condition
        latent — zero-filling here produces a wrong conditioning latent.
        """
        import numpy as np

        if hasattr(image, "convert"):  # PIL.Image: framework-exact preprocessing.
            arr = np.array(image.convert("RGB"))
            img = torch.from_numpy(arr).permute(2, 0, 1).float()  # [3, H, W] in [0, 255]
            # Resize + center crop + uint8 quantization, then -> [-1, 1]
            # (load_conditioning_image / load_conditioning_image_pixels).
            img = cls._resize_and_center_crop(img.unsqueeze(0), height, width).squeeze(0)
            img = img.round().clamp(0, 255) / 127.5 - 1.0  # [3, H, W] in [-1, 1]
        elif isinstance(image, torch.Tensor):  # already-preprocessed conditioning frame.
            img = image.float()
            if img.dim() == 5:  # [B,3,T,H,W]
                img = img[0]
            if img.dim() == 4:  # [3,T,H,W] or [B,3,H,W] -> first frame
                img = img[:, 0]
            if img.max() > 1.5:  # [0, 255] -> [-1, 1]; otherwise assume already [-1, 1].
                img = img / 127.5 - 1.0
            if img.shape[-2:] != (height, width):
                img = cls._resize_and_center_crop(img.unsqueeze(0), height, width).squeeze(0)
        else:
            raise TypeError(f"Unsupported conditioning image type: {type(image)}")

        # Static-repeat video (build_conditioned_video_batch: frame 0 = image,
        # remaining frames repeat the last conditioning frame). The whole clip is
        # VAE-encoded by the caller; only the latent condition frame(s) are kept
        # clean by the condition mask, but the VAE is temporal, so the repeated
        # (not zeroed) frames change the condition latent — zero-filling here
        # produces a wrong conditioning latent.
        img = img.to(device=device, dtype=dtype)
        video = img.unsqueeze(0).unsqueeze(2).expand(1, 3, num_frames, height, width)
        return video.contiguous()
