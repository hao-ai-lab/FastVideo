# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
# APG and frame sampling follow ByteDance/Bernini (Apache-2.0).
"""Bernini-R video conditioning and request-local UniPC/APG sampling."""

import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.bernini import pack_latents, unpack_latents
from fastvideo.models.vaes.common import DiagonalGaussianDistribution
from fastvideo.models.vision_utils import load_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.validators import V, VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE


def sample_frame_indices(total: int, source_fps: float, fps: float, limit: int) -> list[int]:
    """Official smart_video_nframes policy, including repeated final frames."""
    if (total < 1 or source_fps is None or fps is None or not math.isfinite(source_fps) or not math.isfinite(fps)
            or source_fps <= 0 or fps <= 0):
        raise ValueError("A nonempty video and positive source/target fps are required")
    if limit < 1 or (limit - 1) % 4:
        raise ValueError("The frame limit must be positive and 4*k+1")
    count = max(5, math.floor(total / source_fps * fps / 4) * 4 + 1)
    end = (total // 4 * 4 + 1) if source_fps == fps else total
    indices = torch.linspace(0, end - 1, count).round().long().tolist()
    return [min(i, total - 1) for i in indices[:limit]]


class BerniniInputStage(InputValidationStage):

    def forward(self, batch, fastvideo_args):
        if not isinstance(batch.video_path, str):
            raise ValueError("Bernini-R video editing requires one video_path")
        if batch.image_path is not None or batch.pil_image is not None:
            raise ValueError("This Bernini-R integration supports video editing without reference images")
        if batch.references:
            raise ValueError("Bernini-R conditioning uses video_path and text prompts")
        if batch.num_videos_per_prompt != 1 or isinstance(batch.prompt, list):
            raise ValueError("Bernini-R currently supports a single prompt and video per request")
        if batch.max_sequence_length is not None and (not isinstance(batch.max_sequence_length, int)
                                                      or not 1 <= batch.max_sequence_length <= 512):
            raise ValueError("Bernini-R max_sequence_length must be between 1 and 512")
        if not isinstance(batch.num_frames, int) or batch.num_frames < 1 or (batch.num_frames - 1) % 4:
            raise ValueError("Bernini-R num_frames must be 4*k+1")
        if any(not isinstance(n, int) or n <= 0 or n % 16 for n in (batch.height, batch.width)):
            raise ValueError("Bernini-R height and width must be positive and divisible by 16")
        # APG uses both x-predictions even at scale=1 (the delta may be clipped).
        batch.do_classifier_free_guidance = True
        path, batch.video_path = batch.video_path, None
        try:
            super().forward(batch, fastvideo_args)
        finally:
            batch.video_path = path
        images, source_fps = load_video(path, return_fps=True)
        indices = sample_frame_indices(len(images), source_fps, batch.fps, batch.num_frames)
        frames = []
        for i in indices:
            image = images[i].convert("RGB")
            if image.size != (batch.width, batch.height):
                image = TF.resize(image, [batch.height, batch.width], InterpolationMode.BICUBIC, antialias=True)
            frames.append(TF.to_tensor(image).sub_(0.5).div_(0.5))
        batch.video_latent = torch.stack(frames, dim=1).unsqueeze(0)
        batch.num_frames = len(frames)
        return batch

    def verify_output(self, batch, fastvideo_args):
        result = VerificationResult()
        result.add_check("video_pixels", batch.video_latent, [V.is_tensor, V.with_dims(5)])
        return result


class BerniniVideoEncodingStage(PipelineStage):
    performance_component_metric = "vae_encode_time_s"

    def __init__(self, vae):
        self.vae = vae

    @torch.no_grad()
    def forward(self, batch, fastvideo_args):
        device = get_local_torch_device()
        self.vae.to(device)
        try:
            # Official conditioning uses the posterior mode, never an RNG sample.
            latent = self.vae.encode(batch.video_latent.to(device=device, dtype=torch.float32))
            if isinstance(latent, DiagonalGaussianDistribution):
                latent = latent.mean
            elif hasattr(latent, "latent_dist"):
                latent = latent.latent_dist.mode()
            mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(self.vae.config.latents_std, device=device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
            batch.video_latent = (latent - mean) / std
        finally:
            if fastvideo_args.vae_cpu_offload:
                self.vae.to("cpu")
        return batch

    def verify_input(self, batch, fastvideo_args):
        result = VerificationResult()
        result.add_check("video_pixels", batch.video_latent, [V.is_tensor, V.with_dims(5)])
        return result

    def verify_output(self, batch, fastvideo_args):
        result = VerificationResult()
        result.add_check("source_latents", batch.video_latent, [V.is_tensor, V.with_dims(5)])
        return result


def projected_guidance(cond: torch.Tensor,
                       uncond: torch.Tensor,
                       sample: torch.Tensor,
                       sigma: torch.Tensor,
                       scale: float,
                       eta: float = 0.5) -> torch.Tensor:
    """Official CLI v2v_apg recipe: eta=0.5, threshold=50, momentum=0.

    Normalization is per latent frame over C,H,W, not over the whole clip.
    The float64 projection and its float32 round trip are part of the oracle.
    """
    x_cond, x_uncond = sample - sigma * cond, sample - sigma * uncond
    diff = x_cond - x_uncond
    norm = diff.norm(p=2, dim=(-1, -2, -4), keepdim=True)
    diff = diff * torch.minimum(torch.ones_like(diff), 50.0 / norm)
    ref = F.normalize(x_cond.double(), dim=(-1, -2, -4))
    parallel = (diff.double() * ref).sum(dim=(-1, -2, -4), keepdim=True) * ref
    orthogonal = diff.double() - parallel
    x_guided = x_uncond + scale * (orthogonal.to(diff.dtype) + eta * parallel.to(diff.dtype))
    return (sample - x_guided) / sigma


class BerniniDenoisingStage(PipelineStage):
    performance_component_metric = "dit_time_s"

    def __init__(self, transformer, scheduler):
        self.transformer = transformer
        self.scheduler = scheduler

    @torch.no_grad()
    def forward(self, batch, fastvideo_args):
        self.transformer.to(get_local_torch_device())
        try:
            return self._denoise(batch, fastvideo_args)
        finally:
            if fastvideo_args.dit_cpu_offload:
                self.transformer.to("cpu")

    def _denoise(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        device = get_local_torch_device()
        # The first parameter is the FP32-pinned scale/shift table, not the
        # DiT's working dtype. Read the configured compute precision instead.
        dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        source = batch.video_latent.to(device=device, dtype=dtype)
        shape = tuple(source.shape)
        generators = batch.generator
        generator = generators[0] if isinstance(generators, list) else generators
        if batch.latents is None:
            latents = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(device)
        else:
            if tuple(batch.latents.shape) != shape:
                raise ValueError(f"Bernini-R initial latents must have source shape {shape}")
            latents = batch.latents.to(device=device, dtype=torch.float32)
        # Each request gets a fresh multistep history, including repeated calls.
        scheduler = type(self.scheduler).from_config(self.scheduler.config,
                                                     flow_shift=fastvideo_args.pipeline_config.flow_shift)
        scheduler.set_timesteps(batch.num_inference_steps, device=device)
        packed = pack_latents(latents)
        prompt = batch.prompt_embeds[0].to(device=device, dtype=dtype)
        if not batch.negative_prompt_embeds:
            raise ValueError("Bernini-R APG requires negative prompt embeddings, including at guidance_scale=1")
        negative = batch.negative_prompt_embeds[0].to(device=device, dtype=dtype)
        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Bernini-R denoising")):
            spatial = unpack_latents(packed, shape)
            with set_forward_context(current_timestep=int(t.item()), attn_metadata=None):
                cond = self.transformer(spatial.to(dtype), prompt, t.reshape(1), source_latents=source).float()
                uncond = self.transformer(spatial.to(dtype), negative, t.reshape(1), source_latents=source).float()
                cond = projected_guidance(cond, uncond, spatial, scheduler.sigmas[i], batch.guidance_scale)
            packed = scheduler.step(pack_latents(cond), t, packed, return_dict=False)[0]
        batch.latents = unpack_latents(packed, shape)
        batch.timesteps = scheduler.timesteps
        return batch

    def verify_input(self, batch, fastvideo_args):
        result = VerificationResult()
        result.add_check("source_latents", batch.video_latent, [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("negative_prompt_embeds", batch.negative_prompt_embeds,
                         lambda x: not batch.do_classifier_free_guidance or bool(x))
        return result

    def verify_output(self, batch, fastvideo_args):
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result
