# SPDX-License-Identifier: Apache-2.0
"""Stable Audio CFG denoise loop (mirrors `StableAudioPipeline.__call__`
denoise loop, lines 712-746 of diffusers' pipeline_stable_audio.py).
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


class StableAudioDenoisingStage(PipelineStage):
    """Cosine-DPM++ denoising with text + duration CFG."""

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        pc = fastvideo_args.pipeline_config
        device = batch.latents.device
        text_audio = batch.extra["text_audio_duration_embeds"]
        audio_dur = batch.extra["audio_duration_embeds"]
        cos, sin = batch.extra["rotary_embedding"]
        do_cfg = batch.extra.get("do_cfg", False)
        guidance_scale = float(batch.guidance_scale or pc.guidance_scale)

        num_steps = batch.num_inference_steps
        # `set_timesteps` resets the scheduler's internal step state,
        # including the lazily-built `BrownianTreeNoiseSampler`. We also
        # null it out explicitly to make the seeded re-init below
        # deterministic across calls — without this the noise sampler
        # built in step 1 of the *previous* run lingers into step 1 of
        # this run if the scheduler instance is shared.
        self.scheduler.set_timesteps(num_steps, device=device)
        if hasattr(self.scheduler, "noise_sampler"):
            self.scheduler.noise_sampler = None
        timesteps = self.scheduler.timesteps
        # Match diffusers' `prepare_extra_step_kwargs(generator, eta)`:
        # CosineDPMSolverMultistepScheduler seeds its
        # BrownianTreeNoiseSampler from `generator.initial_seed()`.
        generator = batch.extra.get("generator") if batch.extra else None
        step_kwargs: dict = {}
        if generator is not None:
            step_kwargs["generator"] = generator

        latents = batch.latents
        disable_tqdm = not getattr(fastvideo_args, "log_level_progress", True)
        for i, t in enumerate(tqdm(timesteps, disable=disable_tqdm)):
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.transformer(
                latent_model_input,
                t.unsqueeze(0),
                encoder_hidden_states=text_audio,
                global_hidden_states=audio_dur,
                rotary_embedding=(cos, sin),
                return_dict=False,
            )[0]

            if do_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **step_kwargs).prev_sample

        batch.latents = latents
        return batch
