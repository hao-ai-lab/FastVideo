# SPDX-License-Identifier: Apache-2.0
"""Stable Audio latent preparation: random Gaussian latent + 1D RoPE."""
from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


def _get_1d_rotary_pos_embed(dim: int,
                             seq_len: int,
                             theta: float = 10000.0,
                             dtype: torch.dtype = torch.float32,
                             device: torch.device | None = None):
    """Mirrors `diffusers.models.embeddings.get_1d_rotary_pos_embed` with
    ``use_real=True, repeat_interleave_real=False`` (the Stable Audio
    branch). The half-frequency vector gets concatenated with itself so
    the returned cos/sin span the full attention head dim.
    """
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=dtype, device=device) / dim))
    t = torch.arange(seq_len, dtype=dtype, device=device)
    freqs = torch.outer(t, freqs)  # [S, D/2]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
    return cos, sin


class StableAudioLatentPreparationStage(PipelineStage):
    """Build the noisy latent + RoPE freqs for the denoise loop."""

    def __init__(
            self,
            transformer,
            scheduler,
            sample_size: int = 1024,
            num_channels_vae: int = 64,
            rotary_embed_dim: int = 32,  # attention_head_dim // 2 by default for SA
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.sample_size = sample_size
        self.num_channels_vae = num_channels_vae
        self.rotary_embed_dim = rotary_embed_dim

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        device = batch.prompt_embeds[0].device if batch.prompt_embeds else torch.device("cuda")
        # batch.extra was populated by the conditioning stage.
        text_audio = batch.extra["text_audio_duration_embeds"]
        audio_dur = batch.extra["audio_duration_embeds"]
        do_cfg = batch.extra.get("do_cfg", False)

        # batch_size in the latent shape = effective batch (post-CFG, since
        # the DiT will receive [uncond; cond] concatenated). But the
        # latent itself is single-batch — the CFG concat happens inside
        # the denoise loop's `latent_model_input = cat([latents] * 2)`.
        batch_size = (text_audio.shape[0] // 2) if do_cfg else text_audio.shape[0]
        dtype = text_audio.dtype

        # Sample noise. We keep the generator on `batch.extra` so the
        # denoising stage can seed `CosineDPMSolverMultistepScheduler`'s
        # internal `BrownianTreeNoiseSampler` from the same RNG (this is
        # what diffusers does via `prepare_extra_step_kwargs`).
        gen = torch.Generator(device=device)
        if batch.seed is not None:
            gen.manual_seed(int(batch.seed))
        latents = torch.randn(
            (batch_size, self.num_channels_vae, self.sample_size),
            generator=gen,
            device=device,
            dtype=dtype,
        )
        # Scheduler-noise scaling.
        latents = latents * self.scheduler.init_noise_sigma

        # 1D rotary embedding for the DiT (length = latent_seq +
        # global_states_seq). audio_duration_embeds has shape
        # [B, gs_len, dim], so total RoPE length is sample_size + gs_len.
        rope_len = self.sample_size + audio_dur.shape[1]
        cos, sin = _get_1d_rotary_pos_embed(
            self.rotary_embed_dim,
            rope_len,
            dtype=dtype,
            device=device,
        )

        batch.latents = latents
        batch.extra["rotary_embedding"] = (cos, sin)
        batch.extra["generator"] = gen
        return batch
