# SPDX-License-Identifier: Apache-2.0
"""Stable Audio latent preparation — initial Gaussian noise + (optionally)
encoded `init_audio` / `inpaint_audio` references for A2A and inpainting.

Mirrors the upstream `generate_diffusion_cond` initial-noise step:

    torch.manual_seed(seed)
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

When the user supplies an `init_audio` clip (audio-to-audio variation) or
an `inpaint_audio` clip + binary `inpaint_mask` (RePaint-style inpainting),
this stage prepares the corresponding latent-space tensors and stashes
them on `batch.extra` for the denoising stage.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


class StableAudioLatentPreparationStage(PipelineStage):

    def __init__(self,
                 io_channels: int = 64,
                 sample_size: int = 2097152,
                 vae=None,
                 sample_rate: int = 44100,
                 audio_channels: int = 2) -> None:
        super().__init__()
        self.io_channels = io_channels
        # `sample_size` is the audio-domain length the official model was
        # trained for; latent length = sample_size // vae.hop_length
        # (= 2097152 / 2048 = 1024 for Stable Audio Open 1.0).
        self.sample_size = sample_size
        self.vae = vae  # used to encode init_audio / inpaint_audio
        self.sample_rate = sample_rate
        self.audio_channels = audio_channels

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        ext = batch.extra or {}
        device = ext["cross_attn_cond"].device
        latent_sample_size = self.sample_size // self._hop_length()

        seed = int(batch.seed) if batch.seed is not None else 0
        # Match upstream `generate_diffusion_cond`:
        #     torch.manual_seed(seed)
        #     noise = torch.randn([B, io_channels, sample_size], device=device)
        torch.manual_seed(seed)
        latents = torch.randn((1, self.io_channels, latent_sample_size), device=device)

        batch.latents = latents
        if batch.extra is None:
            batch.extra = {}

        # ---- A2A / inpainting reference encoding -----------------------
        init_audio = getattr(batch, "init_audio", None)
        inpaint_audio = getattr(batch, "inpaint_audio", None)
        inpaint_mask = getattr(batch, "inpaint_mask", None)

        # Loud-fail validation: silent fall-through to T2A would hide
        # caller bugs. Enforce that A2A / inpaint kwargs come in
        # complete sets.
        if inpaint_audio is not None and inpaint_mask is None:
            raise ValueError("Stable Audio inpainting requires both `inpaint_audio` and "
                             "`inpaint_mask` (1-D tensor in {0, 1} at the model sample rate, "
                             "1 = keep, 0 = regenerate). Got `inpaint_audio` without `inpaint_mask`.")
        if inpaint_mask is not None and inpaint_audio is None:
            raise ValueError("Stable Audio inpainting requires both `inpaint_audio` and "
                             "`inpaint_mask`. Got `inpaint_mask` without `inpaint_audio` — "
                             "did you mean to pass `init_audio` (audio-to-audio variation)?")
        if init_audio is not None and inpaint_audio is not None:
            raise ValueError("Stable Audio cannot do A2A variation and inpainting in the "
                             "same call. Pass either `init_audio` (variation) or "
                             "`inpaint_audio` + `inpaint_mask` (inpainting), not both.")

        if init_audio is not None:
            batch.extra["init_latent"] = self._encode_audio_reference(init_audio, device)
            # init_noise_level stays on `batch` directly; the denoising
            # stage reads `batch.init_noise_level` (parallel to how
            # `guidance_scale` is read).

        if inpaint_audio is not None and inpaint_mask is not None:
            batch.extra["inpaint_reference_latent"] = self._encode_audio_reference(inpaint_audio, device)
            batch.extra["inpaint_mask_latent"] = self._prepare_mask(inpaint_mask, latent_sample_size, device)
        return batch

    def _hop_length(self) -> int:
        """Resolve the VAE's audio-to-latent downsampling ratio."""
        v = self.vae
        if hasattr(v, "oobleck_vae"):
            return int(v.oobleck_vae.hop_length)
        return int(getattr(v, "hop_length", 2048))

    def _encode_audio_reference(self, audio: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Pad/truncate audio to `sample_size`, then encode via the VAE.

        Expects `audio` shape `[batch, channels, samples]` already at the
        model's sample rate. Caller is responsible for resampling.
        """
        assert self.vae is not None, "VAE required for init_audio / inpaint_audio encoding"
        audio = audio.to(device=device, dtype=torch.float32)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)
        # Match expected channel count (mono → repeat to stereo).
        if audio.shape[1] == 1 and self.audio_channels == 2:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] == 2 and self.audio_channels == 1:
            audio = audio.mean(dim=1, keepdim=True)
        # Pad/truncate to model sample_size.
        cur_len = audio.shape[-1]
        if cur_len < self.sample_size:
            audio = F.pad(audio, (0, self.sample_size - cur_len))
        elif cur_len > self.sample_size:
            audio = audio[..., :self.sample_size]
        # Stochastic posterior — matches upstream `vae_sample(mean, scale)`.
        # The global torch RNG state at this point is "post-`randn(noise)`",
        # same as upstream `generate_diffusion_cond`.
        from fastvideo.models.vaes.sa_audio import SAAudioVAEModel
        if isinstance(self.vae, SAAudioVAEModel):
            return self.vae.encode(audio, sample_posterior=True)
        # Bare OobleckVAE path (used by tests): returns a posterior with `.sample()`.
        return self.vae.encode(audio.to(next(self.vae.parameters()).dtype)).sample()

    def _prepare_mask(self, mask: torch.Tensor, latent_len: int, device: torch.device) -> torch.Tensor:
        """Resample a binary [0/1] mask in audio-sample space to latent space.

        Convention (matches upstream `generate_diffusion_cond_inpaint`):
        `mask == 1` means **keep the reference**, `mask == 0` means
        **regenerate**.
        """
        m = mask.to(device=device, dtype=torch.float32)
        if m.dim() == 1:
            m = m.unsqueeze(0)  # [1, samples]
        # Pad/truncate to model sample_size.
        cur_len = m.shape[-1]
        if cur_len < self.sample_size:
            m = F.pad(m, (0, self.sample_size - cur_len), value=0.0)
        elif cur_len > self.sample_size:
            m = m[..., :self.sample_size]
        # Resample to latent length via nearest neighbour (matches upstream's
        # `interpolate(..., mode='nearest')`).
        m = F.interpolate(m.unsqueeze(1), size=latent_len, mode="nearest")
        return m  # [1, 1, latent_len]
