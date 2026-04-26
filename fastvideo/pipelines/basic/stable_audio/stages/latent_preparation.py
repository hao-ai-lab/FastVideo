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

# Fixed downsampling ratio for Stable Audio Open 1.0 Oobleck VAE.
_DOWNSAMPLING_RATIO = 2048


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
        # trained for; latent length = sample_size // _DOWNSAMPLING_RATIO
        # (= 2097152 / 2048 = 1024).
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
        latent_sample_size = self.sample_size // _DOWNSAMPLING_RATIO

        seed = batch.seed if batch.seed is not None else 0
        # Match upstream `generate_diffusion_cond`:
        #     torch.manual_seed(seed)
        #     noise = torch.randn([B, io_channels, sample_size], device=device)
        torch.manual_seed(int(seed))
        latents = torch.randn((1, self.io_channels, latent_sample_size), device=device)

        batch.latents = latents
        if batch.extra is None:
            batch.extra = {}
        batch.extra["seed"] = int(seed)
        batch.extra["downsampling_ratio"] = _DOWNSAMPLING_RATIO
        batch.extra["sample_size"] = self.sample_size

        # ---- A2A / inpainting reference encoding -----------------------
        init_audio = getattr(batch, "init_audio", None)
        if init_audio is not None:
            init_latent = self._encode_audio_reference(init_audio, device)
            batch.extra["init_latent"] = init_latent
            batch.extra["init_noise_level"] = float(getattr(batch, "init_noise_level", None) or 1.0)

        inpaint_audio = getattr(batch, "inpaint_audio", None)
        inpaint_mask = getattr(batch, "inpaint_mask", None)
        if inpaint_audio is not None and inpaint_mask is not None:
            ref_latent = self._encode_audio_reference(inpaint_audio, device)
            mask_latent = self._prepare_mask(inpaint_mask, latent_sample_size, device)
            batch.extra["inpaint_reference_latent"] = ref_latent
            batch.extra["inpaint_mask_latent"] = mask_latent
        return batch

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
        # Upstream `pretransform.encode` calls `VAEBottleneck.encode` →
        # `vae_sample(mean, scale)` which adds `randn_like(mean) *
        # softplus(scale)+1e-4` (stochastic). To byte-match upstream we
        # need to sample from the posterior, not take .mode() — the
        # global torch RNG state at this point is "post-`randn(noise)`",
        # same as upstream.
        #
        # Bypass `SAAudioVAEModel.encode` (the legacy wrapper that
        # auto-takes .mode()) and go direct to the inner OobleckVAE.
        # `_move_to_input_device` triggers the lazy load + device move
        # so the inner module's params land on the right GPU before we
        # pull `next(inner.parameters()).dtype`.
        if hasattr(self.vae, "oobleck_vae"):
            mover = getattr(self.vae, "_move_to_input_device", None)
            if mover is not None:
                self.vae._move_to_input_device(self.vae.oobleck_vae, audio)
            inner = self.vae.oobleck_vae
        else:
            inner = self.vae
        posterior = inner.encode(audio.to(next(inner.parameters()).dtype))
        if hasattr(posterior, "sample") and not isinstance(posterior, torch.Tensor):
            return posterior.sample()
        return posterior

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
