# SPDX-License-Identifier: Apache-2.0
"""Stable Audio latent preparation.

Seeds + samples the initial Gaussian noise; encodes `init_audio` (A2A
variation) or `inpaint_audio` + `inpaint_mask` (RePaint inpainting) into
latent-space tensors on `batch.extra` for the denoising stage.
"""
from __future__ import annotations

import os

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
        # Audio-domain length the model was trained for; latent length
        # = sample_size // vae.hop_length (= 2097152 / 2048 = 1024).
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
        torch.manual_seed(seed)
        latents = torch.randn((1, self.io_channels, latent_sample_size), device=device)

        batch.latents = latents
        if batch.extra is None:
            batch.extra = {}

        init_audio = getattr(batch, "init_audio", None)
        inpaint_audio = getattr(batch, "inpaint_audio", None)
        inpaint_mask = getattr(batch, "inpaint_mask", None)

        # Loud-fail rather than silently falling through to T2A.
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
            # `init_noise_level` is read straight off `batch` by denoise.

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

    def _encode_audio_reference(self, audio, device: torch.device) -> torch.Tensor:
        """Pad/truncate to `sample_size` and encode via the VAE.

        `audio` may be a tensor (`[samples]`, `[C, samples]`, or
        `[B, C, samples]`) at the model's sample rate, or a path to any
        audio-bearing file (`.wav` / `.mp3` / `.mp4` / `.m4a` / `.flac`,
        ...) that PyAV can decode — we resample on load so callers don't
        have to.
        """
        assert self.vae is not None, "VAE required for init_audio / inpaint_audio encoding"
        if isinstance(audio, str | os.PathLike):
            audio = _decode_audio_file(audio, target_sr=self.sample_rate)
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
        # Stochastic sample (the next random draw after the latent
        # `randn` above), so encode-noise stays on the seeded sequence.
        from fastvideo.models.vaes.sa_audio import SAAudioVAEModel
        if isinstance(self.vae, SAAudioVAEModel):
            return self.vae.encode(audio, sample_posterior=True)
        # Bare-OobleckVAE path (tests).
        return self.vae.encode(audio.to(next(self.vae.parameters()).dtype)).sample()

    def _prepare_mask(self, mask, latent_len: int, device: torch.device) -> torch.Tensor:
        """Pad/truncate a binary mask to `sample_size`, then
        nearest-resample to `[1, 1, latent_len]`. Convention: 1 = keep
        the reference, 0 = regenerate.

        `mask` may be a `[samples]` tensor at the model sample rate or a
        `(keep_seconds, total_seconds)` tuple — the tuple form builds
        "keep first K seconds, regenerate the rest" automatically.
        """
        if isinstance(mask, tuple) and len(mask) == 2:
            keep_s, total_s = (float(x) for x in mask)
            keep_n = int(keep_s * self.sample_rate)
            total_n = int(total_s * self.sample_rate)
            mask = torch.zeros(total_n, dtype=torch.float32)
            mask[:keep_n] = 1.0
        m = mask.to(device=device, dtype=torch.float32)
        if m.dim() == 1:
            m = m.unsqueeze(0)
        cur_len = m.shape[-1]
        if cur_len < self.sample_size:
            m = F.pad(m, (0, self.sample_size - cur_len))
        elif cur_len > self.sample_size:
            m = m[..., :self.sample_size]
        return F.interpolate(m.unsqueeze(1), size=latent_len, mode="nearest")


def _decode_audio_file(path, target_sr: int) -> torch.Tensor:
    """Decode any audio-bearing file (wav, mp3, mp4, m4a, flac, ...) via
    PyAV and resample to `target_sr`. Returns `[channels, samples]`
    float32 in roughly [-1, 1].

    PyAV is already a FastVideo dep (used for muxing in
    `VideoGenerator._mux_audio`). `torchaudio.load` on container
    formats (mp4 / m4a) routes through `torchcodec`, which pulls in a
    full CUDA NVRTC stack we don't otherwise need.
    """
    import av
    import numpy as np
    container = av.open(str(path))
    audio_stream = next(s for s in container.streams if s.type == "audio")
    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=target_sr)
    chunks: list = []
    for frame in container.decode(audio_stream):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray())
    for resampled in resampler.resample(None):
        chunks.append(resampled.to_ndarray())
    container.close()
    if not chunks:
        raise RuntimeError(f"No audio frames decoded from {path}")
    waveform = np.concatenate(chunks, axis=-1)
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    return torch.from_numpy(waveform).float()
