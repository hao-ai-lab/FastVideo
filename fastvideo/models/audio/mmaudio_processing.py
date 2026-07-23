# SPDX-License-Identifier: Apache-2.0
"""Audio preprocessing used by native MMAudio training feature extraction."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class MMAudioMelConverter(nn.Module):
    """Convert waveforms to MMAudio's log-mel representation.

    The FFT, padding, mel filter, and logarithm match the published MMAudio
    preprocessing contract. Keeping this component in FastVideo avoids a
    runtime import from the upstream training repository.
    """

    def __init__(
        self,
        *,
        sampling_rate: int,
        n_fft: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        fmin: float,
        fmax: float,
        log_base: Literal["e", "10"],
    ) -> None:
        super().__init__()
        from librosa.filters import mel as librosa_mel_fn

        mel = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.log_base = log_base
        self.register_buffer("mel_basis", torch.from_numpy(mel).float())
        self.register_buffer("hann_window", torch.hann_window(win_size))

    def forward(self, waveform: torch.Tensor, center: bool = False) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"MMAudio waveforms must have shape [batch, samples], got {tuple(waveform.shape)}")
        waveform = waveform.clamp(min=-1.0, max=1.0).to(self.mel_basis.device)
        padding = int((self.n_fft - self.hop_size) / 2)
        waveform = torch.nn.functional.pad(
            waveform.unsqueeze(1),
            [padding, padding],
            mode="reflect",
        ).squeeze(1)
        spectrum = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(-1) + 1e-9)
        mel = torch.matmul(self.mel_basis, magnitude)
        mel = torch.clamp(mel, min=1e-5)
        return torch.log10(mel) if self.log_base == "10" else torch.log(mel)


def build_mmaudio_mel_converter(mode: Literal["16k", "44k"] = "44k", ) -> MMAudioMelConverter:
    if mode == "16k":
        return MMAudioMelConverter(
            sampling_rate=16_000,
            n_fft=1024,
            num_mels=80,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=8_000,
            log_base="10",
        )
    if mode == "44k":
        return MMAudioMelConverter(
            sampling_rate=44_100,
            n_fft=2048,
            num_mels=128,
            hop_size=512,
            win_size=2048,
            fmin=0,
            fmax=44_100 / 2,
            log_base="e",
        )
    raise ValueError(f"Unknown MMAudio mel mode: {mode}")


__all__ = ["MMAudioMelConverter", "build_mmaudio_mel_converter"]
