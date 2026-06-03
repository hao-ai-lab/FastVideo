# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the LTX-2 BWE mel front-end.

Targets Gob's S1 finding on #1398: ``_STFTFn.forward_basis`` and
``MelSTFT.mel_basis`` were registered as zeros and used immediately by
``F.conv1d`` / ``torch.matmul``.  Because ``VocoderLoader`` loads with
``strict=False``, a converted checkpoint that omits these deterministic
signal-processing buffers does not raise — instead BWE silently sees an
all-zero magnitude spectrogram and a constant log-mel.  These tests pin
the deterministic initialisation that fixes that.

CPU-only: no checkpoint weights or GPU kernels are needed.
"""

from __future__ import annotations

import math

import pytest
import torch

from fastvideo.models.audio.ltx2_audio_vae import (
    MelSTFT,
    _STFTFn,
    _build_mel_basis,
    _build_stft_basis,
)


_FILTER_LENGTH = 512
_HOP_LENGTH = 80
_WIN_LENGTH = 512
_N_MELS = 64
_SAMPLING_RATE = 16000


class TestSTFTBasisDeterministic:
    """``_STFTFn`` no longer leaves ``forward_basis`` as zeros."""

    def test_forward_basis_is_not_zero(self):
        stft = _STFTFn(_FILTER_LENGTH, _HOP_LENGTH, _WIN_LENGTH)
        assert torch.any(stft.forward_basis != 0), (
            "forward_basis is all-zero — the BWE STFT will silently produce "
            "zero magnitudes (regression of #1398's S1 bug).")

    def test_forward_basis_shape(self):
        stft = _STFTFn(_FILTER_LENGTH, _HOP_LENGTH, _WIN_LENGTH)
        n_freqs = _FILTER_LENGTH // 2 + 1
        assert stft.forward_basis.shape == (2 * n_freqs, 1, _FILTER_LENGTH)

    def test_forward_produces_nonzero_magnitude(self):
        """A non-trivial waveform must yield a non-zero magnitude spectrum."""
        stft = _STFTFn(_FILTER_LENGTH, _HOP_LENGTH, _WIN_LENGTH)
        # 0.5 s of a 1 kHz tone at 16 kHz sample rate
        n = _SAMPLING_RATE // 2
        t = torch.arange(n, dtype=torch.float32) / _SAMPLING_RATE
        wave = torch.sin(2 * math.pi * 1000.0 * t).unsqueeze(0)  # (1, n)
        magnitude, phase = stft(wave)
        assert torch.any(magnitude != 0), (
            "STFT magnitude was zero for a non-zero waveform — basis init "
            "regressed.")
        assert magnitude.shape == phase.shape
        assert magnitude.shape[1] == _FILTER_LENGTH // 2 + 1


class TestMelBasisDeterministic:
    """``MelSTFT`` no longer leaves ``mel_basis`` as zeros."""

    def test_mel_basis_is_not_zero(self):
        mel = MelSTFT(_FILTER_LENGTH,
                      _HOP_LENGTH,
                      _WIN_LENGTH,
                      _N_MELS,
                      sampling_rate=_SAMPLING_RATE)
        assert torch.any(mel.mel_basis != 0), (
            "mel_basis is all-zero — log-mel will collapse to log(1e-5) "
            "(regression of #1398's S1 bug).")

    def test_mel_basis_shape(self):
        mel = MelSTFT(_FILTER_LENGTH,
                      _HOP_LENGTH,
                      _WIN_LENGTH,
                      _N_MELS,
                      sampling_rate=_SAMPLING_RATE)
        assert mel.mel_basis.shape == (_N_MELS, _FILTER_LENGTH // 2 + 1)

    def test_mel_basis_is_non_negative(self):
        """Triangular mel filters are non-negative everywhere."""
        mel = MelSTFT(_FILTER_LENGTH,
                      _HOP_LENGTH,
                      _WIN_LENGTH,
                      _N_MELS,
                      sampling_rate=_SAMPLING_RATE)
        assert torch.all(mel.mel_basis >= 0)

    def test_mel_basis_changes_with_sampling_rate(self):
        """Different sample rates -> different mel filterbanks."""
        mel_16k = MelSTFT(_FILTER_LENGTH,
                          _HOP_LENGTH,
                          _WIN_LENGTH,
                          _N_MELS,
                          sampling_rate=16000)
        mel_48k = MelSTFT(_FILTER_LENGTH,
                          _HOP_LENGTH,
                          _WIN_LENGTH,
                          _N_MELS,
                          sampling_rate=48000)
        assert not torch.allclose(mel_16k.mel_basis, mel_48k.mel_basis)


class TestMelSpectrogramSanity:
    """End-to-end sanity: a 1 kHz tone produces a peaked log-mel, not constant."""

    def _build(self) -> MelSTFT:
        return MelSTFT(_FILTER_LENGTH,
                       _HOP_LENGTH,
                       _WIN_LENGTH,
                       _N_MELS,
                       sampling_rate=_SAMPLING_RATE)

    def test_log_mel_is_not_constant(self):
        mel = self._build()
        n = _SAMPLING_RATE  # 1 second
        t = torch.arange(n, dtype=torch.float32) / _SAMPLING_RATE
        wave = torch.sin(2 * math.pi * 1000.0 * t).unsqueeze(0)
        log_mel, magnitude, phase, energy = mel.mel_spectrogram(wave)
        # If forward_basis or mel_basis were zero, log_mel would collapse
        # to log(1e-5) ≈ -11.51 (a constant).
        assert log_mel.std().item() > 1e-3, (
            "log-mel is effectively constant; the deterministic basis init "
            "did not take effect (regression of #1398's S1 bug).")

    def test_log_mel_shapes_match_input(self):
        mel = self._build()
        n = _SAMPLING_RATE
        wave = torch.randn(2, n)
        log_mel, magnitude, _, energy = mel.mel_spectrogram(wave)
        # n_mels along channel axis; energy collapses freq dim.
        assert log_mel.shape[0] == 2
        assert log_mel.shape[1] == _N_MELS
        assert magnitude.shape[0] == 2
        assert energy.shape[0] == 2


class TestStandaloneBasisHelpers:
    """Cover the standalone helpers directly so a future zero-init regression
    is caught even if ``_STFTFn`` / ``MelSTFT`` constructors change."""

    def test_stft_basis_helper_nonzero(self):
        basis = _build_stft_basis(_FILTER_LENGTH, _WIN_LENGTH)
        assert torch.any(basis != 0)
        assert basis.shape == (2 * (_FILTER_LENGTH // 2 + 1), 1,
                               _FILTER_LENGTH)
        assert basis.dtype == torch.float32

    def test_mel_basis_helper_nonzero(self):
        basis = _build_mel_basis(_SAMPLING_RATE, _FILTER_LENGTH, _N_MELS)
        assert torch.any(basis != 0)
        assert basis.shape == (_N_MELS, _FILTER_LENGTH // 2 + 1)
        assert basis.dtype == torch.float32
