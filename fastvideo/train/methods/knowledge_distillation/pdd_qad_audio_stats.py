# SPDX-License-Identifier: Apache-2.0
"""Decoded-audio statistics and the teacher-seed-floor gate for QAD checkpoints.

Latent-space audio error does not show what the Trim DMD post-mortem found mattered: decoded audio drifted dull
(spectral centroid 952 Hz teacher -> 530 Hz late checkpoint) while losses looked fine. These statistics, compared
with how much the bf16 teacher itself varies between two seeds, make that drift a selection gate.
"""
from __future__ import annotations

import math

import torch

STATS = ("spectral_centroid_hz", "share_above_8khz", "dynamic_range_db", "loudness_iqr_db")


def _spectrogram(wave: torch.Tensor, n_fft: int = 2048, hop: int = 512) -> torch.Tensor:
    """Power spectrogram ``[freq, frames]`` of a mono or stereo (``[samples, 2]``) waveform."""
    mono = wave.float().mean(dim=-1) if wave.ndim == 2 else wave.float()
    window = torch.hann_window(n_fft)
    spec = torch.stft(mono, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    return spec.abs().pow(2)


def audio_stats(wave: torch.Tensor, sample_rate: int) -> dict[str, float]:
    power = _spectrogram(wave)
    freqs = torch.linspace(0, sample_rate / 2, power.shape[0])
    energy = power.sum(dim=0)
    voiced = energy > energy.max() * 1e-6
    centroid = (power * freqs[:, None]).sum(0) / energy.clamp_min(1e-20)
    frame_db = 10 * torch.log10(energy.clamp_min(1e-20))
    voiced_db = frame_db[voiced] if bool(voiced.any()) else frame_db
    q = torch.quantile(voiced_db, torch.tensor([0.05, 0.25, 0.75, 0.95]))
    return {
        "spectral_centroid_hz": float(centroid[voiced].median()) if bool(voiced.any()) else 0.0,
        "share_above_8khz": float(power[freqs > 8000].sum() / power.sum().clamp_min(1e-20)),
        "dynamic_range_db": float(q[3] - q[0]),
        "loudness_iqr_db": float(q[2] - q[1]),
    }


def log_mel_distance(a: torch.Tensor, b: torch.Tensor, sample_rate: int, n_mels: int = 80) -> float:
    """Mean absolute difference of log-mel spectrograms (triangular mel filterbank, no torchaudio needed)."""
    pa, pb = _spectrogram(a), _spectrogram(b)
    frames = min(pa.shape[1], pb.shape[1])
    n_freq = pa.shape[0]

    def mel(f: torch.Tensor) -> torch.Tensor:
        return 2595 * torch.log10(1 + f / 700)

    edges = 700 * (10**(torch.linspace(0, float(mel(torch.tensor(sample_rate / 2))), n_mels + 2) / 2595) - 1)
    freqs = torch.linspace(0, sample_rate / 2, n_freq)
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    bank = torch.clamp(torch.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), min=0)
    la = torch.log(bank @ pa[:, :frames] + 1e-8)
    lb = torch.log(bank @ pb[:, :frames] + 1e-8)
    return float((la - lb).abs().mean())


def seed_floor_gate(student: list[dict[str, float]],
                    teacher: list[dict[str, float]],
                    teacher_other_seed: list[dict[str, float]],
                    slack: float = 1.0) -> dict[str, float | bool]:
    """Per statistic: median |student - teacher| (same seed) must be <= slack * median |teacher(s+1) - teacher(s)|.

    Rows are aligned lists. Returns per-stat student gap, floor and a ``pass`` flag over all statistics.
    """
    result: dict[str, float | bool] = {}
    passed = True
    for name in STATS:
        gap = torch.tensor([abs(s[name] - t[name]) for s, t in zip(student, teacher, strict=True)])
        floor = torch.tensor([abs(o[name] - t[name]) for o, t in zip(teacher_other_seed, teacher, strict=True)])
        gap_m, floor_m = float(gap.median()), float(floor.median())
        result[f"{name}/student_gap"] = gap_m
        result[f"{name}/teacher_floor"] = floor_m
        ok = gap_m <= slack * floor_m or math.isclose(gap_m, 0.0, abs_tol=1e-9)
        result[f"{name}/pass"] = ok
        passed &= ok
    result["pass"] = passed
    return result


__all__ = ["STATS", "audio_stats", "log_mel_distance", "seed_floor_gate"]
