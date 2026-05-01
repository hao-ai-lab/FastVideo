"""Verse-Bench KL Divergence (KL) for audio.

Uses PaSST to extract class probabilities from audio windows, then
computes KL divergence.
"""

from __future__ import annotations

import contextlib
import os
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

SAMPLING_RATE = 32000


class _patch_passt_stft:
    def __init__(self):
        self.old_stft = torch.stft
    def __enter__(self):
        torch.stft = partial(torch.stft, return_complex=False)
    def __exit__(self, *exc):
        torch.stft = self.old_stft


def _return_probabilities(model, audio_path, device, window_size=10, overlap=5, collect='mean'):
    import librosa
    import pyloudnorm as pyln

    audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    audio = pyln.normalize.peak(audio, -1.0)

    step_size = int((window_size - overlap) * SAMPLING_RATE)
    probabilities = []
    for i in range(0, max(step_size, len(audio) - step_size), step_size):
        window = audio[i:i + int(window_size * SAMPLING_RATE)]
        if len(window) < int(window_size * SAMPLING_RATE):
            if len(window) > int(window_size * SAMPLING_RATE * 0.15):
                tmp = np.zeros(int(window_size * SAMPLING_RATE))
                tmp[:len(window)] = window
                window = tmp
        audio_wave = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(device)
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), torch.no_grad(), _patch_passt_stft():
            logits = model(audio_wave)
            probabilities.append(torch.squeeze(logits))

    probabilities = torch.stack(probabilities)
    if collect == 'mean':
        probabilities = torch.mean(probabilities, dim=0)
    elif collect == 'max':
        probabilities, _ = torch.max(probabilities, dim=0)

    return F.softmax(probabilities, dim=0).squeeze().cpu()


@register("audio.kl_divergence")
class VBKLDivergenceAudioMetric(BaseMetric):
    """Verse-Bench KL Divergence for audio via PaSST."""

    name = "audio.kl_divergence"
    requires_reference = True
    higher_is_better = False
    needs_gpu = True
    dependencies = ["hear21passt", "librosa", "pyloudnorm"]

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def to(self, device):
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            from hear21passt.base import get_basic_model
            model = get_basic_model(mode="logits")
            model.eval()
            model = model.to(self.device)
        self._model = model

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        if self._model is None:
            self.setup()

        audio = sample["audio"]                  # str or list[str]
        ref_audio = sample["reference_audio"]    # str or list[str]
        if isinstance(audio, str):
            audio = [audio]
        if isinstance(ref_audio, str):
            ref_audio = [ref_audio]

        results = []
        for a, ra in zip(audio, ref_audio):
            ref_p = _return_probabilities(self._model, ra, self.device)
            eval_p = _return_probabilities(self._model, a, self.device)
            kl = F.kl_div((ref_p + 1e-6).log(), eval_p, reduction='sum', log_target=False).cpu().item()
            results.append(MetricResult(name=self.name, score=kl, details={}))
        return results
