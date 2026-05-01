"""Verse-Bench Fréchet Distance (FD) for audio.

Uses LAION CLAP embeddings to compute Fréchet distance between generated
and reference audio.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import linalg

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


def _int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def _float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def _calculate_embd_statistics(embd_lst):
    if isinstance(embd_lst, list):
        embd_lst = np.array(embd_lst)
    mu = np.mean(embd_lst, axis=0)
    sigma = np.cov(embd_lst, rowvar=False)
    return mu, sigma

def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


@register("audio.frechet_distance")
class VBFrechetDistanceAudioMetric(BaseMetric):
    """Verse-Bench Fréchet Distance for audio via LAION CLAP."""

    name = "audio.frechet_distance"
    requires_reference = True
    higher_is_better = False
    needs_gpu = True
    dependencies = ["laion_clap", "librosa", "pyloudnorm"]

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def to(self, device):
        super().to(device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=True, device=str(self.device))
        model.load_ckpt()
        model.eval()
        self._model = model

    def _get_audio_emb(self, audio_path: str) -> torch.Tensor:
        import librosa
        import pyloudnorm as pyln

        audio, _ = librosa.load(audio_path, sr=48000, mono=True)
        audio = pyln.normalize.peak(audio, -1.0)
        audio = audio.reshape(1, -1)
        audio = torch.from_numpy(_int16_to_float32(_float32_to_int16(audio))).float()
        with torch.no_grad():
            emb = self._model.get_audio_embedding_from_data(x=audio, use_tensor=True)
        return emb

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
            emb1 = self._get_audio_emb(a)
            emb2 = self._get_audio_emb(ra)
            mu1, sigma1 = _calculate_embd_statistics(emb1.cpu().numpy())
            mu2, sigma2 = _calculate_embd_statistics(emb2.cpu().numpy())
            fd = _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            results.append(MetricResult(name=self.name, score=float(fd), details={}))
        return results
