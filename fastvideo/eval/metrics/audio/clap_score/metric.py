"""Verse-Bench CLAP Score (CS) — text-audio cosine similarity.

Uses LAION CLAP to compute cosine similarity between audio and text embeddings.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


def _int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def _float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


@register("audio.clap_score")
class VBClapScoreMetric(BaseMetric):
    """Verse-Bench CLAP Score: text-audio cosine similarity."""

    name = "audio.clap_score"
    requires_reference = False
    higher_is_better = True
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

        audio = sample["audio"]          # str or list[str]
        text = sample["text_prompt"]     # str or list[str]
        if isinstance(audio, str):
            audio = [audio]
        if isinstance(text, str):
            text = [text]

        results = []
        for a, t in zip(audio, text):
            audio_emb = self._get_audio_emb(a)
            text_emb = self._model.get_text_embedding([t], use_tensor=True)
            score = F.cosine_similarity(audio_emb, text_emb, dim=1, eps=1e-8)[0].cpu().item()
            results.append(MetricResult(name=self.name, score=score, details={}))
        return results
