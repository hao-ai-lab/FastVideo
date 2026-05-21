"""CLAP Score (CS) — text-audio cosine similarity.

Uses HuggingFace ``transformers.ClapModel`` with the
``laion/clap-htsat-fused`` checkpoint, the closest HF mirror of the
``630k-audioset-fusion-best.pt`` weights that
``hkchengrex/av-benchmark`` and the V2A literature use. The fully
byte-exact comparison still requires the ``laion_clap`` pip package
(which pins ``numpy<2`` and forces a fastvideo-wide downgrade — so
we don't depend on it). Numbers from this metric are in the same
ballpark as av-benchmark's ``LAION_CLAP``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

DEFAULT_CLAP_REPO = "laion/clap-htsat-fused"
CLAP_SAMPLE_RATE = 48000


@register("audio.clap_score")
class ClapScoreMetric(BaseMetric):
    """CLAP score — text-audio cosine similarity via HF CLAP."""

    name = "audio.clap_score"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    is_set_metric = False
    dependencies = ["librosa", "pyloudnorm"]

    def __init__(self, model_name: str = DEFAULT_CLAP_REPO) -> None:
        super().__init__()
        self._model_name = model_name
        self._model: Any = None
        self._processor: Any = None

    def to(self, device):
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        from transformers import ClapModel, ClapProcessor
        self._processor = ClapProcessor.from_pretrained(self._model_name)
        self._model = ClapModel.from_pretrained(self._model_name).to(self.device)
        self._model.eval()

    def _load_audio(self, audio_path: str):
        import librosa
        import pyloudnorm as pyln
        audio, _ = librosa.load(audio_path, sr=CLAP_SAMPLE_RATE, mono=True)
        return pyln.normalize.peak(audio, -1.0)

    def _audio_emb(self, audio_path: str) -> torch.Tensor:
        audio = self._load_audio(audio_path)
        inputs = self._processor(audio=audio, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._model.get_audio_features(**inputs)

    def _text_emb(self, text: str) -> torch.Tensor:
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._model.get_text_features(**inputs)

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        if self._model is None:
            self.setup()

        audio = sample.get("audio")
        text = sample.get("text_prompt")
        if audio is None or text is None:
            return self._skip(sample, "missing 'audio' or 'text_prompt'")

        audio_emb = self._audio_emb(audio)
        text_emb = self._text_emb(text)
        score = F.cosine_similarity(audio_emb, text_emb, dim=1, eps=1e-8)[0].cpu().item()
        return MetricResult(name=self.name, score=score, details={})
