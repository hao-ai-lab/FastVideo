"""Verse-Bench Word Error Rate (WER).

Uses SenseVoice (FunASR) for ASR + jiwer for WER.
"""

from __future__ import annotations

import string

import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

PUNCTUATION_SET = set(string.punctuation)


@register("audio.wer")
class VBWERMetric(BaseMetric):
    """Verse-Bench WER via SenseVoice ASR + jiwer."""

    name = "audio.wer"
    requires_reference = False
    higher_is_better = False
    needs_gpu = True
    dependencies = ["jiwer", "funasr"]

    def __init__(self, model_name: str = "FunAudioLLM/SenseVoiceSmall") -> None:
        super().__init__()
        self._model_name = model_name
        self._model = None

    def to(self, device):
        super().to(device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        from funasr import AutoModel
        self._model = AutoModel(
            model=self._model_name,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=str(self.device),
            hub="hf",
            disable_update=True,
        )

    def _transcribe(self, audio_path: str) -> str:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        res = self._model.generate(
            input=audio_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        return rich_transcription_postprocess(res[0]["text"])

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        if self._model is None:
            self.setup()

        audio = sample["audio"]           # str or list[str]
        text = sample["text_prompt"]      # str or list[str]
        if isinstance(audio, str):
            audio = [audio]
        if isinstance(text, str):
            text = [text]

        import jiwer

        results = []
        for a, gt_text in zip(audio, text):
            asr = self._transcribe(a)
            gt_text = gt_text.strip().lower()
            if set(asr).issubset(PUNCTUATION_SET):
                asr = ""
            asr = asr.strip().lower()
            wer = jiwer.wer(gt_text, asr)
            results.append(MetricResult(
                name=self.name,
                score=wer,
                details={"transcription": asr, "reference_text": gt_text},
            ))
        return results
