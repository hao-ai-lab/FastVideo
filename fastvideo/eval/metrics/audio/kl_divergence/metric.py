"""PaSST KL divergence ``KL(gt || pred)`` on AudioSet-527 logits.

Ports ``av_bench.metrics.kl.compute_kl`` 1:1. The primary ``score`` is
the softmax variant (reported as "MKL" / "KL_PaSST" by V2A papers);
the sigmoid variant is exposed in ``details["kl_sigmoid"]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

SAMPLING_RATE = 32000


def _collect_logits(
    model: Any,
    audio_path: str,
    device: torch.device,
    *,
    window_size: int = 10,
    overlap: int = 5,
    collect: str = "mean",
) -> torch.Tensor:
    """Run PaSST over sliding windows of *audio_path*; return mean logits ``(527,)``."""
    import librosa
    import pyloudnorm as pyln

    audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    audio = pyln.normalize.peak(audio, -1.0)

    step_size = int((window_size - overlap) * SAMPLING_RATE)
    win_samples = int(window_size * SAMPLING_RATE)
    per_window: list[torch.Tensor] = []
    for i in range(0, max(step_size, len(audio) - step_size), step_size):
        window = audio[i:i + win_samples]
        # Pad short tail windows up to the model's expected length, but
        # only if the tail is at least 15% of a window — mirrors AudioGen.
        if len(window) < win_samples and len(window) > int(win_samples * 0.15):
            tmp = np.zeros(win_samples, dtype=np.float32)
            tmp[:len(window)] = window
            window = tmp
        wav = torch.from_numpy(np.asarray(window, dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(wav)
        per_window.append(logits.squeeze().detach().cpu())

    stacked = torch.stack(per_window)  # (W, 527)
    if collect == "mean":
        return stacked.mean(dim=0)
    if collect == "max":
        return stacked.max(dim=0).values
    raise ValueError(f"unknown collect mode {collect!r}")


def _kl_softmax(pred_logits: torch.Tensor, gt_logits: torch.Tensor) -> float:
    """``KL(gt || pred)`` over class-probability softmax. Matches av-benchmark."""
    return float(
        F.kl_div(
            F.log_softmax(pred_logits, dim=-1),
            F.log_softmax(gt_logits, dim=-1),
            reduction="sum",
            log_target=True,
        ))


def _kl_sigmoid(pred_logits: torch.Tensor, gt_logits: torch.Tensor) -> float:
    """``KL(gt || pred)`` over per-class Bernoulli sigmoid (multi-label variant)."""
    return float(F.kl_div(
        F.logsigmoid(pred_logits),
        F.logsigmoid(gt_logits),
        reduction="sum",
        log_target=True,
    ))


@register("audio.kl_divergence")
class KLDivergenceMetric(BaseMetric):
    """PaSST KL divergence ``KL(gt || pred)`` on AudioSet-527 logits.

    Per-sample. Requires ``sample["audio"]`` (generated) and
    ``sample["reference_audio"]`` (ground truth).
    """

    name = "audio.kl_divergence"
    requires_reference = True
    higher_is_better = False
    needs_gpu = True
    is_set_metric = False
    dependencies = ["hear21passt", "librosa", "pyloudnorm"]

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    def to(self, device):
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        from hear21passt.base import get_basic_model
        model = get_basic_model(mode="logits")
        model.eval()
        model = model.to(self.device)
        self._model = model

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        if self._model is None:
            self.setup()
        gen_path = sample.get("audio")
        ref_path = sample.get("reference_audio")
        if gen_path is None or ref_path is None:
            return self._skip(sample, "missing 'audio' or 'reference_audio'")

        gt_logits = _collect_logits(self._model, ref_path, self.device)
        pred_logits = _collect_logits(self._model, gen_path, self.device)
        if not torch.isfinite(gt_logits).all() or not torch.isfinite(pred_logits).all():
            which = []
            if not torch.isfinite(gt_logits).all():
                which.append("reference_audio")
            if not torch.isfinite(pred_logits).all():
                which.append("audio")
            return self._skip(
                sample,
                f"non-finite PaSST logits on {'+'.join(which)} (silent or corrupt audio?)",
            )
        kl_sm = _kl_softmax(pred_logits, gt_logits)
        kl_sg = _kl_sigmoid(pred_logits, gt_logits)
        return MetricResult(
            name=self.name,
            score=kl_sm,
            details={
                "kl_softmax": kl_sm,
                "kl_sigmoid": kl_sg
            },
        )
