"""ImageBind audio↔video cosine similarity (``IB-Score``).

Per-sample. Reads ``sample["video_path"]`` (or ``sample["video"].source``
for a :class:`Video` wrapper) and ``sample["audio"]``; ImageBind decodes
its own clips so the *path* is required, not the pool-decoded tensor.
"""

from __future__ import annotations

import threading
from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.models import ensure_checkpoint
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult, Video

_IMAGEBIND_URL = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
_IMAGEBIND_NAME = "imagebind_huge.pth"

# pytorchvideo's decord-backed loader is not thread-safe; serialize the
# decode step across workers. The forward pass on each GPU still runs in
# parallel.
_IB_DECODE_LOCK = threading.Lock()


def _video_source(sample: dict) -> str | None:
    vp = sample.get("video_path")
    if isinstance(vp, str):
        return vp
    v = sample.get("video")
    if isinstance(v, Video) and isinstance(v.source, str):
        return v.source
    return None


@register("audio.imagebind_score")
class ImageBindScoreMetric(BaseMetric):
    """ImageBind audio↔video cosine similarity, per-sample."""

    name = "audio.imagebind_score"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    is_set_metric = False
    backbone = "imagebind"
    dependencies = ["imagebind", "decord", "pytorchvideo"]

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
        from imagebind.models import imagebind_model
        # Route the state-dict through ``ensure_checkpoint``; the stock
        # ``imagebind_huge(pretrained=True)`` hardcodes ``.checkpoints/`` in cwd.
        model = imagebind_model.imagebind_huge(pretrained=False)
        ckpt = ensure_checkpoint(_IMAGEBIND_NAME, _IMAGEBIND_URL)
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
        self._model = model.eval().to(self.device)

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        if self._model is None:
            self.setup()

        video_path = _video_source(sample)
        audio_path = sample.get("audio")
        if video_path is None or audio_path is None:
            return self._skip(sample, "missing 'video_path'/'video.source' or 'audio'")

        import decord
        from imagebind import data as ib_data
        from imagebind.models.imagebind_model import ModalityType

        # decord.bridge is thread-local with a missing-``global`` upstream
        # bug; worker threads otherwise see ``"native"`` and pytorchvideo
        # crashes on the resulting NDArray. Re-set it on this thread.
        decord.bridge.set_bridge("torch")
        with _IB_DECODE_LOCK:
            inputs = {
                ModalityType.VISION: ib_data.load_and_transform_video_data([video_path], self.device),
                ModalityType.AUDIO: ib_data.load_and_transform_audio_data([audio_path], self.device),
            }
        embeds = self._model(inputs)
        score = F.cosine_similarity(embeds[ModalityType.VISION], embeds[ModalityType.AUDIO], dim=-1)[0].item()
        return MetricResult(name=self.name, score=float(score), details={})
