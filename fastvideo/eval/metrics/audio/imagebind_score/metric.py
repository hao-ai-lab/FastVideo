"""ImageBind audio↔video cosine similarity (``IB-Score``).

Per-sample. Loads the audio and video files separately through
ImageBind's ``data.load_and_transform_audio_data`` /
``load_and_transform_video_data``, runs them through
``imagebind_huge``, and returns the cosine similarity between the
``AUDIO`` and ``VISION`` embeddings. Matches
``hkchengrex/av-benchmark``'s ``IB-Score`` (mean over a paired set).

Sample shape (per row)::

    {"video_path": "clip.mp4", "audio": "clip.wav"}

We need the video *path*, not the pool-decoded tensor, because
ImageBind's video preprocessing decodes its own clips with decord
and runs a deterministic temporal sampler. Pass the path explicitly
as ``video_path`` (or wrap as ``Video(...)`` and the metric pulls
``video.source``).

ImageBind is CC BY-NC-SA 4.0; it is *not* vendored into FastVideo.
The ``[eval-audio]`` extra pulls it from the upstream GitHub repo
via ``[tool.uv.sources]``. See ``fastvideo/eval/README.md``.
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

# ImageBind's video loader delegates to ``pytorchvideo.encoded_video_decord``,
# which keeps non-thread-safe global decord state. When multiple
# ``EvalWorker`` threads call it concurrently, one of them sees a raw
# ``decord.NDArray`` instead of a torch tensor and pytorchvideo crashes
# with ``AttributeError: 'NDArray' object has no attribute 'to'``.
# Serialize the decode step across workers; the forward pass on each
# GPU still runs in parallel.
_IB_DECODE_LOCK = threading.Lock()


def _video_source(sample: dict) -> str | None:
    """Recover the original video path from a sample.

    Accepted shapes:
    * ``sample["video_path"]`` — explicit path-string kwarg.
    * ``sample["video"]`` — :class:`Video` whose ``.source`` is a path.
    """
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
        # Build an unloaded model and route the state-dict download
        # through our cache. ImageBind's stock ``imagebind_huge(pretrained=True)``
        # hardcodes ``.checkpoints/imagebind_huge.pth`` relative to the cwd,
        # which would pollute the working dir and bypass FASTVIDEO_EVAL_CACHE.
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

        # ``pytorchvideo.encoded_video_decord`` sets decord's bridge to
        # ``"torch"`` at module import, but decord stores the bridge in
        # ``threading.local()`` and its ``set_bridge`` never updates the
        # process-wide fallback (missing ``global``). So in multi-GPU
        # mode the EvalWorker thread sees the default ``"native"`` and
        # ``get_clip`` returns ``decord.NDArray`` — which pytorchvideo
        # then tries to ``.to(torch.float32)`` and crashes. Force the
        # right bridge on this thread before each call.
        decord.bridge.set_bridge("torch")
        with _IB_DECODE_LOCK:
            inputs = {
                ModalityType.VISION: ib_data.load_and_transform_video_data([video_path], self.device),
                ModalityType.AUDIO: ib_data.load_and_transform_audio_data([audio_path], self.device),
            }
        embeds = self._model(inputs)
        score = F.cosine_similarity(embeds[ModalityType.VISION], embeds[ModalityType.AUDIO], dim=-1)[0].item()
        return MetricResult(name=self.name, score=float(score), details={})
