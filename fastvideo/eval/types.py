from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MetricResult:
    """Standard result container returned by all metrics.

    ``score`` is ``None`` when the metric was skipped (e.g. missing
    required input).  Check ``details["skipped"]`` for the reason.
    """
    name: str
    score: float | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Video:
    """A media handle bundling frames + audio behind one typed value.

    The name matches the package; in practice this also accepts pure
    audio inputs (``.wav`` / ``.mp3``) — for those, ``frames`` stays
    ``None`` and only ``audio`` is populated. Conversely, a silent
    video has ``audio=None``. This intentionally mirrors how a
    container format (mp4) carries optional streams.

    The ``VideoPool`` (a thin async prefetcher in front of the
    Evaluator) walks the per-sample kwargs dict, finds every ``Video``
    value, and triggers the decodes asked for by the registered
    metrics — so by the time a metric's ``compute(sample)`` runs, the
    ``frames`` / ``audio`` attributes are already populated. Metric
    code reads them directly:

        def compute(self, sample):
            video = sample["video"]
            frames = video.frames           # (T, C, H, W) in [0, 1]
            audio = video.audio             # 1D float32

    Constructors accepted at the user boundary::

        Video("clip.mp4")             # mp4 / wav / image-dir / etc.
        Video(tensor)                 # pre-decoded (T, C, H, W) tensor
        Video(list_of_PIL_images)     # frame list

    A folder of clips is **not** a single ``Video`` — set-vs-set
    inputs are a list of ``Video`` objects (one per clip), wired by
    the user (e.g. via ``Evaluator.evaluate(samples=[...])``).
    """

    source: Any  # VideoSource at runtime; typed broadly to avoid a torch import here
    fps: float | None = None

    # Decoded streams. ``None`` until the pool (or the user) calls a
    # decode helper. Both can be ``None`` simultaneously — that's the
    # state of a fresh handle whose source is a path.
    frames: Any = None  # torch.Tensor | None
    audio: Any = None  # np.ndarray | None
    audio_sr: int | None = None

    def has_frames(self) -> bool:
        return self.frames is not None

    def has_audio(self) -> bool:
        return self.audio is not None

    def __post_init__(self) -> None:
        # Coerce Path → str so downstream loaders see a single shape.
        if isinstance(self.source, Path):
            self.source = str(self.source)
