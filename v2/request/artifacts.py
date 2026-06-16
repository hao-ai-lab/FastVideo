"""Artifact — a named, typed output with provenance (design_v3 §12).

"This kills the ``extra['audio']`` pattern — audio carries its sample rate as a
first-class artifact, not a dict passenger." Every artifact records which program
node produced it (``producer``), so multi-modal fan-out outputs are traceable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .._types import TensorLike
from .modalpart import Modality


@dataclass
class Artifact:
    name: str                       # output slot name, e.g. "video", "audio", "text"
    producer: str = ""              # program node id that produced it (provenance)
    meta: dict[str, object] = field(default_factory=dict)


@dataclass
class VideoArtifact(Artifact):
    frames: TensorLike = None       # T x H x W x C
    fps: int = 16


@dataclass
class AudioArtifact(Artifact):
    samples: TensorLike = None
    sample_rate: int = 44100        # first-class, not an extra-dict passenger


@dataclass
class TextArtifact(Artifact):
    token_ids: tuple[int, ...] = ()
    text: str = ""


@dataclass
class TensorArtifact(Artifact):
    tensor: TensorLike = None       # generic (e.g. action stream)


@dataclass
class LatentArtifact(Artifact):
    latent: TensorLike = None
    of_modality: Modality = Modality.VIDEO


@dataclass
class Output:
    """The collected result of running a program for one request."""
    request_id: str
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None         # set on partial/aborted delivery (failure isolation)

    def get(self, name: str) -> Artifact | None:
        return self.artifacts.get(name)
