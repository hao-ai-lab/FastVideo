"""ModalPart — typed multimodal request inputs (design_v3 §12; design.md §6.1 G2).

Requests carry a list of typed modality parts instead of a god-batch with
per-model fields. This kills the ``ForwardBatch`` blackboard / ``extra[...]``
pattern (design.md P3) at the input boundary; artifacts (outputs) do the same
on the way out (see ``artifacts.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from v2._types import TensorLike


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    ACTION = "action"
    LATENT = "latent"


@dataclass(frozen=True)
class ModalPart:
    modality: Modality


@dataclass(frozen=True)
class TextPart(ModalPart):
    text: str = ""

    def __init__(self, text: str = ""):
        object.__setattr__(self, "modality", Modality.TEXT)
        object.__setattr__(self, "text", text)


@dataclass(frozen=True)
class ImagePart(ModalPart):
    pixels: TensorLike = None          # HxWxC or CxHxW (backend decides), or a path
    path: str | None = None

    def __init__(self, pixels: TensorLike = None, path: str | None = None):
        object.__setattr__(self, "modality", Modality.IMAGE)
        object.__setattr__(self, "pixels", pixels)
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class VideoPart(ModalPart):
    frames: TensorLike = None          # T x H x W x C
    fps: float = 16.0
    path: str | None = None

    def __init__(self, frames: TensorLike = None, fps: float = 16.0, path: str | None = None):
        object.__setattr__(self, "modality", Modality.VIDEO)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class AudioPart(ModalPart):
    samples: TensorLike = None
    sample_rate: int = 44100

    def __init__(self, samples: TensorLike = None, sample_rate: int = 44100):
        object.__setattr__(self, "modality", Modality.AUDIO)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "sample_rate", sample_rate)


@dataclass(frozen=True)
class ActionPart(ModalPart):
    """Camera / mouse / keyboard / pose conditioning for interactive world models."""
    tensor: TensorLike = None
    kind: str = "camera"  # camera | mouse | keyboard | pose

    def __init__(self, tensor: TensorLike = None, kind: str = "camera"):
        object.__setattr__(self, "modality", Modality.ACTION)
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True)
class LatentPart(ModalPart):
    latent: TensorLike = None
    of_modality: Modality = Modality.VIDEO

    def __init__(self, latent: TensorLike = None, of_modality: Modality = Modality.VIDEO):
        object.__setattr__(self, "modality", Modality.LATENT)
        object.__setattr__(self, "latent", latent)
        object.__setattr__(self, "of_modality", of_modality)
