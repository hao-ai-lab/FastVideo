"""Pluggable video feature extractors for FVD.

Three extractors, sharing the ``_BaseExtractor`` contract:

* ``i3d``      — Kinetics-400 I3D (TorchScript, ``flateon/FVD-I3D-torchscript``).
                 The standard FVD feature space used in the literature.
* ``clip``     — CLIP ViT-B/32 per-frame embeddings, mean-pooled over time.
                 Captures semantic / content quality.
* ``videomae`` — VideoMAE-base last-hidden-state, mean-pooled over patch tokens.
                 Captures structural / motion quality.

The contract is intentionally narrow: each extractor takes a
``(B, T, C, H, W)`` float tensor in ``[0, 1]`` and returns ``(B, D)`` numpy
features.  Preprocessing (resize, normalize, layout) is the extractor's job;
its callers should not care.

Adapted from the standalone code at ``benchmarks/fvd/feature_extractors.py``
that this module replaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

_I3D_REPO_ID = "flateon/FVD-I3D-torchscript"
_I3D_FILENAME = "i3d_torchscript.pt"
_I3D_MIN_FRAMES = 10  # I3D hard minimum (Kinetics-400 sampling)
_I3D_FEATURE_DIM = 400

_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_VIDEOMAE_MODEL_NAME = "MCG-NJU/videomae-base"


class _BaseExtractor(ABC):
    """Common contract: ``forward`` ((B,T,C,H,W) float[0,1]) → ``(B, feature_dim)`` ndarray."""

    feature_dim: int

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def to(self, device: torch.device) -> _BaseExtractor:
        self.device = device
        return self

    @abstractmethod
    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> np.ndarray:
        """Extract features for ``video`` and return a ``(B, feature_dim)`` numpy array."""


class _I3DExtractor(_BaseExtractor):
    """Kinetics-400 I3D — the canonical FVD feature space.

    ``torch.jit.fuser('none')`` disables NVRTC kernel fusion for the I3D
    TorchScript forward pass.  Without it, PyTorch tries to JIT-compile fused
    kernels via ``libnvrtc-builtins``, which is only available on the exact
    CUDA version the binary was built against (e.g. fails on Colab CUDA 12 when
    the lib expects CUDA 13).  No effect on numerical correctness.
    """

    feature_dim = _I3D_FEATURE_DIM

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)
        from fastvideo.eval.models import ensure_checkpoint
        path = ensure_checkpoint(_I3D_FILENAME, source=_I3D_REPO_ID, filename=_I3D_FILENAME)
        model = torch.jit.load(path, map_location=device)
        model.eval()
        self._model = model

    def to(self, device: torch.device) -> _I3DExtractor:
        super().to(device)
        self._model = self._model.to(device)
        return self

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> np.ndarray:
        B, T, C, H, W = video.shape
        if T < _I3D_MIN_FRAMES:
            raise ValueError(f"I3D requires at least {_I3D_MIN_FRAMES} frames, got {T}. "
                             "Increase num_frames or use a longer video.")
        if H != 224 or W != 224:
            video = video.reshape(B * T, C, H, W)
            video = F.interpolate(video, size=(224, 224), mode="bilinear", align_corners=False)
            video = video.reshape(B, T, C, 224, 224)
        # [0, 1] → [-1, 1] then (B, T, C, H, W) → (B, C, T, H, W)
        batch = (video * 2.0 - 1.0).permute(0, 2, 1, 3, 4).contiguous().to(self.device)
        with torch.jit.fuser("none"):
            feats = self._model(batch, rescale=False, resize=False, return_features=True)
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)  # (D,) → (1, D) when I3D squeezes B=1
        return feats.cpu().numpy()


class _CLIPExtractor(_BaseExtractor):
    """CLIP ViT-B/32 per-frame embeds, mean-pooled over time. Semantic features."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:
            raise ImportError("common.fvd with extractor='clip' requires transformers. "
                              "Install with: uv pip install -e '.[eval]'") from e
        self._processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)
        self._model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME).to(device)
        self._model.eval()
        self.feature_dim = self._model.config.projection_dim

    def to(self, device: torch.device) -> _CLIPExtractor:
        super().to(device)
        self._model = self._model.to(device)
        return self

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> np.ndarray:
        B, T, C, H, W = video.shape
        # CLIPProcessor expects uint8 images in [0, 255]; the eval pipeline
        # hands us float [0, 1].
        frames = (video.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        frames = frames.view(B * T, C, H, W).cpu()
        inputs = self._processor(images=list(frames), return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self._model.get_image_features(**inputs)  # (B*T, D)
        feats = feats.view(B, T, -1).mean(dim=1)  # mean-pool over time → (B, D)
        return feats.cpu().numpy()


class _VideoMAEExtractor(_BaseExtractor):
    """VideoMAE-base last-hidden-state, patch-token mean-pooled. Structural features."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)
        try:
            from transformers import VideoMAEModel
        except ImportError as e:
            raise ImportError("common.fvd with extractor='videomae' requires transformers. "
                              "Install with: uv pip install -e '.[eval]'") from e
        self._model = VideoMAEModel.from_pretrained(_VIDEOMAE_MODEL_NAME).to(device)
        self._model.eval()
        self.feature_dim = self._model.config.hidden_size
        # ImageNet normalization buffers, broadcastable over (B, T, C, H, W)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    def to(self, device: torch.device) -> _VideoMAEExtractor:
        super().to(device)
        self._model = self._model.to(device)
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        return self

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> np.ndarray:
        B, T, C, H, W = video.shape
        if H != 224 or W != 224:
            video = video.view(B * T, C, H, W)
            video = F.interpolate(video, size=(224, 224), mode="bilinear", align_corners=False)
            video = video.view(B, T, C, 224, 224)
        pixel_values = ((video.float().to(self.device) - self._mean) / self._std)
        outputs = self._model(pixel_values)
        # (B, n_patches, D) → mean over patches → (B, D)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


_EXTRACTORS: dict[str, type[_BaseExtractor]] = {
    "i3d": _I3DExtractor,
    "clip": _CLIPExtractor,
    "videomae": _VideoMAEExtractor,
}


def available_extractors() -> list[str]:
    return sorted(_EXTRACTORS.keys())


def load_extractor(name: str, device: torch.device) -> _BaseExtractor:
    """Instantiate the named extractor on *device*. Raises ``ValueError`` on unknown names."""
    cls = _EXTRACTORS.get(name)
    if cls is None:
        raise ValueError(f"Unknown FVD extractor '{name}'. Available: {available_extractors()}")
    return cls(device)
