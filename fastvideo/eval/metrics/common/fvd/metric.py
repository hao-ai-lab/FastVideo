"""common.fvd — Fréchet Video Distance.

Measures distributional similarity between generated and reference videos
using I3D features (trained on Kinetics-400). Lower score → generated
videos are closer to the real video distribution.

FVD is a dataset-level metric — it compares the
distribution of a collection of videos rather than scoring each video
individually. For statistically reliable results at least 256 videos are
recommended; the standard protocol uses 2048.

This metric follows the set-vs-set protocol (is_set_metric=True):
  - :meth:`accumulate` is called once per video to buffer I3D features.
  - :meth:`finalize`   is called once after all videos to compute FVD.
  - :meth:`reset`      clears buffers between evaluation runs.

Reference features
------------------
On the first call to :meth:`accumulate` that includes
``sample["reference"]``, I3D features are extracted from those reference
videos and saved to *cache_path*. Every subsequent run loads that cache
automatically — no need to pass ``sample["reference"]`` again.

Cache default: ``${FASTVIDEO_EVAL_CACHE}/fvd/real_features.pt``
(override via ``FASTVIDEO_EVAL_CACHE`` env-var; see
:func:`fastvideo.eval.models.get_cache_dir`).

I3D model
---------
Downloaded automatically from HuggingFace on first use via
:func:`fastvideo.eval.models.ensure_checkpoint` (filelock-safe):
    flateon/FVD-I3D-torchscript  (i3d_torchscript.pt)
Requires ``huggingface_hub`` (included in fastvideo deps).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_I3D_REPO_ID      = "flateon/FVD-I3D-torchscript"
_I3D_FILENAME     = "i3d_torchscript.pt"
_I3D_MIN_FRAMES   = 10     # I3D hard minimum
_MIN_VIDEOS_WARN  = 256    # below this FVD is unreliable
_REAL_FEAT_RELPATH = "fvd/real_features.pt"  # relative to get_cache_dir()


# ---------------------------------------------------------------------------
# I3D helpers  (adapted from FastVideo benchmarks/fvd/i3d_model.py)
# ---------------------------------------------------------------------------

def _load_i3d(device: torch.device) -> torch.nn.Module:
    """Download I3D TorchScript from HuggingFace Hub and load it.

    Uses :func:`fastvideo.eval.models.ensure_checkpoint` so the download
    is filelock-safe across threads, processes, and SLURM ranks.
    """
    from fastvideo.eval.models import ensure_checkpoint
    path = ensure_checkpoint(_I3D_FILENAME, source=_I3D_REPO_ID, filename=_I3D_FILENAME)
    model = torch.jit.load(path, map_location=device)
    model.eval()
    return model


def _preprocess(video: torch.Tensor) -> torch.Tensor:
    """(B, T, C, H, W) float [0, 1] → (B, C, T, 224, 224) float [-1, 1].

    Matches I3D preprocessing from FastVideo benchmarks/fvd/feature_extractors.py.
    """
    B, T, C, H, W = video.shape
    if T < _I3D_MIN_FRAMES:
        raise ValueError(
            f"I3D requires at least {_I3D_MIN_FRAMES} frames, got {T}. "
            "Increase num_frames or use a longer video."
        )
    if H != 224 or W != 224:
        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, size=(224, 224), mode="bilinear", align_corners=False)
        video = video.reshape(B, T, C, 224, 224)
    # Scale [0, 1] → [-1, 1] and permute to (B, C, T, H, W)
    video = video * 2.0 - 1.0
    return video.permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def _extract_features(
    model: torch.nn.Module,
    video: torch.Tensor,      # (B, T, C, H, W) float [0, 1]
    chunk: int,
    device: torch.device,
) -> np.ndarray:
    """Extract I3D features → (B, 400) numpy array, chunked to fit VRAM.

    ``torch.jit.fuser("none")`` disables NVRTC kernel fusion for the I3D
    TorchScript forward pass.  Without it, PyTorch tries to JIT-compile fused
    kernels via ``libnvrtc-builtins``, which is only available on the exact
    CUDA version the binary was built against (e.g. fails on Colab CUDA 12
    when the lib expects CUDA 13).  Disabling fusion has no effect on
    numerical correctness — the I3D model still runs in full precision on GPU.
    """
    parts = []
    with torch.jit.fuser("none"):
        for i in range(0, video.shape[0], chunk):
            batch = _preprocess(video[i : i + chunk].to(device))
            feats = model(batch, rescale=False, resize=False, return_features=True)
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)  # (D,) → (1, D) when I3D squeezes B=1
            parts.append(feats.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Fréchet distance  (adapted from FastVideo benchmarks/fvd/fvd.py)
# ---------------------------------------------------------------------------

def _gaussian_params(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of a feature matrix (N, D).

    ``np.atleast_2d`` guards against a 1-D array (e.g. a single feature
    vector squeezed by I3D or loaded from a stale cache), which would
    cause ``np.cov`` to return a 0-d scalar and break ``sigma.shape[0]``.
    """
    features = np.atleast_2d(features)
    mu    = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    if sigma.ndim == 0:                    # n==1 edge case: variance scalar
        sigma = sigma.reshape(1, 1)
    return mu, sigma


def _frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute Fréchet distance between two Gaussians N(mu1,sigma1) and N(mu2,sigma2)."""
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])

    diff    = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            warnings.warn(
                f"FVD: large imaginary component in sqrtm "
                f"({np.max(np.abs(covmean.imag)):.4f}). Result may be inaccurate.",
                stacklevel=3,
            )
        covmean = covmean.real

    return float(np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

@register("common.fvd")
class FVDMetric(BaseMetric):
    """Fréchet Video Distance (FVD) using I3D features.

    Set-vs-set metric (``is_set_metric=True``). The Evaluator calls
    :meth:`accumulate` once per video and :meth:`finalize` once after
    all videos have been processed.

    For meaningful scores, evaluate over ≥ 256 videos (2048 is the
    standard protocol used in the literature).

    Parameters
    ----------
    cache_path : str, optional
        Where extracted reference (real video) features are cached.
        Built from ``sample["reference"]`` on the first run, reused
        automatically on every subsequent run.  Defaults to
        ``${FASTVIDEO_EVAL_CACHE}/fvd/real_features.pt`` (resolved at
        ``setup()`` time so the env-var can be set after import).
    chunk_size : int
        Videos per I3D forward pass. Reduce if GPU runs OOM.
    """

    name             = "common.fvd"
    is_set_metric    = True
    requires_reference = False   # uses cached real features, not per-sample ref
    higher_is_better = False     # lower FVD = better
    needs_gpu        = True
    dependencies     = ["huggingface_hub", "scipy"]

    def __init__(
        self,
        cache_path: str | None = None,
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        # None → resolved lazily from get_cache_dir() in setup() so that
        # FASTVIDEO_EVAL_CACHE is honoured even if set after import time.
        self._cache_path_arg = cache_path
        self._chunk          = chunk_size
        self._i3d: torch.nn.Module | None = None

        # Accumulated feature buffers — cleared by reset()
        self._gen_features:  list[np.ndarray] = []
        self._real_features: np.ndarray | None = None

    def to(self, device):
        super().to(device)
        if self._i3d is not None:
            self._i3d = self._i3d.to(self.device)
        return self

    def setup(self) -> None:
        if self._i3d is not None:
            return
        from fastvideo.eval.models import get_cache_dir
        # Resolve cache_path now so FASTVIDEO_EVAL_CACHE is read at run-time.
        if self._cache_path_arg is None:
            self.cache_path = str(get_cache_dir() / _REAL_FEAT_RELPATH)
        else:
            self.cache_path = os.path.expanduser(self._cache_path_arg)
        self._i3d = _load_i3d(self.device)
        # Pre-load cached real features if available
        self._real_features = self._load_cache()

    # ------------------------------------------------------------------
    # Set-vs-set protocol
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear generated feature buffer. Called before each evaluation run."""
        self._gen_features = []

    def accumulate(self, sample: dict) -> None:
        """Extract I3D features from one generated video and buffer them.

        If ``sample["reference"]`` is provided and no cache exists yet,
        reference features are extracted and saved to *cache_path*.
        """
        if self._i3d is None:
            self.setup()

        video = sample["video"]           # (T, C, H, W) from evaluator
        if video.dim() == 4:
            video = video.unsqueeze(0)    # → (1, T, C, H, W)

        # Buffer generated features
        feats = _extract_features(self._i3d, video, self._chunk, self.device)
        self._gen_features.append(feats)

        # Build real feature cache on first encounter
        if self._real_features is None:
            self._real_features = self._load_cache()

        if self._real_features is None:
            ref = sample.get("reference")
            if ref is not None:
                if ref.dim() == 4:
                    ref = ref.unsqueeze(0)
                self._real_features = _extract_features(
                    self._i3d, ref, self._chunk, self.device
                )
                self._save_cache(self._real_features)

    def finalize(self) -> MetricResult:
        """Compute FVD from all accumulated generated features vs. real features."""
        if not self._gen_features:
            return MetricResult(
                name=self.name,
                score=None,
                details={"skipped": "No generated videos accumulated before finalize()."},
            )

        if self._real_features is None:
            return MetricResult(
                name=self.name,
                score=None,
                details={
                    "skipped": (
                        "No reference features available. Pass sample['reference'] "
                        "in at least one accumulate() call to build the cache at: "
                        f"{self.cache_path}"
                    )
                },
            )

        all_gen  = np.concatenate(self._gen_features, axis=0)
        n_gen    = len(all_gen)
        n_real   = len(self._real_features)

        if n_gen < _MIN_VIDEOS_WARN or n_real < _MIN_VIDEOS_WARN:
            warnings.warn(
                f"FVD computed with only {n_gen} generated and {n_real} real videos. "
                f"At least {_MIN_VIDEOS_WARN} recommended (standard protocol: 2048). "
                "Score may not be statistically reliable.",
                stacklevel=2,
            )

        mu_gen,  sigma_gen  = _gaussian_params(all_gen)
        mu_real, sigma_real = _gaussian_params(self._real_features)
        fvd = _frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)

        return MetricResult(
            name=self.name,
            score=fvd,
            details={
                "n_generated": n_gen,
                "n_reference": n_real,
            },
        )

    def merge_from(self, other: BaseMetric) -> None:
        """Fold another worker's accumulated features into this one (multi-GPU)."""
        assert isinstance(other, FVDMetric)
        self._gen_features.extend(other._gen_features)
        # Real features are identical across workers (same cache); keep ours.
        if self._real_features is None and other._real_features is not None:
            self._real_features = other._real_features

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> np.ndarray | None:
        if os.path.exists(self.cache_path):
            data = torch.load(self.cache_path, map_location="cpu", weights_only=True)
            arr = np.atleast_2d(data.numpy())  # guard against stale 1-D cache
            return arr
        return None

    def _save_cache(self, features: np.ndarray) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        torch.save(torch.from_numpy(features), self.cache_path)
