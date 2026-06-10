"""common.fvd — Fréchet Video Distance.

Measures distributional similarity between generated and reference videos
using a pluggable feature backbone (I3D / CLIP / VideoMAE). Lower score →
generated videos are closer to the real video distribution.

FVD is a dataset-level metric — it compares the distribution of a collection
of videos rather than scoring each video individually. For statistically
reliable results at least 256 videos are recommended; the standard protocol
uses 2048.

This metric follows the set-vs-set protocol (``is_set_metric=True``):

  - :meth:`accumulate` is called once per video to buffer features. Routes
    on ``sample["role"]`` — ``"reference"`` → real buffer, anything else →
    generated buffer.  This is the same convention
    :class:`audio.frechet_distance.FrechetAudioDistanceMetric` uses.
  - :meth:`finalize` is called once after all videos to compute FVD.
  - :meth:`reset`    clears both buffers between evaluation runs.

To score two corpora, build a samples list with
:func:`fastvideo.eval.io.inputs.samples_from` and hand it to
:meth:`Evaluator.evaluate` — the path-friendly form is::

    from fastvideo.eval import create_evaluator, samples_from
    ev = create_evaluator(metrics=["common.fvd"], device="cuda:0")
    result = ev.evaluate(samples=samples_from(
        video="gen/", reference="ref/",
    )).corpus["common.fvd"]

Extractors
----------

* ``i3d`` (default) — Kinetics-400 I3D, the standard FVD feature space used
  in the literature.
* ``clip`` — CLIP ViT-B/32 per-frame embeds, mean-pooled over time. Captures
  semantic / content quality.
* ``videomae`` — VideoMAE-base last-hidden-state, mean-pooled over patches.
  Captures structural / motion quality.

CLIP and VideoMAE are research-grade and not directly comparable to FVD
scores from the literature; use ``i3d`` for paper comparisons.

Reference features
------------------

Two ways to supply reference features:

1. **Stream**: pass ``reference=`` to :func:`samples_from`.  Equal
   cardinality emits paired samples (``sample["video"]`` +
   ``sample["reference"]``); unequal cardinality role-tags the unmatched
   references.  ``accumulate`` routes both shapes correctly.  If
   ``cache_mode="read_write"`` (default) and no cache exists, the
   extracted features are persisted to *cache_path* on :meth:`finalize`.
2. **Cache hit**: a prior run wrote ``cache_path``; :meth:`setup` loads it
   automatically and streamed references are unnecessary.

Streamed references always win over the cache when both are present.

Cache resolution order (first match wins):
  1. ``cache_path=`` constructor kwarg, if set.
  2. ``$FASTVIDEO_FVD_REF_FEATURES``, if set.
  3. ``${FASTVIDEO_EVAL_CACHE}/fvd/real_features_{extractor}.pt`` (default).

The env-var override mirrors ``audio.frechet_distance``'s
``FASTVIDEO_FAD_REF_FEATURES`` pattern.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import scipy.linalg
import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.metrics.common.fvd.extractors import (_BaseExtractor, available_extractors, load_extractor)
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

_MIN_VIDEOS_WARN = 256  # below this FVD is unreliable
_REF_FEATURES_ENV = "FASTVIDEO_FVD_REF_FEATURES"


def _gaussian_params(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of a feature matrix ``(N, D)``.

    ``np.atleast_2d`` guards against a 1-D array (e.g. a single feature
    vector squeezed by a model or loaded from a stale cache), which would
    cause ``np.cov`` to return a 0-d scalar and break ``sigma.shape[0]``.
    """
    features = np.atleast_2d(features)
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    if sigma.ndim == 0:  # n==1 edge case: variance scalar
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

    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            warnings.warn(
                f"FVD: large imaginary component in sqrtm "
                f"({np.max(np.abs(covmean.imag)):.4f}). Result may be inaccurate.",
                stacklevel=3,
            )
        covmean = covmean.real

    return float(np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def _extract_chunked(
    extractor: _BaseExtractor,
    video: torch.Tensor,  # (B, T, C, H, W)
    chunk: int,
) -> np.ndarray:
    """Run *extractor* over *video* in chunks of *chunk* to bound VRAM."""
    if video.shape[0] <= chunk:
        return extractor.forward(video)
    parts = [extractor.forward(video[i:i + chunk]) for i in range(0, video.shape[0], chunk)]
    return np.concatenate(parts, axis=0)


def _default_cache_path(extractor_name: str) -> str:
    from fastvideo.eval.models import get_cache_dir
    return str(get_cache_dir() / "fvd" / f"real_features_{extractor_name}.pt")


@register("common.fvd")
class FVDMetric(BaseMetric):
    """Fréchet Video Distance (FVD) over a pluggable feature backbone.

    Set-vs-set metric (``is_set_metric=True``). The Evaluator calls
    :meth:`accumulate` once per video and :meth:`finalize` once after
    all videos have been processed.

    For meaningful scores, evaluate over ≥ 256 videos (2048 is the
    standard protocol used in the literature).

    Parameters
    ----------
    extractor : str
        Feature backbone name. One of ``"i3d"`` (default), ``"clip"``,
        ``"videomae"``. See module docstring for guidance.
    cache_path : str, optional
        Where extracted reference features are cached. Resolution order:
        constructor kwarg → ``$FASTVIDEO_FVD_REF_FEATURES`` env-var →
        ``${FASTVIDEO_EVAL_CACHE}/fvd/real_features_{extractor}.pt``
        (default, resolved at :meth:`setup` time so the env-var can be set
        after import).
    cache_mode : str
        ``"read_write"`` (default) — read at setup, write at finalize on
        cache miss. ``"read"`` — read at setup, never write.  ``"off"`` —
        ignore the cache entirely.  Auto-write happens only when the cache
        was missing AND reference features streamed in this run.
    chunk_size : int
        Videos per forward pass. Reduce if GPU runs OOM.
    """

    name = "common.fvd"
    is_set_metric = True
    requires_reference = False  # reference is supplied via role="reference" samples or cache
    higher_is_better = False  # lower FVD = better
    needs_gpu = True
    dependencies = ["huggingface_hub", "scipy", "transformers"]

    def __init__(
        self,
        extractor: str = "i3d",
        cache_path: str | None = None,
        cache_mode: str = "read_write",
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if extractor not in available_extractors():
            raise ValueError(f"Unknown FVD extractor '{extractor}'. "
                             f"Available: {available_extractors()}")
        if cache_mode not in {"off", "read", "read_write"}:
            raise ValueError(f"cache_mode must be one of 'off', 'read', 'read_write'; got {cache_mode!r}")
        self._extractor_name = extractor
        self._cache_path_arg = cache_path
        self._cache_mode = cache_mode
        self._chunk = chunk_size
        self._extractor: _BaseExtractor | None = None

        # Accumulated feature buffers — cleared by reset()
        self._gen_buf: list[np.ndarray] = []
        self._real_buf: list[np.ndarray] = []
        # Cache loaded at setup() time; survives reset() (the cached corpus
        # is part of the metric's configured state, not its run state).
        self._cached_real: np.ndarray | None = None

    def to(self, device: str | torch.device) -> FVDMetric:
        super().to(device)
        if self._extractor is not None:
            self._extractor.to(self.device)
        return self

    def setup(self) -> None:
        if self._extractor is not None:
            return
        # Resolve cache_path at runtime so $FASTVIDEO_EVAL_CACHE /
        # $FASTVIDEO_FVD_REF_FEATURES are both honoured even when set after
        # import. Precedence: kwarg > env-var > default.
        if self._cache_path_arg is not None:
            self.cache_path = os.path.expanduser(self._cache_path_arg)
        elif env_path := os.environ.get(_REF_FEATURES_ENV):
            self.cache_path = os.path.expanduser(env_path)
        else:
            self.cache_path = _default_cache_path(self._extractor_name)
        self._extractor = load_extractor(self._extractor_name, self.device)
        if self._cache_mode != "off":
            self._cached_real = self._load_cache()

    # ------------------------------------------------------------------
    # Set-vs-set protocol
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear generated and streamed-reference buffers. Cache survives."""
        self._gen_buf = []
        self._real_buf = []

    def accumulate(self, sample: dict) -> None:
        """Extract features from one video and route by ``sample["role"]``.

        Routing rules:

        * ``role="reference"`` → real buffer; the reference key is ignored.
        * Otherwise → generated buffer.  If ``sample["reference"]`` is
          also present, its features also go into the real buffer — this
          lets a paired samples list (the shape per-sample metrics like
          LPIPS need) feed FVD in the same pass without a second decode.

        Accepts either a raw tensor or a populated :class:`Video` instance
        under either key.
        """
        if self._extractor is None:
            self.setup()
        assert self._extractor is not None  # for type narrowing

        if sample.get("role") == "reference":
            self._real_buf.append(self._extract(sample["video"]))
            return

        self._gen_buf.append(self._extract(sample["video"]))
        if "reference" in sample:
            self._real_buf.append(self._extract(sample["reference"]))

    def _extract(self, video: torch.Tensor) -> np.ndarray:
        """Batchify (4-D → 5-D) and run the chunked extractor forward.

        Pre-condition: ``video`` is already a tensor — :class:`EvalWorker`
        unwraps :class:`Video` instances before any metric sees the sample.
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)  # (T, C, H, W) → (1, T, C, H, W)
        assert self._extractor is not None
        return _extract_chunked(self._extractor, video, self._chunk)

    def finalize(self) -> MetricResult:
        """Compute FVD from buffered generated features vs. reference features.

        Reference source priority: streamed (``_real_buf``) > disk cache
        (``_cached_real``).  On a fresh cache + streamed refs, write the
        accumulated real features back to disk if ``cache_mode="read_write"``.
        """
        if not self._gen_buf:
            return MetricResult(
                name=self.name,
                score=None,
                details={"skipped": "No generated videos accumulated before finalize()."},
            )
        all_gen = np.concatenate(self._gen_buf, axis=0)

        if self._real_buf:
            all_real = np.concatenate(self._real_buf, axis=0)
            ref_source = "streamed"
            # Persist to cache only when we built real features fresh and
            # the cache slot was empty — never silently overwrite an
            # existing cache.
            if self._cache_mode == "read_write" and self._cached_real is None:
                self._save_cache(all_real)
        elif self._cached_real is not None:
            all_real = self._cached_real
            ref_source = "cached"
        else:
            return MetricResult(
                name=self.name,
                score=None,
                details={
                    "skipped": ("No reference features available. Pass role='reference' samples "
                                "(or a paired samples list with 'reference' key) to "
                                f"Evaluator.evaluate(), or pre-build the cache at: {self.cache_path}")
                },
            )

        n_gen = len(all_gen)
        n_real = len(all_real)

        if n_gen < _MIN_VIDEOS_WARN or n_real < _MIN_VIDEOS_WARN:
            warnings.warn(
                f"FVD computed with only {n_gen} generated and {n_real} real videos. "
                f"At least {_MIN_VIDEOS_WARN} recommended (standard protocol: 2048). "
                "Score may not be statistically reliable.",
                stacklevel=2,
            )

        mu_gen, sigma_gen = _gaussian_params(all_gen)
        mu_real, sigma_real = _gaussian_params(all_real)
        fvd = _frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)

        return MetricResult(
            name=self.name,
            score=fvd,
            details={
                "extractor": self._extractor_name,
                "n_generated": n_gen,
                "n_reference": n_real,
                "ref_source": ref_source,
            },
        )

    def merge_from(self, other: BaseMetric) -> None:
        """Fold another worker's accumulated features into this one (multi-GPU).

        Both gen and ref buffers concatenate.  The disk cache is the same
        across workers, so we only adopt *other*'s ``_cached_real`` if our
        own slot is empty (defensive — should always already match).
        """
        assert isinstance(other, FVDMetric)
        assert other._extractor_name == self._extractor_name, (
            f"merge_from extractor mismatch: {other._extractor_name!r} vs {self._extractor_name!r}")
        self._gen_buf.extend(other._gen_buf)
        self._real_buf.extend(other._real_buf)
        if self._cached_real is None and other._cached_real is not None:
            self._cached_real = other._cached_real

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> np.ndarray | None:
        if os.path.exists(self.cache_path):
            data = torch.load(self.cache_path, map_location="cpu", weights_only=True)
            return np.atleast_2d(data.numpy())  # guard against stale 1-D cache
        return None

    def _save_cache(self, features: np.ndarray) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        torch.save(torch.from_numpy(features), self.cache_path)
