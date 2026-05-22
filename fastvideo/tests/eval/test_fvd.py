"""CPU-only regression tests for ``common.fvd``.

No GPU, no model download — everything that needs a backbone is
monkeypatched to a deterministic dummy that returns random features.
GPU-side correctness is exercised in the example script
``examples/inference/eval/eval_fvd.py``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fastvideo.eval import get_metric, list_metrics
from fastvideo.eval.metrics.common.fvd import extractors
from fastvideo.eval.metrics.common.fvd.metric import (
    FVDMetric,
    _default_cache_path,
    _frechet_distance,
    _gaussian_params,
)


# ---------------------------------------------------------------------------
# 1. Registry contract
# ---------------------------------------------------------------------------


def test_common_fvd_is_registered():
    assert "common.fvd" in list_metrics()


def test_fvd_class_attributes():
    cls = FVDMetric
    assert cls.name == "common.fvd"
    assert cls.is_set_metric is True
    assert cls.higher_is_better is False
    assert cls.needs_gpu is True
    assert cls.requires_reference is False  # uses cached ref features, not per-sample
    assert "huggingface_hub" in cls.dependencies
    assert "scipy" in cls.dependencies


# ---------------------------------------------------------------------------
# 2. Extractor factory
# ---------------------------------------------------------------------------


def test_available_extractors_lists_three_choices():
    assert extractors.available_extractors() == ["clip", "i3d", "videomae"]


def test_load_extractor_unknown_raises():
    with pytest.raises(ValueError, match="Unknown FVD extractor"):
        extractors.load_extractor("bogus", torch.device("cpu"))


def test_fvd_constructor_rejects_unknown_extractor():
    with pytest.raises(ValueError, match="Unknown FVD extractor"):
        get_metric("common.fvd", extractor="bogus")


def test_extractor_registry_maps_to_correct_classes():
    # Inspect the factory table directly — instantiating would download weights.
    assert extractors._EXTRACTORS["i3d"] is extractors._I3DExtractor
    assert extractors._EXTRACTORS["clip"] is extractors._CLIPExtractor
    assert extractors._EXTRACTORS["videomae"] is extractors._VideoMAEExtractor


# ---------------------------------------------------------------------------
# 3. Skip path — no reference features available
# ---------------------------------------------------------------------------


class _DummyExtractor:
    """Stand-in for a real extractor: returns deterministic random features.

    Used so CI doesn't pay the cost of downloading I3D / CLIP / VideoMAE
    weights just to exercise the metric's bookkeeping.
    """

    feature_dim = 16

    def __init__(self, device: torch.device, seed: int = 0) -> None:
        self.device = device
        self._rng = np.random.default_rng(seed)

    def to(self, device):
        self.device = device
        return self

    def forward(self, video: torch.Tensor) -> np.ndarray:
        return self._rng.standard_normal((video.shape[0], self.feature_dim)).astype(np.float32)


def _install_dummy_loader(monkeypatch, *, seed: int | None = None) -> None:
    """Route ``extractors.load_extractor`` to a no-download dummy."""

    def factory(name: str, device: torch.device) -> _DummyExtractor:
        s = seed if seed is not None else (hash(name) & 0xFFFF)
        return _DummyExtractor(device, seed=s)

    monkeypatch.setattr(extractors, "load_extractor", factory)


def test_finalize_without_reference_returns_skipped(tmp_path, monkeypatch):
    cache_path = tmp_path / "real_features_i3d.pt"
    metric = get_metric("common.fvd", extractor="i3d", cache_path=str(cache_path))
    metric.to("cpu")
    _install_dummy_loader(monkeypatch)
    metric.setup()
    metric.reset()

    metric.accumulate({"video": torch.zeros(16, 3, 32, 32)})
    result = metric.finalize()

    assert result.score is None
    assert "skipped" in result.details
    assert str(cache_path) in result.details["skipped"]


def test_finalize_with_reference_returns_finite_score(tmp_path, monkeypatch):
    cache_path = tmp_path / "real_features_i3d.pt"
    metric = get_metric("common.fvd", extractor="i3d", cache_path=str(cache_path))
    metric.to("cpu")
    _install_dummy_loader(monkeypatch, seed=42)
    metric.setup()
    metric.reset()

    # First sample carries the reference set (4 ref videos); the rest are gen.
    ref = torch.zeros(4, 16, 3, 32, 32)
    for i in range(5):
        sample = {"video": torch.zeros(16, 3, 32, 32)}
        if i == 0:
            sample["reference"] = ref
        metric.accumulate(sample)

    with pytest.warns(UserWarning, match="At least 256 recommended"):
        result = metric.finalize()
    assert result.score is not None
    assert np.isfinite(result.score)
    assert result.details["extractor"] == "i3d"
    assert result.details["n_generated"] == 5
    assert result.details["n_reference"] == 4
    assert cache_path.exists()


# ---------------------------------------------------------------------------
# 4. Math sanity
# ---------------------------------------------------------------------------


def test_frechet_distance_of_identical_gaussians_is_zero():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((128, 16))
    mu, sigma = _gaussian_params(feats)
    d = _frechet_distance(mu, sigma, mu, sigma)
    assert d == pytest.approx(0.0, abs=1e-6)


def test_gaussian_params_handles_single_sample():
    feats = np.array([[1.0, 2.0, 3.0]])  # n=1 → variance is scalar
    mu, sigma = _gaussian_params(feats)
    assert mu.shape == (3,)
    assert sigma.shape == (1, 1)  # the n=1 reshape kicks in


def test_gaussian_params_atleast_2d_guard():
    """1-D feature vector (e.g. from a stale cache) shouldn't blow up cov."""
    feats = np.array([1.0, 2.0, 3.0])  # 1-D, atleast_2d → (1, 3)
    mu, sigma = _gaussian_params(feats)
    assert mu.shape == (3,)


def test_frechet_distance_translation_invariance():
    """``d(N(μ,Σ), N(μ+v,Σ)) == ||v||^2``."""
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((128, 8))
    mu, sigma = _gaussian_params(feats)
    v = np.arange(8).astype(float)
    d = _frechet_distance(mu, sigma, mu + v, sigma)
    assert d == pytest.approx(float(np.sum(v**2)), rel=1e-4)


# ---------------------------------------------------------------------------
# 5. Cache partitioning by extractor
# ---------------------------------------------------------------------------


def test_default_cache_path_partitions_by_extractor():
    p_i3d = _default_cache_path("i3d")
    p_clip = _default_cache_path("clip")
    p_videomae = _default_cache_path("videomae")
    assert p_i3d.endswith("real_features_i3d.pt")
    assert p_clip.endswith("real_features_clip.pt")
    assert p_videomae.endswith("real_features_videomae.pt")
    assert len({p_i3d, p_clip, p_videomae}) == 3


def test_env_var_overrides_default_cache_path(tmp_path, monkeypatch):
    target = tmp_path / "my_cache.pt"
    monkeypatch.setenv("FASTVIDEO_FVD_REF_FEATURES", str(target))
    metric = get_metric("common.fvd", extractor="clip")
    metric.to("cpu")
    _install_dummy_loader(monkeypatch)
    metric.setup()
    assert metric.cache_path == str(target)


def test_constructor_kwarg_outranks_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTVIDEO_FVD_REF_FEATURES", str(tmp_path / "ignored.pt"))
    explicit = tmp_path / "explicit.pt"
    metric = get_metric("common.fvd", extractor="i3d", cache_path=str(explicit))
    metric.to("cpu")
    _install_dummy_loader(monkeypatch)
    metric.setup()
    assert metric.cache_path == str(explicit)


def test_merge_from_rejects_extractor_mismatch(monkeypatch):
    _install_dummy_loader(monkeypatch)
    a = get_metric("common.fvd", extractor="i3d")
    b = get_metric("common.fvd", extractor="clip")
    with pytest.raises(AssertionError, match="extractor mismatch"):
        a.merge_from(b)



