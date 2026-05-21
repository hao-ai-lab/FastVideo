"""Smoke tests for the metric registry surface.

These exercise the public ``fastvideo.eval`` API only:
``list_metrics``, ``get_metric``, and the group-resolution logic that
``create_evaluator(metrics="vbench")`` uses.
"""
from __future__ import annotations

import pytest

from fastvideo.eval import create_evaluator, get_metric, list_metrics


# Metrics that should ship out of the box. Keep this list short so it's
# resistant to future renames — we want to catch wholesale registry
# breakage, not chase every individual sub-metric rename.
_CORE_METRICS = (
    "common.psnr",
    "common.ssim",
    "common.lpips",
    "optical_flow.gt_optical_flow",
    "optical_flow.synthetic_optical_flow",
    "physics_iq",
    "vbench.aesthetic_quality",
    "vbench.subject_consistency",
)


def test_list_metrics_returns_sorted_unique():
    names = list_metrics()
    assert len(names) > 0
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_core_metrics_are_registered():
    names = set(list_metrics())
    for m in _CORE_METRICS:
        assert m in names, f"expected core metric {m!r} to be registered"


def test_get_metric_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="Unknown metric"):
        get_metric("nope.nonsense")


def test_create_evaluator_resolves_group_prefix():
    """``metrics="<group>"`` should expand to every ``<group>.*`` sub-metric.

    Use the ``physics_iq`` group because it has multiple sub-metrics and
    none of them load model weights — the group-resolution behavior is
    what we're testing, not metric setup."""
    ev = create_evaluator(metrics="physics_iq", device="cpu")
    try:
        names = ev.metric_names
        assert all(n.startswith("physics_iq.") for n in names)
        # Don't pin an exact count — sub-metrics may be added later.
        assert "physics_iq.spatial_iou" in names
        assert "physics_iq.spatiotemporal_iou" in names
        assert "physics_iq.mse" in names
    finally:
        ev.shutdown()


def test_create_evaluator_with_explicit_list_preserves_order():
    ev = create_evaluator(metrics=["common.psnr", "common.ssim"], device="cpu")
    try:
        assert ev.metric_names == ["common.psnr", "common.ssim"]
    finally:
        ev.shutdown()


def test_create_evaluator_unknown_metric_raises():
    with pytest.raises(KeyError, match="Unknown metric"):
        create_evaluator(metrics=["common.nope"], device="cpu")
