"""Identity-pair / identity-set invariants.

For paired and set-vs-set metrics, duplicating the asset into ``gen/``
and ``ref/`` should yield the perfect-match value (SSIM=1, PSNR≳99,
LPIPS=0, FVD=0, FAD=0, KL=0, gt_optical_flow=0). No reference JSON
needed — the math itself is the gold.
"""
from __future__ import annotations

import pytest

from fastvideo.tests.eval.conftest import INVARIANT_SPEC

REQUIRED_GPUS = 1


@pytest.mark.parametrize("metric,expected,tol,where", INVARIANT_SPEC)
def test_metric_identity_invariant(metric: str, expected: float, tol: float | None,
                                   where: str, invariant_results) -> None:
    if where == "corpus":
        result = invariant_results.corpus.get(metric)
    else:
        result = invariant_results[0].get(metric)

    if result is None:
        pytest.skip(f"{metric} not registered on this Evaluator (missing dep)")
    if result.score is None:
        pytest.skip(f"fastvideo skipped {metric}: {result.details.get('skipped', 'no score')}")

    if tol is None:
        # threshold form (e.g. PSNR ≥ 99); for identical inputs PSNR
        # approaches infinity but float epsilon caps it at ~100.
        assert result.score >= expected, (
            f"{metric}: {result.score:.4f} below threshold {expected}"
        )
    else:
        delta = abs(result.score - expected)
        assert delta <= tol, (
            f"{metric}: got {result.score:.6f}, expected ≈ {expected} (|Δ|={delta:.6f} > tol={tol})"
        )
