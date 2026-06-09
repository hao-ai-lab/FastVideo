"""Score-regression tests against reference values from each metric's upstream repo.

Each reference JSON under ``reference_scores/`` was produced by running
the canonical upstream implementation against the asset video under
``asset/`` (provenance recorded in each JSON). This test asserts that
FastVideo's in-process eval, driven through the top-level
``samples_from`` + ``Evaluator.evaluate`` API, lands within tolerance.
"""
from __future__ import annotations

import math

import pytest

from fastvideo.tests.eval.conftest import _reference_metric_names, load_reference

REQUIRED_GPUS = 1


# Per-metric absolute tolerances. Tight for closed-form math, looser for
# anything that runs CLIP/DINO/ViCLIP/VLM under bf16 or cuDNN.
TOLERANCE: dict[str, float] = {
    # WER tolerates ~1 word/clip drift: reference runs openai-whisper with an
    # explicit language hint; fastvideo runs transformers WhisperForConditionalGeneration
    # without one (auto-detect). Same model weights, different decode paths.
    "audio.wer":                       0.2,
    "audio.desync":                    1e-3,
    "audio.clap_score":                1e-2,
    "audio.audiobox_aesthetics":       1e-2,
    "audio.imagebind_score":           1e-2,
    "videoscore2":                     1e-1,  # greedy + bf16 reduction order
    "vbench.aesthetic_quality":        1e-2,
    "vbench.background_consistency":   1e-2,
    "vbench.dynamic_degree":           0.0,   # binary 0/1 — must match exactly
    "vbench.imaging_quality":          1e-2,
    "vbench.motion_smoothness":        1e-2,
    "vbench.overall_consistency":      1e-2,
    "vbench.subject_consistency":      1e-2,
    "vbench.temporal_flickering":      1e-2,
    "vbench.temporal_style":           1e-2,
}


@pytest.mark.parametrize("metric_name", _reference_metric_names())
def test_metric_score_regression(metric_name: str, gold_results: dict) -> None:
    reference = load_reference(metric_name)
    if reference["score"] is None:
        pytest.skip(reference["details"].get("skipped", "deferred"))

    result = gold_results.get(metric_name)
    if result is None:
        pytest.skip(f"{metric_name} not in registered metrics this run "
                    f"(missing dep or filtered)")
    if result.score is None:
        pytest.skip(f"fastvideo skipped {metric_name}: "
                    f"{result.details.get('skipped', 'no score')}")

    tol = TOLERANCE.get(metric_name)
    if tol is None:
        raise AssertionError(f"no tolerance configured for {metric_name}")

    delta = abs(result.score - reference["score"])
    assert math.isfinite(result.score), f"{metric_name} returned non-finite score"
    assert delta <= tol, (
        f"{metric_name}: fastvideo={result.score:.6f} vs reference={reference['score']:.6f} "
        f"(|Δ|={delta:.6f} > tol={tol})"
    )
