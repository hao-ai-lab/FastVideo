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
    # videoscore2 is handled separately (per-dimension hard-score check) —
    # see _assert_videoscore2 below.
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


def _assert_videoscore2(result, reference: dict) -> None:
    """Regress VideoScore2 on its per-dimension integer (hard) scores, ±1.

    VideoScore2 is a bf16 VLM judge: its three sub-scores are argmaxed
    integers in 1-5, and they can shift by ±1 across GPU architectures /
    transformers versions (the gold is generated on a different GPU than CI
    runs on). A single ±1 flip moves the combined soft mean by ~0.33, so the
    aggregate score is too brittle to regress directly. The integer hard
    scores are far more portable; assert each is within 1 of the reference,
    which still fails loudly on a parse miss (hard score ``None``) or a ≥2
    point divergence.
    """
    for dim in ("visual_quality_hard", "text_alignment_hard",
                "physical_consistency_hard"):
        ref_h = reference["details"].get(dim)
        got_h = result.details.get(dim)
        assert ref_h is not None, f"reference missing {dim}"
        assert got_h is not None, (
            f"videoscore2 did not parse {dim} "
            f"(raw_output head: {result.details.get('raw_output', '')[:160]!r})")
        assert abs(got_h - ref_h) <= 1, (
            f"videoscore2 {dim}: fastvideo={got_h} vs reference={ref_h} "
            f"(integer sub-score drifted >1 across environments)")


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

    if metric_name == "videoscore2":
        _assert_videoscore2(result, reference)
        return

    tol = TOLERANCE.get(metric_name)
    if tol is None:
        raise AssertionError(f"no tolerance configured for {metric_name}")

    delta = abs(result.score - reference["score"])
    assert math.isfinite(result.score), f"{metric_name} returned non-finite score"
    assert delta <= tol, (
        f"{metric_name}: fastvideo={result.score:.6f} vs reference={reference['score']:.6f} "
        f"(|Δ|={delta:.6f} > tol={tol})"
    )
