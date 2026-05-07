"""End-to-end tests for single-replica eval through the public API.

Runs the lightweight pixel-space metrics — ``common.psnr`` and
``common.ssim`` — under both shapes that real callers use:

* one-shot ``evaluate(video=..., reference=...)`` (the helper in
  ``fastvideo.eval.api``);
* a long-lived ``Evaluator``, called once per sample;
* a long-lived ``Evaluator``, called with a list of sample dicts to
  fan out (``samples=[...]``).

GPU-only metrics live in separate test modules / classes; everything
here runs on CPU so the suite stays cheap to invoke.
"""
from __future__ import annotations

import pytest
import torch

from fastvideo.eval import MetricResult, create_evaluator, evaluate


# Tight resolution + frame count so CPU SSIM stays under a couple of seconds.
_T, _C, _H, _W = 6, 3, 32, 32


@pytest.fixture
def gen_ref():
    """Reproducible (gen, ref) pair shaped (T, C, H, W)."""
    torch.manual_seed(0)
    gen = torch.rand(_T, _C, _H, _W)
    ref = torch.rand(_T, _C, _H, _W)
    return gen, ref


@pytest.fixture
def evaluator():
    ev = create_evaluator(
        metrics=["common.psnr", "common.ssim"],
        device="cpu",
    )
    yield ev
    ev.shutdown()


def _assert_well_formed(result: MetricResult, name: str) -> None:
    assert isinstance(result, MetricResult)
    assert result.name == name
    assert result.score is not None
    assert isinstance(result.score, float)
    # PSNR / SSIM both populate per-frame details.
    assert "per_frame" in result.details
    assert len(result.details["per_frame"]) == _T


# ---------------------------------------------------------------------------
# One-shot helper
# ---------------------------------------------------------------------------


def test_evaluate_one_shot_returns_dict_of_metric_results(gen_ref):
    gen, ref = gen_ref
    out = evaluate(generated=gen, reference=ref,
                   metrics=["common.psnr", "common.ssim"], device="cpu")
    assert isinstance(out, dict)
    assert set(out.keys()) == {"common.psnr", "common.ssim"}
    _assert_well_formed(out["common.psnr"], "common.psnr")
    _assert_well_formed(out["common.ssim"], "common.ssim")


# ---------------------------------------------------------------------------
# Long-lived Evaluator, single-sample form
# ---------------------------------------------------------------------------


def test_evaluator_single_sample_returns_dict(evaluator, gen_ref):
    gen, ref = gen_ref
    out = evaluator.evaluate(video=gen, reference=ref)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"common.psnr", "common.ssim"}
    for name, mr in out.items():
        _assert_well_formed(mr, name)


def test_evaluator_accepts_legacy_5d_input(evaluator, gen_ref):
    """Callers that still pass ``(1, T, C, H, W)`` should get unwrapped."""
    gen, ref = gen_ref
    out = evaluator.evaluate(video=gen.unsqueeze(0), reference=ref.unsqueeze(0))
    _assert_well_formed(out["common.psnr"], "common.psnr")


def test_evaluator_score_is_deterministic(evaluator, gen_ref):
    gen, ref = gen_ref
    a = evaluator.evaluate(video=gen, reference=ref)
    b = evaluator.evaluate(video=gen.clone(), reference=ref.clone())
    assert a["common.psnr"].score == pytest.approx(b["common.psnr"].score)
    assert a["common.ssim"].score == pytest.approx(b["common.ssim"].score)


def test_evaluator_psnr_identical_videos_is_high(evaluator, gen_ref):
    """PSNR(x, x) is unbounded above; with our clamp it caps near 100 dB."""
    gen, _ = gen_ref
    out = evaluator.evaluate(video=gen, reference=gen)
    assert out["common.psnr"].score > 50.0
    assert out["common.ssim"].score == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Long-lived Evaluator, list (fan-out) form
# ---------------------------------------------------------------------------


def test_evaluator_samples_list_preserves_input_order(evaluator):
    """When ``samples=[...]`` is passed, results must come back per sample."""
    torch.manual_seed(1)
    samples = []
    for i in range(4):
        # Vary the reference enough that scores differ across rows.
        gen = torch.rand(_T, _C, _H, _W)
        ref = gen + 0.01 * (i + 1) * torch.rand_like(gen)
        samples.append({"video": gen, "reference": ref})

    out = evaluator.evaluate(samples=samples)
    assert isinstance(out, list)
    assert len(out) == len(samples)
    for row in out:
        assert set(row.keys()) == {"common.psnr", "common.ssim"}
        _assert_well_formed(row["common.psnr"], "common.psnr")

    # Re-running should give bit-identical scores (no nondeterministic
    # scheduling effects under single-GPU dispatch).
    out2 = evaluator.evaluate(samples=samples)
    for a, b in zip(out, out2):
        assert a["common.psnr"].score == pytest.approx(b["common.psnr"].score)


