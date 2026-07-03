"""Shared fixtures for the eval regression suite.

Two session fixtures keep model loads off the per-test path:

* ``gold_results`` — runs every reference-backed metric once via the
  top-level :func:`samples_from` + :meth:`Evaluator.evaluate` API and
  caches the per-sample dict for the asset.
* ``invariant_results`` — runs every identity-invariant metric on
  N duplicate copies of the asset (paired-and-corpus shape).

Both use ``skip_missing_deps=True`` so a slim test venv missing one
metric's deps drops only that metric, never the whole run.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from fastvideo.eval import EvalResults, create_evaluator, samples_from

ASSET_DIR = Path(__file__).resolve().parent / "asset"
ASSET_VIDEO = ASSET_DIR / "ltx2.mp4"
ASSET_AUDIO = ASSET_DIR / "ltx2.wav"
ASSET_META = ASSET_DIR / "ltx2.metadata.json"
REF_DIR = Path(__file__).resolve().parent / "reference_scores"

N_DUP = 4  # duplicates for set-metric invariants (fvd/fad need ≥2)


def _all_reference_files() -> list[Path]:
    return sorted(REF_DIR.glob("*.json"))


def _reference_metric_names() -> list[str]:
    return [json.loads(p.read_text())["metric"] for p in _all_reference_files()]


# Identity invariants — both fastvideo and the math itself give the
# perfect-match value on duplicate (gen, ref). No reference JSON needed.
INVARIANT_SPEC = [
    # (metric, expected, abs_tolerance, location)
    ("common.ssim",                  1.0, 1e-3, "per_sample"),
    ("common.psnr",                 99.0, None, "per_sample"),  # ≥ threshold (identical → ~inf, capped by floats)
    ("common.lpips",                 0.0, 1e-2, "per_sample"),
    ("optical_flow.gt_optical_flow", 0.0, 1e-2, "per_sample"),
    ("common.fvd",                   0.0, 1.0,  "corpus"),
    ("audio.frechet_distance",       0.0, 1.0,  "corpus"),
    ("audio.kl_divergence",          0.0, 1e-3, "per_sample"),
]


def load_reference(metric: str) -> dict:
    return json.loads((REF_DIR / f"{metric}.json").read_text())


@pytest.fixture(scope="session")
def asset_meta() -> dict:
    return json.loads(ASSET_META.read_text())


@pytest.fixture(scope="session")
def gold_results(asset_meta) -> dict:
    """Top-level API run on the single regression asset.

    Equivalent to four lines in user code::

        ev = create_evaluator(metrics=[...], skip_missing_deps=True)
        samples = samples_from(video=ASSET, audio=ASSET_WAV,
                               text_prompt=..., fps=..., extras={...})
        return ev.evaluate(samples=samples)[0]
    """
    metrics = _reference_metric_names()
    ev = create_evaluator(metrics=metrics, skip_missing_deps=True)
    samples = samples_from(
        video=ASSET_VIDEO,
        audio=ASSET_AUDIO,
        text_prompt=asset_meta["text_prompt"],
        fps=asset_meta["fps"],
        extras={
            "reference_text": asset_meta["reference_text"],
            "video_path": str(ASSET_VIDEO),
            "language": asset_meta["language"],
        },
    )
    results: EvalResults = ev.evaluate(samples=samples)
    yield results[0]
    ev.shutdown()


@pytest.fixture(scope="session")
def invariant_results(asset_meta) -> EvalResults:
    """Run identity-invariant metrics on N duplicate copies of the asset.

    Same top-level shape — five lines.  ``extract_audio`` pulls a ``.wav``
    next to each copy so audio set-metrics
    (``audio.frechet_distance``, ``audio.kl_divergence``) see paired
    audio without us pre-staging it.
    """
    tmp = Path(tempfile.mkdtemp(prefix="eval_regression_inv_"))
    try:
        gen = tmp / "gen"; ref = tmp / "ref"
        gen.mkdir(); ref.mkdir()
        for i in range(N_DUP):
            shutil.copyfile(ASSET_VIDEO, gen / f"clip_{i:03d}.mp4")
            shutil.copyfile(ASSET_VIDEO, ref / f"clip_{i:03d}.mp4")
        ev = create_evaluator(
            metrics=[m for m, *_ in INVARIANT_SPEC],
            skip_missing_deps=True,
        )
        samples = samples_from(
            video=gen, reference=ref,
            text_prompt=asset_meta["text_prompt"],
            fps=asset_meta["fps"],
            extract_audio=tmp / "audio_cache",
        )
        yield ev.evaluate(samples=samples)
        ev.shutdown()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
