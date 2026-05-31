"""End-to-end test: prompt dataset → Evaluator.

Mirrors the canonical user flow:

    ds = get_dataset("vbench", dimensions=[...])
    ev = create_evaluator(metrics=[...], device=...)
    for row in ds:
        video = my_generator(row["prompt"])
        scores = ev.evaluate(video=video, **row)

We don't actually generate videos — that would pull in a diffusion
model. Instead we synthesize a reproducible random tensor per row, so
the test exercises the dataset-iteration → evaluator-call wiring
without depending on any model weights.
"""
from __future__ import annotations

import pytest
import torch

from fastvideo.eval import MetricResult, create_evaluator
from fastvideo.eval.datasets import get_dataset, list_datasets


_T, _C, _H, _W = 6, 3, 32, 32


def _synth_video(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(_T, _C, _H, _W, generator=g)


def test_vbench_dataset_is_registered():
    assert "vbench" in list_datasets()


def test_vbench_dataset_rows_drop_into_evaluator():
    """Every row from the corpus must be a kwargs-friendly dict for
    ``Evaluator.evaluate``: extra keys flow through without breaking the
    metric, and the metric returns a well-formed ``MetricResult``."""
    ds = get_dataset("vbench", dimensions=["color"])
    rows = [ds[i] for i in range(3)]
    assert all("prompt" in r for r in rows)
    assert all("auxiliary_info" in r for r in rows)

    ev = create_evaluator(metrics=["common.psnr"], device="cpu")
    try:
        for i, row in enumerate(rows):
            video = _synth_video(seed=i)
            reference = _synth_video(seed=100 + i)
            # Real-world shape: pass dataset row through verbatim, plus
            # the generated/reference tensors. Extra dataset fields
            # ('prompt', 'n_samples', 'dimensions', 'auxiliary_info') are
            # ignored by common.psnr — that's the contract we want to pin.
            scores = ev.evaluate(video=video, reference=reference, **row)
            mr = scores["common.psnr"]
            assert isinstance(mr, MetricResult)
            assert mr.name == "common.psnr"
            assert mr.score is not None
    finally:
        ev.shutdown()


def test_vbench_dataset_full_corpus_iteration():
    """Iterating the whole dataset should be cheap (no evaluator calls).
    This guards against a future refactor that accidentally makes
    ``__iter__`` do real work."""
    ds = get_dataset("vbench")
    rows = list(ds)
    assert len(rows) == len(ds)
    assert all(isinstance(r, dict) for r in rows)


def test_dataset_samples_form_through_evaluator():
    """``Evaluator.evaluate(samples=[...])`` is the canonical batched
    entry point; verify it works when the per-row dicts come from a
    dataset (kwargs form) rather than being hand-built in the test."""
    ds = get_dataset("vbench", dimensions=["color"])
    rows = [ds[i] for i in range(3)]

    samples = []
    for i, row in enumerate(rows):
        samples.append({
            "video": _synth_video(seed=i),
            "reference": _synth_video(seed=100 + i),
            **row,
        })

    ev = create_evaluator(metrics=["common.psnr"], device="cpu")
    try:
        out = ev.evaluate(samples=samples)
    finally:
        ev.shutdown()

    assert len(out) == len(rows)
    for row in out:
        assert "common.psnr" in row
        assert row["common.psnr"].score is not None
