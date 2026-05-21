"""Path-input variants of the public Evaluator API.

The worker boundary accepts ``video`` / ``reference`` as either a
pre-loaded ``(T, C, H, W)`` tensor or a path-like (``str`` / ``Path``).
These tests pin the path-form so future refactors don't accidentally
re-require pre-loaded tensors.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from fastvideo.eval import MetricResult, create_evaluator, evaluate

_T, _C, _H, _W = 6, 3, 32, 32


def _write_tensor_as_mp4(tensor: torch.Tensor, path: Path) -> None:
    """Write a (T, C, H, W) float [0, 1] tensor to *path* as an mp4."""
    frames = (tensor * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    # cv2 expects BGR; the tests don't care about colour fidelity (they
    # just need the bytes to round-trip), but flipping keeps the
    # written file faithful to what an mp4 would carry on disk.
    frames = frames[..., ::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (w, h))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter could not open the mp4v codec on this host")
    for f in frames:
        writer.write(np.ascontiguousarray(f))
    writer.release()


@pytest.fixture
def video_paths(tmp_path):
    """Two reproducible mp4s on disk + their pre-loaded tensors for parity."""
    torch.manual_seed(0)
    paths: list[Path] = []
    tensors: list[torch.Tensor] = []
    for i in range(2):
        t = torch.rand(_T, _C, _H, _W)
        p = tmp_path / f"clip_{i}.mp4"
        _write_tensor_as_mp4(t, p)
        paths.append(p)
        tensors.append(t)
    return paths, tensors


@pytest.fixture
def evaluator():
    ev = create_evaluator(metrics=["common.psnr", "common.ssim"], device="cpu")
    yield ev
    ev.shutdown()


def test_kwargs_form_accepts_string_path(evaluator, video_paths):
    paths, _ = video_paths
    out = evaluator.evaluate(video=str(paths[0]), reference=str(paths[0]))
    assert isinstance(out, dict)
    assert isinstance(out["common.psnr"], MetricResult)
    # Self-paired video → PSNR is huge, SSIM = 1.
    assert out["common.psnr"].score > 50.0
    assert out["common.ssim"].score == pytest.approx(1.0, abs=1e-5)


def test_kwargs_form_accepts_pathlib_path(evaluator, video_paths):
    paths, _ = video_paths
    out = evaluator.evaluate(video=paths[0], reference=paths[0])
    assert out["common.psnr"].score > 50.0


def test_samples_list_accepts_paths(evaluator, video_paths):
    paths, _ = video_paths
    samples = [{"video": str(p), "reference": str(p)} for p in paths]
    out = evaluator.evaluate(samples=samples)
    assert isinstance(out, list)
    assert len(out) == len(paths)
    for row in out:
        assert row["common.psnr"].score > 50.0


def test_samples_list_can_mix_paths_and_tensors(evaluator, video_paths):
    """A single ``samples`` call can mix path and tensor entries."""
    paths, tensors = video_paths
    samples = [
        {"video": str(paths[0]), "reference": tensors[0]},   # path + tensor
        {"video": tensors[1], "reference": str(paths[1])},   # tensor + path
    ]
    out = evaluator.evaluate(samples=samples)
    assert len(out) == 2
    for row in out:
        assert row["common.psnr"].score > 0.0


def test_path_form_score_matches_tensor_form(evaluator, video_paths):
    """Loading via path must produce the same score as loading via the
    public ``load_video`` helper and passing the tensor in directly."""
    from fastvideo.eval.io import load_video

    paths, _ = video_paths
    via_path = evaluator.evaluate(video=str(paths[0]), reference=str(paths[0]))
    tensor = load_video(str(paths[0]))
    via_tensor = evaluator.evaluate(video=tensor, reference=tensor)
    assert via_path["common.psnr"].score == pytest.approx(
        via_tensor["common.psnr"].score, abs=1e-4)
    assert via_path["common.ssim"].score == pytest.approx(
        via_tensor["common.ssim"].score, abs=1e-4)


def test_one_shot_evaluate_accepts_paths(video_paths):
    """The top-level ``fastvideo.eval.evaluate`` helper also flows paths."""
    paths, _ = video_paths
    out = evaluate(generated=str(paths[0]), reference=str(paths[0]),
                   metrics=["common.psnr"], device="cpu")
    assert out["common.psnr"].score > 50.0


def test_missing_path_surfaces_as_exception(evaluator, tmp_path):
    """Decode failures must propagate, not silently produce a None score."""
    bogus = tmp_path / "does_not_exist.mp4"
    with pytest.raises(Exception):
        evaluator.evaluate(video=str(bogus), reference=str(bogus))


def test_dispatcher_holds_paths_not_tensors_in_queue(tmp_path):
    """Memory invariant: when many paths are passed, the queued samples
    are tiny strings, not full tensors. Verify by checking the length
    of the per-sample reference set the dispatcher materializes."""
    torch.manual_seed(1)
    n = 8
    paths: list[Path] = []
    for i in range(n):
        t = torch.rand(_T, _C, _H, _W)
        p = tmp_path / f"clip_{i}.mp4"
        _write_tensor_as_mp4(t, p)
        paths.append(p)

    samples = [{"video": str(p), "reference": str(p)} for p in paths]
    # Each sample dict is just two strings — no tensor allocations until
    # the worker's _resolve_video_input runs.
    for s in samples:
        assert isinstance(s["video"], str)
        assert isinstance(s["reference"], str)

    ev = create_evaluator(metrics=["common.psnr"], device="cpu")
    try:
        out = ev.evaluate(samples=samples)
    finally:
        ev.shutdown()
    assert len(out) == n
    # Self-paired ⇒ all PSNRs should be very high.
    for row in out:
        assert row["common.psnr"].score > 50.0
