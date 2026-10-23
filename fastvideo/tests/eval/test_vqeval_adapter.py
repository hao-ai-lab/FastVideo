"""CPU-only contract tests for the VQeval adapter.

These tests never download or initialize CLIP, DINOv2, or pyiqa weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastvideo.eval.metrics.vqeval.metric import (
    VQevalCompositeMetric,
    _FALLBACK_MIN_SAMPLE_FPS,
    _FALLBACK_SAMPLE_ALL_THRESHOLD_SEC,
    _UPSTREAM_COMMIT,
    _sample_indices,
)
from fastvideo.eval.registry import _install_hint


@dataclass
class _FakeMeta:
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str
    has_audio: bool


class _FakeVideoData:

    def __init__(self, *, meta, frames, frame_indices, frames_rgb):
        self.meta = meta
        self.frames = frames
        self.frame_indices = frame_indices
        self.frames_rgb = frames_rgb


class _FakeConfig:

    def __init__(self, *, video_path, prompt, device):
        self.video_path = video_path
        self.prompt = prompt
        self.device = device

    def get_active_dimensions(self):
        return ["spatial_quality", "loop_quality"]

    def get_effective_weights(self):
        return {"spatial_quality": 0.6, "loop_quality": 0.4}


class _FakeEvaluator:

    def __init__(self, *, config, model_registry, score):
        self.config = config
        self.model_registry = model_registry
        self.score = score

    def is_applicable(self, video):
        return True

    def evaluate(self, video):
        return SimpleNamespace(score=self.score, verdict="good", metrics={"raw": self.score / 100.0})


def _evaluator(score):

    class BoundFakeEvaluator(_FakeEvaluator):

        def __init__(self, **kwargs):
            super().__init__(score=score, **kwargs)

    return BoundFakeEvaluator


def _mocked_metric() -> VQevalCompositeMetric:
    metric = VQevalCompositeMetric()
    metric._registry = object()
    metric._config_cls = _FakeConfig
    metric._video_data_cls = _FakeVideoData
    metric._video_meta_cls = _FakeMeta
    metric._evaluator_classes = {
        "spatial_quality": _evaluator(80.0),
        "loop_quality": _evaluator(40.0),
    }
    return metric


def test_sample_indices_matches_upstream_policy():
    assert _sample_indices(total_frames=50, fps=10.0) == list(range(50))
    assert _sample_indices(total_frames=60, fps=10.0) == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 59]


def test_missing_upstream_hint_includes_submodule_and_extra():
    hint = _install_hint("vqeval.composite", "vqeval")
    assert "git submodule update --init fastvideo/third_party/eval/vqeval" in hint
    assert ".[eval-vqeval]" in hint


def test_composite_preserves_dimensions_and_sampling_metadata():
    metric = _mocked_metric()
    video = torch.zeros(60, 3, 2, 3)
    video[:, 0] = 1.0

    result = metric.compute({
        "video": video,
        "video_path": "fixture.mp4",
        "fps": 10.0,
        "text_prompt": "a red field",
    })

    assert result.name == "vqeval.composite"
    assert result.score == pytest.approx(64.0)
    assert set(result.details["dimensions"]) == {"spatial_quality", "loop_quality"}
    assert result.details["source_frames"] == 60
    assert result.details["sampled_frames"] == 13
    assert result.details["fps"] == 10.0


def test_tensor_conversion_is_uint8_bgr_and_keeps_rgb_cache():
    metric = _mocked_metric()
    video = torch.zeros(2, 3, 1, 1)
    video[:, 0] = 1.0

    upstream, indices = metric._to_upstream_video(video, fps=1.0, source="fixture.mp4")

    assert indices == [0, 1]
    assert upstream.frames.dtype == np.uint8
    assert upstream.frames_rgb.dtype == np.uint8
    assert upstream.frames[0, 0, 0].tolist() == [0, 0, 255]
    assert upstream.frames_rgb[0, 0, 0].tolist() == [255, 0, 0]
    assert upstream.meta.width == 1
    assert upstream.meta.height == 1


@pytest.mark.parametrize(
    "sample,reason",
    [
        ({}, "missing video"),
        ({"video": torch.zeros(2, 3, 2, 2)}, "missing fps"),
        ({"video": torch.zeros(2, 3, 2, 2), "fps": 0}, "fps must be greater than zero"),
        ({"video": torch.zeros(1, 3, 2, 2), "fps": 8}, "at least two RGB frames"),
    ],
)
def test_invalid_input_skips_cleanly(sample, reason):
    result = _mocked_metric().compute(sample)
    assert result.score is None
    assert reason in result.details["skipped"]


def test_setup_is_lazy_and_does_not_load_models():
    pytest.importorskip("cv2")
    pytest.importorskip("vqeval")  # git submodule; skip when not checked out
    metric = VQevalCompositeMetric().to("cpu")
    metric.setup()

    assert metric._registry._models == {}
    assert metric._registry._processors == {}
    assert set(metric._evaluator_classes) == {
        "spatial_quality",
        "temporal_coherence",
        "loop_quality",
        "artifact_detection",
        "dynamic_quality",
        "text_alignment",
    }


def test_upstream_loop_dimension_separates_repetition_without_model_downloads():
    pytest.importorskip("cv2")
    pytest.importorskip("vqeval")  # git submodule; skip when not checked out
    metric = VQevalCompositeMetric().to("cpu")
    metric.setup()

    class FakeRegistry:

        def __init__(self, embeddings):
            self.embeddings = embeddings

        def compute_clip_image_embeddings(self, tensors):
            return self.embeddings

        def compute_optical_flow(self, frame1, frame2):
            return torch.zeros(1, 2, 4, 4)

    def loop_score(embeddings):
        video = torch.zeros(len(embeddings), 3, 4, 4)
        upstream, _ = metric._to_upstream_video(video, fps=8.0, source="synthetic")
        config = metric._config_cls(device="cpu")
        evaluator = metric._evaluator_classes["loop_quality"](
            config=config,
            model_registry=FakeRegistry(embeddings),
        )
        return evaluator.evaluate(upstream).score

    n_frames, embedding_dim = 24, 16
    static = torch.zeros(n_frames, embedding_dim)
    static[:, 0] = 1
    periodic = torch.eye(4).repeat(6, 1)
    generator = torch.Generator().manual_seed(7)
    unique = torch.randn(n_frames, embedding_dim, generator=generator)
    unique = unique / unique.norm(dim=-1, keepdim=True)

    assert loop_score(static) < loop_score(unique)
    assert loop_score(periodic) < loop_score(unique)


def _raising_evaluator():
    class _RaisingEvaluator(_FakeEvaluator):

        def __init__(self, **kwargs):
            super().__init__(score=0.0, **kwargs)

        def evaluate(self, video):
            raise RuntimeError("simulated dimension failure")

    return _RaisingEvaluator


_REAL_DIMENSION_SCORES = {
    "spatial_quality": 80.0,
    "temporal_coherence": 60.0,
    "loop_quality": 40.0,
    "artifact_detection": 20.0,
    "dynamic_quality": 100.0,
    "text_alignment": 50.0,
}


def _real_config_metric(failing=()):
    metric = VQevalCompositeMetric()
    metric._registry = object()
    metric._config_cls = pytest.importorskip("vqeval.core.config").EvalConfig
    metric._video_data_cls = _FakeVideoData
    metric._video_meta_cls = _FakeMeta
    metric._evaluator_classes = {
        name: (_raising_evaluator() if name in failing else _evaluator(score))
        for name, score in _REAL_DIMENSION_SCORES.items()
    }
    return metric


@pytest.mark.parametrize("prompt", [None, "a red field"])
@pytest.mark.parametrize("failing", [(), ("loop_quality",)])
def test_real_evalconfig_weights_renormalize_over_successful_dimensions(prompt, failing):
    """Drive compute() through upstream's real EvalConfig weight wiring.

    The other tests fake get_active_dimensions()/get_effective_weights();
    this asserts the real DEFAULT_WEIGHTS redistribution (text_alignment
    only with a prompt) and that a failed dimension's weight is
    redistributed over the dimensions that succeeded, matching upstream's
    EvalPipeline._compute_composite.
    """
    config_mod = pytest.importorskip("vqeval.core.config")
    metric = _real_config_metric(failing)
    video = torch.zeros(60, 3, 2, 3)
    video[:, 0] = 1.0

    result = metric.compute({
        "video": video,
        "video_path": "fixture.mp4",
        "fps": 10.0,
        "text_prompt": prompt,
    })

    config = config_mod.EvalConfig(video_path="fixture.mp4", prompt=prompt, device="cpu")
    active = config.get_active_dimensions()
    assert ("text_alignment" in active) == (prompt is not None)

    raw = config_mod.DEFAULT_WEIGHTS
    total = sum(raw[d] for d in active)
    assert config.get_effective_weights() == pytest.approx({d: raw[d] / total for d in active})

    assert set(result.details["dimensions"]) == set(active) - set(failing)
    assert set(result.details["errors"]) == set(failing)

    ok = [d for d in active if d not in failing]
    weights = config.get_effective_weights()
    expected = sum(weights[d] * _REAL_DIMENSION_SCORES[d] for d in ok) / sum(weights[d] for d in ok)
    assert result.score == pytest.approx(expected)


def test_fallback_sampling_thresholds_match_upstream_constants():
    config_mod = pytest.importorskip("vqeval.core.config")
    assert _FALLBACK_SAMPLE_ALL_THRESHOLD_SEC == config_mod.SAMPLE_ALL_THRESHOLD_SEC
    assert _FALLBACK_MIN_SAMPLE_FPS == config_mod.MIN_SAMPLE_FPS


def test_upstream_commit_matches_submodule_checkout():
    import subprocess
    from pathlib import Path

    submodule = Path(__file__).resolve().parents[2] / "third_party" / "eval" / "vqeval"
    if not (submodule / ".git").exists():
        pytest.skip("vqeval submodule not checked out")
    try:
        head = subprocess.run(
            ["git", "-C", str(submodule), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git unavailable to verify the submodule commit")
    assert _UPSTREAM_COMMIT == head, (
        "_UPSTREAM_COMMIT must match the pinned submodule checkout so "
        "details['upstream_commit'] provenance stays truthful"
    )
