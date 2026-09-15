"""CPU-only contract tests for the VQeval adapter.

These tests never download or initialize CLIP, DINOv2, or pyiqa weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastvideo.eval.metrics.vqeval.metric import VQevalCompositeMetric, _sample_indices
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
