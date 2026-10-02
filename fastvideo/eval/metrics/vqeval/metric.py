"""VQeval composite long-video quality metric.

The upstream implementation exposes six evaluators that share CLIP, DINOv2,
pyiqa, and optical-flow state. Registering one FastVideo metric preserves that
sharing and returns the individual dimension results under ``details``.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

_UPSTREAM_COMMIT = "100666e06026b98dfdab39036d9013e02319b479"
_EVALUATOR_MODULES = (
    "spatial_quality",
    "temporal_coherence",
    "loop_quality",
    "artifact_detection",
    "dynamic_quality",
    "text_alignment",
)


@register("vqeval.composite")
class VQevalCompositeMetric(BaseMetric):
    """Run VQeval's active dimensions and return its weighted composite.

    ``sample["fps"]`` is required because VQeval samples long videos based on
    duration and frame rate. ``sample["text_prompt"]`` is optional; when it is
    absent, VQeval omits text alignment and redistributes its weight exactly as
    the upstream pipeline does.
    """

    name = "vqeval.composite"
    requires_reference = False
    higher_is_better = True
    # VQeval owns its model transfers. Keep FastVideo's input tensor on CPU
    # because this adapter must first build VQeval's uint8 RGB/BGR buffers.
    needs_gpu = False
    dependencies = ["vqeval", "cv2", "open_clip", "pyiqa", "mediapipe"]
    backbone = "vqeval_shared"

    def __init__(self) -> None:
        super().__init__()
        self._registry: Any = None
        self._config_cls: Any = None
        self._video_data_cls: Any = None
        self._video_meta_cls: Any = None
        self._evaluator_classes: dict[str, Any] = {}

    def to(self, device: str | torch.device) -> VQevalCompositeMetric:
        target = torch.device(device)
        if self._registry is not None and target != self.device:
            self._registry.clear()
            self._registry = None
        super().to(target)
        return self

    def setup(self) -> None:
        if self._registry is not None:
            return

        for module in _EVALUATOR_MODULES:
            importlib.import_module(f"vqeval.evaluators.{module}")

        from vqeval.core.config import EvalConfig
        from vqeval.core.video_loader import VideoData, VideoMeta
        from vqeval.evaluators.base import get_evaluator_class
        from vqeval.models.model_registry import ModelRegistry

        self._config_cls = EvalConfig
        self._video_data_cls = VideoData
        self._video_meta_cls = VideoMeta
        self._evaluator_classes = {name: get_evaluator_class(name) for name in _EVALUATOR_MODULES}

        # Upstream uses a process-global singleton. FastVideo can hold one eval
        # worker per GPU in the same process, so each metric replica needs its
        # own registry to avoid binding every worker to the first CUDA device.
        registry = object.__new__(ModelRegistry)
        registry._initialized = False
        ModelRegistry.__init__(registry, device=str(self.device))
        self._registry = registry

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        video = sample.get("video")
        if video is None:
            return self._skip(sample, "missing video")
        if not isinstance(video, torch.Tensor) or video.ndim != 4:
            return self._skip(sample, "video must be a (T, C, H, W) tensor")
        if video.shape[0] < 2 or video.shape[1] != 3:
            return self._skip(sample, "video must contain at least two RGB frames")

        fps_value = sample.get("fps")
        if fps_value is None:
            return self._skip(sample, "missing fps (required for VQeval frame sampling)")
        fps = float(fps_value)
        if fps <= 0:
            return self._skip(sample, "fps must be greater than zero")

        if self._registry is None:
            self.setup()

        prompt = sample.get("text_prompt")
        source = str(sample.get("video_path", "<tensor>"))
        upstream_video, sampled_indices = self._to_upstream_video(video, fps=fps, source=source)
        config = self._config_cls(
            video_path=source,
            prompt=prompt,
            device=str(self.device),
        )

        active_dimensions = config.get_active_dimensions()
        weights = config.get_effective_weights()
        dimension_results: dict[str, dict[str, Any]] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for dimension in active_dimensions:
            evaluator = self._evaluator_classes[dimension](config=config, model_registry=self._registry)
            if not evaluator.is_applicable(upstream_video):
                continue
            result = evaluator.evaluate(upstream_video)
            dimension_results[dimension] = {
                "score": float(result.score),
                "verdict": result.verdict,
                "metrics": result.metrics,
            }
            weight = float(weights.get(dimension, 0.0))
            weighted_sum += float(result.score) * weight
            total_weight += weight

        if not dimension_results or total_weight == 0:
            return self._skip(sample, "no VQeval dimensions were applicable")

        score = weighted_sum / total_weight
        return MetricResult(
            name=self.name,
            score=score,
            details={
                "dimensions": dimension_results,
                "weights": {
                    name: float(weights[name])
                    for name in dimension_results
                },
                "source_frames": int(video.shape[0]),
                "sampled_frames": len(sampled_indices),
                "sampled_indices": sampled_indices,
                "fps": fps,
                "upstream_commit": _UPSTREAM_COMMIT,
            },
        )

    def _to_upstream_video(self, video: torch.Tensor, *, fps: float, source: str):
        source_frames, _, height, width = video.shape
        sampled_indices = _sample_indices(source_frames, fps)
        sampled = video.detach()[sampled_indices].float().clamp(0, 1)
        frames_rgb = (sampled.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        frames_bgr = frames_rgb[..., ::-1].copy()

        meta = self._video_meta_cls(
            path=source,
            width=int(width),
            height=int(height),
            fps=fps,
            total_frames=int(source_frames),
            duration=float(source_frames / fps),
            codec="",
            has_audio=False,
        )
        return (
            self._video_data_cls(
                meta=meta,
                frames=frames_bgr,
                frame_indices=np.asarray(sampled_indices),
                frames_rgb=frames_rgb,
            ),
            sampled_indices,
        )


def _sample_indices(total_frames: int, fps: float) -> list[int]:
    """Match VQeval's default all-frames/2-fps sampling policy."""
    duration = total_frames / fps
    if duration <= 5.0:
        return list(range(total_frames))

    interval = max(1, int(fps / 2.0))
    indices = set(range(0, total_frames, interval))
    indices.update((0, total_frames - 1, total_frames // 2))
    return sorted(indices)
