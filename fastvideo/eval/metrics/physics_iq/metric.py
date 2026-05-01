from __future__ import annotations

from typing import Any
from collections.abc import Iterable, Mapping

import numpy as np
import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.metrics.physics_iq.models import DEFAULT_DURATION_SECONDS, DEFAULT_TARGET_FPS
from fastvideo.eval.metrics.physics_iq.mse.metric import PhysicsIQMSEMetric
from fastvideo.eval.metrics.physics_iq.spatial_iou.metric import SpatialIoUMetric
from fastvideo.eval.metrics.physics_iq.spatiotemporal_iou.metric import SpatiotemporalIoUMetric
from fastvideo.eval.metrics.physics_iq.weighted_spatial_iou.metric import WeightedSpatialIoUMetric
from fastvideo.eval.metrics.physics_iq.utils import mean, prepare_pair_inputs, prepare_triplet_inputs, unpack_batch_value
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("physics_iq")
class PhysicsIQMetric(BaseMetric):
    name = "physics_iq"
    requires_reference = True
    higher_is_better = True
    batch_unit = "video"

    def __init__(
        self,
        *,
        target_fps: int = DEFAULT_TARGET_FPS,
        duration_seconds: int = DEFAULT_DURATION_SECONDS,
        video_time_selection: str = "first",
        threshold: int = 10,
        alpha: float = 0.3,
        roundtrip_generated_masks: bool = True,
    ) -> None:
        super().__init__()
        self._prep_kwargs = {
            "target_fps": target_fps,
            "duration_seconds": duration_seconds,
            "video_time_selection": video_time_selection,
            "threshold": threshold,
            "alpha": alpha,
            "roundtrip_generated_masks": roundtrip_generated_masks,
        }
        self._mse = PhysicsIQMSEMetric(**self._prep_kwargs)
        self._spatiotemporal_iou = SpatiotemporalIoUMetric(**self._prep_kwargs)
        self._spatial_iou = SpatialIoUMetric(**self._prep_kwargs)
        self._weighted_spatial_iou = WeightedSpatialIoUMetric(**self._prep_kwargs)

    def trial_forward(self, batch_size: int, *, height: int, width: int, num_frames: int) -> None:
        sample = {
            "video": torch.rand(batch_size, num_frames, 3, height, width, device=self.device),
            "reference": torch.rand(batch_size, num_frames, 3, height, width, device=self.device),
            "reference_take2": torch.rand(batch_size, num_frames, 3, height, width, device=self.device),
        }
        with torch.no_grad():
            self.compute(sample)

    @staticmethod
    def _extract_payload(result: MetricResult | Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(result, MetricResult):
            return result.details
        return result

    def _compute_pair_metrics(self, prepared_pair) -> dict[str, Any]:
        sample = {"_physics_iq_pair": prepared_pair}
        mse = self._mse.compute(sample)[0]
        st = self._spatiotemporal_iou.compute(sample)[0]
        spatial = self._spatial_iou.compute(sample)[0]
        weighted = self._weighted_spatial_iou.compute(sample)[0]
        return {
            "mse_per_frame": mse.details["per_frame"],
            "spatiotemporal_iou_per_frame": st.details["per_frame"],
            "spatial_iou": float(spatial.score),
            "weighted_spatial_iou": float(weighted.score),
            "mse_mean": float(mse.score),
            "spatiotemporal_iou_mean": float(st.score),
        }

    def compute_single(
        self,
        generated: Any,
        reference: Any,
        reference_take2: Any,
        *,
        generated_mask: Any | None = None,
        reference_mask: Any | None = None,
        reference_take2_mask: Any | None = None,
        scenario: str | None = None,
        view: str | None = None,
    ) -> dict[str, Any]:
        prepared = prepare_triplet_inputs(
            generated,
            reference,
            reference_take2,
            generated_mask=generated_mask,
            reference_mask=reference_mask,
            reference_take2_mask=reference_take2_mask,
            **self._prep_kwargs,
        )
        pair_metrics = self._compute_pair_metrics(prepared)
        variance_pair = prepare_pair_inputs(
            reference,
            reference_take2,
            generated_mask=reference_mask,
            reference_mask=reference_take2_mask,
            **self._prep_kwargs,
        )
        variance_metrics = self._compute_pair_metrics(variance_pair)
        details = {
            **pair_metrics,
            "pv_mse_per_frame": variance_metrics["mse_per_frame"],
            "pv_spatiotemporal_iou_per_frame": variance_metrics["spatiotemporal_iou_per_frame"],
            "pv_spatial_iou": variance_metrics["spatial_iou"],
            "pv_weighted_spatial_iou": variance_metrics["weighted_spatial_iou"],
            "pv_mse_mean": variance_metrics["mse_mean"],
            "pv_spatiotemporal_iou_mean": variance_metrics["spatiotemporal_iou_mean"],
        }
        if scenario is not None:
            details["scenario"] = scenario
        if view is not None:
            details["view"] = view
        return details

    @classmethod
    def aggregate(cls, results_list: Iterable[MetricResult | Mapping[str, Any]]) -> float:
        payloads = [cls._extract_payload(result) for result in results_list]
        if not payloads:
            raise ValueError("PhysicsIQMetric.aggregate requires at least one result.")

        a_mse = mean([value for payload in payloads for value in payload["mse_per_frame"]])
        a_st = mean([value for payload in payloads for value in payload["spatiotemporal_iou_per_frame"]])
        a_s = mean([float(payload["spatial_iou"]) for payload in payloads])
        a_ws = mean([float(payload["weighted_spatial_iou"]) for payload in payloads])

        v_mse = mean([value for payload in payloads for value in payload["pv_mse_per_frame"]])
        v_st = mean([value for payload in payloads for value in payload["pv_spatiotemporal_iou_per_frame"]])
        v_s = mean([float(payload["pv_spatial_iou"]) for payload in payloads])
        v_ws = mean([float(payload["pv_weighted_spatial_iou"]) for payload in payloads])

        score = 100.0 * ((((a_st / v_st) + (a_s / v_s) + (a_ws / v_ws)) / 3.0) - (a_mse - v_mse))
        return round(float(np.clip(score, 0.0, 100.0)), 2)

    @classmethod
    def aggregate_components(cls, results_list: Iterable[MetricResult | Mapping[str, Any]]) -> dict[str, float]:
        payloads = [cls._extract_payload(result) for result in results_list]
        return {
            "physics_iq": cls.aggregate(payloads),
            "a_mse": mean([value for payload in payloads for value in payload["mse_per_frame"]]),
            "a_st": mean([value for payload in payloads for value in payload["spatiotemporal_iou_per_frame"]]),
            "a_s": mean([float(payload["spatial_iou"]) for payload in payloads]),
            "a_ws": mean([float(payload["weighted_spatial_iou"]) for payload in payloads]),
            "v_mse": mean([value for payload in payloads for value in payload["pv_mse_per_frame"]]),
            "v_st": mean([value for payload in payloads for value in payload["pv_spatiotemporal_iou_per_frame"]]),
            "v_s": mean([float(payload["pv_spatial_iou"]) for payload in payloads]),
            "v_ws": mean([float(payload["pv_weighted_spatial_iou"]) for payload in payloads]),
        }

    def _per_video_score(self, details: Mapping[str, Any]) -> float:
        score = 100.0 * (
            (
                (mean(details["spatiotemporal_iou_per_frame"]) / mean(details["pv_spatiotemporal_iou_per_frame"]))
                + (float(details["spatial_iou"]) / float(details["pv_spatial_iou"]))
                + (float(details["weighted_spatial_iou"]) / float(details["pv_weighted_spatial_iou"]))
            ) / 3.0
            - (mean(details["mse_per_frame"]) - mean(details["pv_mse_per_frame"]))
        )
        return round(float(np.clip(score, 0.0, 100.0)), 2)

    def compute(self, sample: dict) -> list[MetricResult]:
        if "reference" not in sample:
            raise KeyError("PhysicsIQMetric requires sample['reference'].")

        take2_key = None
        for candidate in ("reference_take2", "real_take2", "take2"):
            if candidate in sample:
                take2_key = candidate
                break
        if take2_key is None:
            raise KeyError("PhysicsIQMetric requires sample['reference_take2'] or an alias.")

        videos = unpack_batch_value(sample["video"])
        references = unpack_batch_value(sample["reference"])
        references_take2 = unpack_batch_value(sample[take2_key])
        generated_masks = unpack_batch_value(sample["video_mask"]) if "video_mask" in sample else [None] * len(videos)
        reference_masks = unpack_batch_value(sample["reference_mask"]) if "reference_mask" in sample else [None] * len(videos)
        reference_take2_masks = (
            unpack_batch_value(sample["reference_take2_mask"])
            if "reference_take2_mask" in sample
            else [None] * len(videos)
        )
        if not (
            len(videos)
            == len(references)
            == len(references_take2)
            == len(generated_masks)
            == len(reference_masks)
            == len(reference_take2_masks)
        ):
            raise ValueError("Physics-IQ batched inputs must have the same batch size.")

        scenarios = sample.get("scenario")
        views = sample.get("view")
        results: list[MetricResult] = []
        for idx, items in enumerate(
            zip(videos, references, references_take2, generated_masks, reference_masks, reference_take2_masks)
        ):
            video, reference, reference_take2, generated_mask, reference_mask, reference_take2_mask = items
            scenario = scenarios[idx] if isinstance(scenarios, list) else scenarios
            view = views[idx] if isinstance(views, list) else views
            details = self.compute_single(
                video,
                reference,
                reference_take2,
                generated_mask=generated_mask,
                reference_mask=reference_mask,
                reference_take2_mask=reference_take2_mask,
                scenario=scenario,
                view=view,
            )
            results.append(MetricResult(name=self.name, score=self._per_video_score(details), details=details))
        return results

