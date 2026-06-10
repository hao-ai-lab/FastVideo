# SPDX-License-Identifier: Apache-2.0
"""Validation callback.

All configuration is read from the YAML ``callbacks.validation``
section.  The pipeline class is resolved from
``pipeline_target``.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.dataset.validation_dataset import (
    ValidationDataset, )
from fastvideo.distributed import (
    get_sp_group,
    get_world_group,
)
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch
from fastvideo.train.callbacks.callback import Callback
from fastvideo.train.utils.instantiate import resolve_target
from fastvideo.train.utils.moduleloader import (
    make_inference_args, )
from fastvideo.training.trackers import DummyTracker
from fastvideo.utils import shallow_asdict

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


@dataclass(slots=True)
class _ValidationStepResult:
    videos: list[list[np.ndarray]]
    captions: list[str]
    overlay_videos: list[list[np.ndarray]] = field(default_factory=list)
    overlay_captions: list[str] = field(default_factory=list)
    ref_videos: list[str | None] = field(default_factory=list)
    actions: list[dict[str, Any] | None] = field(default_factory=list)
    mouse_pitch_signs: list[int | None] = field(default_factory=list)


@dataclass(slots=True)
class _ValidationMetricStats:
    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, float] = field(default_factory=dict)
    per_video: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _SavedValidationVideos:
    filenames: list[str] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)


@dataclass(slots=True)
class _ValidationMetricsConfig:
    enabled: bool = False
    names: list[str] = field(default_factory=list)
    device: str = "cuda"
    calibration_path: str | None = None
    mouse_pitch_sign: int = 1
    skip_missing_deps: bool = True
    strict: bool = False
    unload_after_validation: bool = True
    loader_threads: int = 1
    prefetch_factor: int = 2
    log_prefix: str = "metrics/validation"


DEFAULT_VALIDATION_VBENCH_METRICS = [
    "vbench.imaging_quality",
    "vbench.aesthetic_quality",
    "vbench.temporal_flickering",
    "vbench.motion_smoothness",
    "vbench.subject_consistency",
    "vbench.background_consistency",
    "vbench.dynamic_degree",
]


class ValidationCallback(Callback):
    """Generic validation callback driven entirely by YAML
    config.

    Works with any pipeline that follows the
    ``PipelineCls.from_pretrained(...)`` + ``pipeline.forward()``
    contract.
    """

    def __init__(
        self,
        *,
        pipeline_target: str,
        dataset_file: str,
        every_steps: int = 100,
        sampling_steps: list[int] | None = None,
        guidance_scale: float | None = None,
        num_frames: int | None = None,
        output_dir: str | None = None,
        sampling_timesteps: list[int] | None = None,
        overlay_actions: bool = False,
        **pipeline_kwargs: Any,
    ) -> None:
        self.pipeline_target = str(pipeline_target)
        self.dataset_file = str(dataset_file)
        self.every_steps = int(every_steps)
        self.sampling_steps = ([int(s) for s in sampling_steps] if sampling_steps else [40])
        self.guidance_scale = (float(guidance_scale) if guidance_scale is not None else None)
        self.num_frames = (int(num_frames) if num_frames is not None else None)
        self.output_dir = (str(output_dir) if output_dir is not None else None)
        self.sampling_timesteps = ([int(s) for s in sampling_timesteps] if sampling_timesteps is not None else None)
        self.overlay_actions = bool(overlay_actions)
        metrics_config = pipeline_kwargs.pop("metrics", None)
        self.metrics_config = self._parse_metrics_config(metrics_config)
        self.pipeline_kwargs = dict(pipeline_kwargs)

        # Set after on_train_start.
        self._pipeline: Any | None = None
        self._pipeline_key: tuple[Any, ...] | None = None
        self._sampling_param: SamplingParam | None = None
        self._metric_evaluator: Any | None = None
        self.tracker: Any = DummyTracker()
        self.validation_random_generator: (torch.Generator | None) = None
        self.seed: int = 0

    @staticmethod
    def _parse_metrics_config(config: Any) -> _ValidationMetricsConfig:
        if config is None or config is False:
            return _ValidationMetricsConfig(enabled=False)
        if config is True:
            return _ValidationMetricsConfig(
                enabled=True,
                names=list(DEFAULT_VALIDATION_VBENCH_METRICS),
            )
        if isinstance(config, str):
            return _ValidationMetricsConfig(
                enabled=True,
                names=[config],
            )
        if isinstance(config, list | tuple):
            return _ValidationMetricsConfig(
                enabled=bool(config),
                names=[str(name) for name in config],
            )
        if not isinstance(config, dict):
            raise TypeError("callbacks.validation.metrics must be a bool, string, list, or mapping")

        enabled = bool(config.get("enabled", True))
        raw_names = config.get("names", config.get("metrics", None))
        if raw_names is None:
            names = list(DEFAULT_VALIDATION_VBENCH_METRICS)
        elif isinstance(raw_names, str):
            names = [raw_names]
        else:
            names = [str(name) for name in raw_names]

        return _ValidationMetricsConfig(
            enabled=enabled and bool(names),
            names=names,
            device=str(config.get("device", "cuda")),
            calibration_path=(str(config["calibration_path"]) if config.get("calibration_path") is not None else None),
            mouse_pitch_sign=int(config.get("mouse_pitch_sign", 1)),
            skip_missing_deps=bool(config.get("skip_missing_deps", True)),
            strict=bool(config.get("strict", False)),
            unload_after_validation=bool(config.get("unload_after_validation", True)),
            loader_threads=int(config.get("loader_threads", 1)),
            prefetch_factor=int(config.get("prefetch_factor", 2)),
            log_prefix=str(config.get("log_prefix", "metrics/validation")),
        )

    # ----------------------------------------------------------
    # Callback hooks
    # ----------------------------------------------------------

    def on_train_start(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        self.method = method
        tc = self.training_config

        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self.global_rank = self.world_group.rank
        self.rank_in_sp_group = (self.sp_group.rank_in_group)
        self.sp_world_size = self.sp_group.world_size

        seed = tc.data.seed
        if seed is None:
            raise ValueError("training.data.seed must be set "
                             "for validation")
        self.seed = int(seed)
        self.validation_random_generator = (torch.Generator(device="cpu").manual_seed(self.seed))

        tracker = getattr(method, "tracker", None)
        if tracker is not None:
            self.tracker = tracker

    def on_validation_begin(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        if self.every_steps <= 0:
            return
        if iteration % self.every_steps != 0:
            return

        self._run_validation(method, iteration)

    # ----------------------------------------------------------
    # Core validation logic
    # ----------------------------------------------------------

    def _run_validation(
        self,
        method: TrainingMethod,
        step: int,
    ) -> None:

        transformer = method.student.transformer
        # Look for an EMA callback to temporarily swap
        # EMA weights during validation.
        ema_cb = self._find_ema_callback()
        ctx = ema_cb.ema_context(transformer) if ema_cb is not None else contextlib.nullcontext(transformer)
        with ctx as t:
            self._run_validation_inner(
                method,
                step,
                t,
            )

    def _find_ema_callback(self) -> Any | None:
        """Find the EMA callback in the callback dict."""
        from fastvideo.train.callbacks.ema import (
            EMACallback, )

        cb_dict = getattr(self, "_callback_dict", None)
        if cb_dict is not None:
            for cb in cb_dict._callbacks.values():
                if isinstance(cb, EMACallback):
                    return cb
        return None

    def _run_validation_inner(
        self,
        method: TrainingMethod,
        step: int,
        transformer: torch.nn.Module,
    ) -> None:
        tc = self.training_config
        was_training = bool(getattr(transformer, "training", False))

        output_dir = (self.output_dir or tc.checkpoint.output_dir)

        try:
            transformer.eval()
            num_sp_groups = (self.world_group.world_size // self.sp_group.world_size)
            sp = self._get_sampling_param()

            for num_inference_steps in self.sampling_steps:
                result = self._run_validation_for_steps(
                    num_inference_steps,
                    transformer=transformer,
                )

                if self.rank_in_sp_group != 0:
                    continue

                os.makedirs(
                    output_dir,
                    exist_ok=True,
                )
                local_videos = self._save_validation_videos(
                    result.videos,
                    output_dir=output_dir,
                    step=step,
                    num_inference_steps=num_inference_steps,
                    fps=sp.fps,
                )
                local_overlay_videos = self._save_validation_videos(
                    result.overlay_videos,
                    output_dir=output_dir,
                    step=step,
                    num_inference_steps=num_inference_steps,
                    fps=sp.fps,
                    suffix="_overlay",
                )
                local_video_filenames = local_videos.filenames
                local_overlay_video_filenames = local_overlay_videos.filenames
                local_captions = self._select_by_indices(
                    result.captions,
                    local_videos.indices,
                )
                local_overlay_captions = self._select_by_indices(
                    result.overlay_captions,
                    local_overlay_videos.indices,
                )
                local_ref_videos = self._select_by_indices(
                    result.ref_videos,
                    local_videos.indices,
                )
                local_actions = self._select_by_indices(
                    result.actions,
                    local_videos.indices,
                )
                local_mouse_pitch_signs = self._select_by_indices(
                    result.mouse_pitch_signs,
                    local_videos.indices,
                )
                local_metric_stats = self._evaluate_validation_metrics(
                    video_filenames=local_video_filenames,
                    captions=local_captions,
                    ref_videos=local_ref_videos,
                    actions=local_actions,
                    mouse_pitch_signs=local_mouse_pitch_signs,
                    fps=sp.fps,
                    output_dir=output_dir,
                    step=step,
                    num_inference_steps=num_inference_steps,
                )

                if self.global_rank == 0:
                    all_video_filenames = list(local_video_filenames)
                    all_overlay_video_filenames = list(local_overlay_video_filenames)
                    all_captions = list(local_captions)
                    all_overlay_captions = list(local_overlay_captions)
                    all_metric_stats = local_metric_stats
                    for sp_idx in range(1, num_sp_groups):
                        src = (sp_idx * self.sp_world_size)
                        recv_v = (self.world_group.recv_object(src=src))
                        recv_c = (self.world_group.recv_object(src=src))
                        recv_ov = (self.world_group.recv_object(src=src))
                        recv_oc = (self.world_group.recv_object(src=src))
                        recv_m = (self.world_group.recv_object(src=src))
                        all_video_filenames.extend(recv_v)
                        all_overlay_video_filenames.extend(recv_ov)
                        all_captions.extend(recv_c)
                        all_overlay_captions.extend(recv_oc)
                        self._merge_metric_stats(
                            all_metric_stats,
                            recv_m,
                        )

                    self._log_validation_metrics(
                        all_metric_stats,
                        step=step,
                    )
                    self._log_validation_video_artifacts(
                        all_video_filenames,
                        all_captions,
                        key=f"validation_videos_{num_inference_steps}_steps",
                        step=step,
                    )
                    if all_overlay_video_filenames:
                        self._log_validation_video_artifacts(
                            all_overlay_video_filenames,
                            all_overlay_captions,
                            key=(f"validation_videos_{num_inference_steps}"
                                 f"_steps_overlay"),
                            step=step,
                        )
                else:
                    self.world_group.send_object(
                        local_video_filenames,
                        dst=0,
                    )
                    self.world_group.send_object(
                        local_captions,
                        dst=0,
                    )
                    self.world_group.send_object(
                        local_overlay_video_filenames,
                        dst=0,
                    )
                    self.world_group.send_object(
                        local_overlay_captions,
                        dst=0,
                    )
                    self.world_group.send_object(
                        local_metric_stats,
                        dst=0,
                    )
        finally:
            if was_training:
                transformer.train()
            self._release_metric_evaluator()

    def _save_validation_videos(
        self,
        videos: list[list[np.ndarray]],
        *,
        output_dir: str,
        step: int,
        num_inference_steps: int,
        fps: int,
        suffix: str = "",
    ) -> _SavedValidationVideos:
        saved = _SavedValidationVideos()
        for i, video in enumerate(videos):
            fname = os.path.join(
                output_dir,
                f"validation_step_{step}"
                f"_inference_steps_{num_inference_steps}"
                f"_rank_{self.global_rank}"
                f"_video_{i}{suffix}.mp4",
            )
            # Validation video encoding can fail due transient ffmpeg or
            # filesystem errors; skip the artifact so training can continue.
            try:
                imageio.mimsave(
                    fname,
                    video,
                    fps=fps,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to save validation video %s on rank %s; skipping artifact: %s",
                    fname,
                    self.global_rank,
                    exc,
                )
                with contextlib.suppress(OSError):
                    os.remove(fname)
                continue
            saved.filenames.append(fname)
            saved.indices.append(i)
        return saved

    @staticmethod
    def _select_by_indices(
        values: list[Any],
        indices: list[int],
    ) -> list[Any]:
        return [values[i] for i in indices if i < len(values)]

    def _log_validation_video_artifacts(
        self,
        video_filenames: list[str],
        captions: list[str],
        *,
        key: str,
        step: int,
    ) -> None:
        video_logs = []
        for fname, cap in zip(
                video_filenames,
                captions,
                strict=True,
        ):
            art = self.tracker.video(
                fname,
                caption=cap,
            )
            if art is not None:
                video_logs.append(art)
        if video_logs:
            self.tracker.log_artifacts(
                {key: video_logs},
                step,
            )

    # ----------------------------------------------------------
    # Metric evaluation
    # ----------------------------------------------------------

    def _metric_device(self) -> str:
        device = self.metrics_config.device
        if device == "cuda" and torch.cuda.is_available():
            local_rank = int(getattr(self.world_group, "local_rank", 0))
            return f"cuda:{local_rank}"
        return device

    def _get_metric_evaluator(self) -> Any:
        if self._metric_evaluator is not None:
            return self._metric_evaluator
        from fastvideo.eval import Evaluator

        cfg = self.metrics_config
        self._metric_evaluator = Evaluator(
            metrics=cfg.names,
            device=self._metric_device(),
            num_gpus=1,
            loader_threads=cfg.loader_threads,
            prefetch_factor=cfg.prefetch_factor,
            pre_upload=False,
            skip_missing_deps=cfg.skip_missing_deps,
        )
        logger.info(
            "Initialized validation metrics: %s",
            ", ".join(self._metric_evaluator.metric_names),
        )
        return self._metric_evaluator

    def _release_metric_evaluator(self) -> None:
        if self._metric_evaluator is None:
            return
        if self.metrics_config.unload_after_validation:
            self._metric_evaluator.unload()
            self._metric_evaluator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _evaluate_validation_metrics(
        self,
        *,
        video_filenames: list[str],
        captions: list[str],
        ref_videos: list[str | None],
        actions: list[dict[str, Any] | None],
        mouse_pitch_signs: list[int | None],
        fps: int,
        output_dir: str,
        step: int,
        num_inference_steps: int,
    ) -> _ValidationMetricStats:
        stats = _ValidationMetricStats()
        if not self.metrics_config.enabled or not video_filenames:
            return stats

        try:
            evaluator = self._get_metric_evaluator()
            from fastvideo.eval import samples_from

            samples = samples_from(
                video=video_filenames,
                reference=self._available_paths(ref_videos),
                text_prompts=captions,
                fps=float(fps),
                extras=self._validation_metric_extras(
                    video_filenames=video_filenames,
                    actions=actions,
                    mouse_pitch_signs=mouse_pitch_signs,
                ),
            )
            results = evaluator.evaluate(samples=samples)
            for filename, metric_results in zip(
                    video_filenames,
                    results,
                    strict=True,
            ):
                row: dict[str, Any] = {"path": filename}
                self._accumulate_metric_results(
                    stats,
                    row,
                    metric_results,
                )
                stats.per_video.append(row)
            for metric_name, metric_result in results.corpus.items():
                row = {"path": "<corpus>"}
                self._accumulate_metric_results(
                    stats,
                    row,
                    {metric_name: metric_result},
                )
                stats.per_video.append(row)
        except Exception as exc:
            message = ("Validation metric evaluation failed on rank "
                       f"{self.global_rank}: {exc}")
            logger.exception(message)
            stats.errors.append(message)
            if self.metrics_config.strict:
                raise
        finally:
            if self._metric_evaluator is not None:
                self._metric_evaluator.release_cuda_memory()

        self._write_metric_summary(
            stats,
            output_dir=output_dir,
            step=step,
            num_inference_steps=num_inference_steps,
        )
        return stats

    @staticmethod
    def _accumulate_metric_results(
        stats: _ValidationMetricStats,
        row: dict[str, Any],
        metric_results: dict[str, Any],
    ) -> None:
        for metric_name, metric_result in metric_results.items():
            score = getattr(metric_result, "score", None)
            ValidationCallback._accumulate_scalar(
                stats,
                row,
                metric_name,
                score,
            )
            details = getattr(metric_result, "details", {}) or {}
            if not isinstance(details, dict):
                continue
            for detail_name, value in details.items():
                key = f"{metric_name}.{detail_name}"
                ValidationCallback._accumulate_scalar(
                    stats,
                    row,
                    key,
                    value,
                )

    @staticmethod
    def _accumulate_scalar(
        stats: _ValidationMetricStats,
        row: dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        if not isinstance(value, float | int | np.floating | np.integer):
            return
        value_float = float(value)
        if not np.isfinite(value_float):
            return
        row[key] = value_float
        stats.sums[key] = stats.sums.get(key, 0.0) + value_float
        stats.counts[key] = stats.counts.get(key, 0.0) + 1.0

    @staticmethod
    def _merge_metric_stats(
        dst: _ValidationMetricStats,
        src: _ValidationMetricStats,
    ) -> None:
        for key, value in src.sums.items():
            dst.sums[key] = dst.sums.get(key, 0.0) + float(value)
        for key, value in src.counts.items():
            dst.counts[key] = dst.counts.get(key, 0.0) + float(value)
        dst.per_video.extend(src.per_video)
        dst.errors.extend(src.errors)

    def _log_validation_metrics(
        self,
        stats: _ValidationMetricStats,
        *,
        step: int,
    ) -> None:
        if stats.errors:
            message = "Validation metric evaluation failed:\n" + "\n".join(stats.errors)
            if self.metrics_config.strict:
                raise RuntimeError(message)
            logger.warning(message)

        logs: dict[str, float] = {}
        for key, metric_sum in stats.sums.items():
            metric_count = stats.counts.get(key, 0.0)
            if metric_count <= 0:
                continue
            value = float(metric_sum / metric_count)
            if not np.isfinite(value):
                continue
            logs[self._metric_log_name(key)] = value
        if logs:
            self.tracker.log(
                logs,
                step,
            )

    def _metric_log_name(
        self,
        key: str,
    ) -> str:
        return f"{self.metrics_config.log_prefix}/{key.replace('.', '/')}"

    def _write_metric_summary(
        self,
        stats: _ValidationMetricStats,
        *,
        output_dir: str,
        step: int,
        num_inference_steps: int,
    ) -> None:
        if not self.metrics_config.enabled:
            return
        summary_dir = os.path.join(
            output_dir,
            "eval",
            f"step_{step}",
        )
        os.makedirs(
            summary_dir,
            exist_ok=True,
        )
        metrics = {
            key: float(stats.sums[key] / stats.counts[key])
            for key in sorted(stats.sums) if stats.counts.get(key, 0.0) > 0
        }
        payload = {
            "step": int(step),
            "num_inference_steps": int(num_inference_steps),
            "rank": int(self.global_rank),
            "metrics": metrics,
            "per_video": stats.per_video,
            "errors": stats.errors,
        }
        path = os.path.join(
            summary_dir,
            f"inference_steps_{num_inference_steps}_rank_{self.global_rank}.json",
        )
        with open(
                path,
                "w",
                encoding="utf-8",
        ) as f:
            json.dump(
                payload,
                f,
                indent=2,
            )

    def _validation_metric_extras(
        self,
        *,
        video_filenames: list[str],
        actions: list[dict[str, Any] | None],
        mouse_pitch_signs: list[int | None],
    ) -> list[dict[str, Any]]:
        extras: list[dict[str, Any]] = []
        for i, _filename in enumerate(video_filenames):
            extra: dict[str, Any] = {}
            action = actions[i] if i < len(actions) else None
            if action is not None:
                extra["actions"] = action
            if self.metrics_config.calibration_path is not None:
                extra["calibration"] = self.metrics_config.calibration_path
            mouse_pitch_sign = mouse_pitch_signs[i] if i < len(mouse_pitch_signs) else None
            extra["mouse_pitch_sign"] = int(mouse_pitch_sign or self.metrics_config.mouse_pitch_sign)
            extras.append(extra)
        return extras

    @staticmethod
    def _available_paths(paths: list[str | None]) -> list[str] | None:
        if not paths or any(path is None for path in paths):
            return None
        return [str(path) for path in paths]

    @staticmethod
    def _validation_actions(validation_batch: dict[str, Any]) -> dict[str, Any] | None:
        keyboard = validation_batch.get("keyboard_cond")
        mouse = validation_batch.get("mouse_cond")
        if keyboard is not None and mouse is not None:
            return {
                "keyboard": np.asarray(keyboard),
                "mouse": np.asarray(mouse),
            }

        action_path = validation_batch.get("action_path")
        if not isinstance(action_path, str) or not os.path.isfile(action_path):
            return None
        try:
            actions_obj = np.load(action_path, allow_pickle=True)
            if isinstance(actions_obj, np.ndarray) and actions_obj.dtype == object:
                actions_obj = actions_obj.item()
        except Exception as exc:
            logger.warning(
                "Failed to load validation action file %s for metrics: %s",
                action_path,
                exc,
            )
            return None
        if not isinstance(actions_obj, dict):
            return None
        keyboard = actions_obj.get("keyboard")
        mouse = actions_obj.get("mouse")
        if keyboard is None or mouse is None:
            return None
        return {
            "keyboard": np.asarray(keyboard),
            "mouse": np.asarray(mouse),
        }

    def _validation_mouse_pitch_sign(
        self,
        validation_batch: dict[str, Any],
    ) -> int:
        if validation_batch.get("mouse_pitch_sign") is not None:
            return int(validation_batch["mouse_pitch_sign"])
        flipped = validation_batch.get("mouse_pitch_flipped")
        if isinstance(flipped, str):
            flipped = flipped.lower() in {"1", "true", "yes", "y"}
        if bool(flipped):
            return -1
        return int(self.metrics_config.mouse_pitch_sign)

    # ----------------------------------------------------------
    # Pipeline management
    # ----------------------------------------------------------

    def _get_sampling_param(self) -> SamplingParam:
        if self._sampling_param is None:
            self._sampling_param = (SamplingParam.from_pretrained(self._pipeline_model_path()))
        return self._sampling_param

    def _pipeline_model_path(self) -> str:
        tc = self.training_config
        pipeline_config = getattr(tc, "pipeline_config", None)
        model_path = getattr(pipeline_config, "_fastvideo_train_model_path", None)
        return str(model_path or tc.model_path)

    def _validation_pipeline_config(self, transformer: torch.nn.Module) -> Any:
        tc = self.training_config
        pipeline_config = deepcopy(tc.pipeline_config)
        self._sync_runtime_dit_arch_config(
            pipeline_config,
            transformer,
        )
        return pipeline_config

    @staticmethod
    def _sync_runtime_dit_arch_config(
        pipeline_config: Any,
        transformer: torch.nn.Module,
    ) -> None:
        dit_config = getattr(
            pipeline_config,
            "dit_config",
            None,
        )
        arch_config = getattr(
            dit_config,
            "arch_config",
            None,
        )
        if arch_config is None:
            return

        for name in ("local_attn_size", "sink_size"):
            if not hasattr(arch_config, name) or not hasattr(transformer, name):
                continue
            setattr(
                arch_config,
                name,
                getattr(transformer, name),
            )

    def _get_pipeline(
        self,
        *,
        transformer: torch.nn.Module,
    ) -> Any:
        key = (id(transformer), )
        if (self._pipeline is not None and self._pipeline_key == key):
            return self._pipeline

        tc = self.training_config
        PipelineCls = resolve_target(self.pipeline_target)
        flow_shift = getattr(
            tc.pipeline_config,
            "flow_shift",
            None,
        )

        kwargs: dict[str, Any] = {
            "inference_mode": True,
            "loaded_modules": {
                "transformer": transformer,
            },
            "tp_size": tc.distributed.tp_size,
            "sp_size": tc.distributed.sp_size,
            "num_gpus": tc.distributed.num_gpus,
            "pin_cpu_memory": (tc.distributed.pin_cpu_memory),
            "dit_cpu_offload": False,
            "dit_layerwise_offload": False,
        }
        if flow_shift is not None:
            kwargs["flow_shift"] = float(flow_shift)

        self._pipeline = PipelineCls.from_pretrained(
            self._pipeline_model_path(),
            **kwargs,
        )
        if tc.pipeline_config is not None:
            self._pipeline.fastvideo_args.pipeline_config = self._validation_pipeline_config(transformer)
            arch_config = self._pipeline.fastvideo_args.pipeline_config.dit_config.arch_config
            logger.info(
                "Validation pipeline runtime config: local_attn_size=%s sink_size=%s boundary_ratio=%s",
                getattr(arch_config, "local_attn_size", None),
                getattr(arch_config, "sink_size", None),
                getattr(self._pipeline.fastvideo_args.pipeline_config.dit_config, "boundary_ratio", None),
            )

        scheduler = self._pipeline.get_module("scheduler")
        if (scheduler is not None and type(scheduler).__name__ == "SelfForcingFlowMatchScheduler"):
            scheduler.sigma_min = 0.0
            scheduler.extra_one_step = True
            scheduler.set_timesteps(num_inference_steps=1000, training=True)

        self._pipeline_key = key
        return self._pipeline

    # ----------------------------------------------------------
    # Batch preparation
    # ----------------------------------------------------------

    def _prepare_validation_batch(
        self,
        sampling_param: SamplingParam,
        validation_batch: dict[str, Any],
        num_inference_steps: int,
    ) -> ForwardBatch:
        tc = self.training_config

        sampling_param.prompt = validation_batch["prompt"]
        sampling_param.height = tc.data.num_height
        sampling_param.width = tc.data.num_width
        sampling_param.num_inference_steps = int(num_inference_steps)
        sampling_param.data_type = "video"
        if self.guidance_scale is not None:
            sampling_param.guidance_scale = float(self.guidance_scale)
        sampling_param.seed = self.seed

        # image_path for I2V pipelines.
        img_path = (validation_batch.get("image_path") or validation_batch.get("video_path"))
        if img_path is not None and (img_path.startswith("http") or os.path.isfile(img_path)):
            sampling_param.image_path = img_path

        temporal_compression_factor = int(
            tc.pipeline_config.vae_config.arch_config.temporal_compression_ratio  # type: ignore[union-attr]
        )
        default_num_frames = ((tc.data.num_latent_t - 1) * temporal_compression_factor + 1)
        if self.num_frames is not None:
            sampling_param.num_frames = int(self.num_frames)
        else:
            sampling_param.num_frames = int(default_num_frames)

        latents_size = [
            (sampling_param.num_frames - 1) // 4 + 1,
            sampling_param.height // 8,
            sampling_param.width // 8,
        ]
        n_tokens = (latents_size[0] * latents_size[1] * latents_size[2])

        sampling_timesteps_tensor = (torch.tensor(
            [int(s) for s in self.sampling_timesteps],
            dtype=torch.long,
        ) if self.sampling_timesteps is not None else None)

        inference_args = make_inference_args(
            tc,
            model_path=tc.model_path,
        )

        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            generator=self.validation_random_generator,
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=tc.vsa_sparsity,
            timesteps=sampling_timesteps_tensor,
        )
        batch._inference_args = inference_args  # type: ignore[attr-defined]

        # Conditionally set I2V fields.
        if ("image" in validation_batch and validation_batch["image"] is not None):
            batch.pil_image = validation_batch["image"]

        self._attach_action_conditions(
            batch,
            validation_batch,
            sampling_param.num_frames,
        )

        return batch

    def _attach_action_conditions(
        self,
        batch: ForwardBatch,
        validation_batch: dict[str, Any],
        num_frames: int,
    ) -> None:
        for name in ("keyboard_cond", "mouse_cond"):
            value = validation_batch.get(name)
            if value is None:
                continue
            array = np.asarray(value)
            if array.size == 0:
                continue
            array = array[:num_frames]
            tensor = torch.as_tensor(
                array,
                dtype=torch.bfloat16,
            )
            student = getattr(self.method, "student", None)
            prepare_action = getattr(student, "prepare_validation_action_condition", None)
            if prepare_action is not None:
                tensor = prepare_action(tensor, name=name)
            tensor = tensor.unsqueeze(0)
            setattr(
                batch,
                name,
                tensor,
            )

    # ----------------------------------------------------------
    # Validation loop
    # ----------------------------------------------------------

    def _run_validation_for_steps(
        self,
        num_inference_steps: int,
        *,
        transformer: torch.nn.Module,
    ) -> _ValidationStepResult:
        tc = self.training_config
        pipeline = self._get_pipeline(transformer=transformer, )
        sampling_param = self._get_sampling_param()

        dataset = ValidationDataset(self.dataset_file)
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
        )

        inference_args = make_inference_args(
            tc,
            model_path=tc.model_path,
        )
        self._sync_runtime_dit_arch_config(
            inference_args.pipeline_config,
            transformer,
        )

        # Propagate sampling_timesteps to pipeline_config so
        # causal/DMD denoising stages can read them.
        if (self.sampling_timesteps is not None and inference_args.pipeline_config.dmd_denoising_steps is None):
            inference_args.pipeline_config.dmd_denoising_steps = ([int(s) for s in self.sampling_timesteps])

        videos: list[list[np.ndarray]] = []
        overlay_videos: list[list[np.ndarray]] = []
        captions: list[str] = []
        overlay_captions: list[str] = []
        ref_videos: list[str | None] = []
        actions: list[dict[str, Any] | None] = []
        mouse_pitch_signs: list[int | None] = []

        for validation_batch in dataloader:
            batch = self._prepare_validation_batch(
                sampling_param,
                validation_batch,
                num_inference_steps,
            )

            assert (batch.prompt is not None and isinstance(batch.prompt, str))
            captions.append(batch.prompt)
            ref_video = validation_batch.get("ref_video")
            ref_videos.append(ref_video if isinstance(ref_video, str) else None)
            action = self._validation_actions(validation_batch)
            actions.append(action)
            mouse_pitch_signs.append(self._validation_mouse_pitch_sign(validation_batch))

            with torch.no_grad():
                output_batch = pipeline.forward(
                    batch,
                    inference_args,
                )

            samples = output_batch.output.cpu()
            if self.rank_in_sp_group != 0:
                continue

            video = rearrange(
                samples,
                "b c t h w -> t b c h w",
            )
            frames: list[np.ndarray] = []
            for x in video:
                x = torchvision.utils.make_grid(
                    x,
                    nrow=6,
                )
                x = (x.transpose(0, 1).transpose(1, 2).squeeze(-1))
                frames.append((x * 255).numpy().astype(np.uint8))
            videos.append(frames)
            if self.overlay_actions:
                overlay_frames = self._post_process_validation_frames(
                    frames,
                    action=action,
                )
                if overlay_frames is not None:
                    overlay_videos.append(overlay_frames)
                    overlay_captions.append(batch.prompt)

        return _ValidationStepResult(
            videos=videos,
            captions=captions,
            overlay_videos=overlay_videos,
            overlay_captions=overlay_captions,
            ref_videos=ref_videos,
            actions=actions,
            mouse_pitch_signs=mouse_pitch_signs,
        )

    def _post_process_validation_frames(
        self,
        frames: list[np.ndarray],
        *,
        action: dict[str, Any] | None,
    ) -> list[np.ndarray] | None:
        student = getattr(self.method, "student", None)
        post_process = getattr(student, "post_process_validation_frames", None)
        if post_process is None:
            return None
        return post_process(
            frames,
            action=action,
        )

    # ----------------------------------------------------------
    # State management
    # ----------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.validation_random_generator is not None:
            state["validation_rng"] = (self.validation_random_generator.get_state())
        return state

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        rng_state = state_dict.get("validation_rng")
        if (rng_state is not None and self.validation_random_generator is not None):
            self.validation_random_generator.set_state(rng_state)
