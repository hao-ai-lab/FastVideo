# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
from tqdm.auto import tqdm

from fastvideo.distributed import get_sp_group, get_world_group
from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import CallbackDict
from fastvideo.train.methods.base import TrainingMethod
from fastvideo.train.utils.tracking import build_tracker

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )

logger = init_logger(__name__)


def _coerce_log_scalar(value: Any, *, where: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor at {where}, "
                             f"got shape={tuple(value.shape)}")
        return float(value.detach().item())
    if isinstance(value, float | int):
        return float(value)
    raise TypeError(f"Expected a scalar (float/int/Tensor) at "
                    f"{where}, got {type(value).__name__}")


def _maybe_log_resume_fingerprint(
    method: TrainingMethod,
    *,
    global_rank: int,
    step: int,
) -> None:
    if os.getenv("FASTVIDEO_DEBUG_RESUME_HASH", "").lower() not in {"1", "true", "yes"}:
        return

    transformer = method.transformer_inference
    fingerprints: list[str] = []
    for idx, (name, param) in enumerate(transformer.named_parameters()):
        if idx >= 3:
            break
        data = param.detach().reshape(-1)
        if data.numel() == 0:
            fingerprints.append(f"{name}:empty")
            continue
        sample = data[:16].float()
        fingerprints.append(
            f"{name}:shape={tuple(param.shape)} "
            f"sample_sum={sample.sum().item():.8f} "
            f"sample_mean={sample.mean().item():.8f} "
            f"first={sample[0].item():.8f}"
        )

    print(
        "DEBUG_RESUME_FINGERPRINT "
        f"rank={global_rank} step={step} " + " | ".join(fingerprints),
        flush=True,
    )


def _decode_latent_video(
    vae: Any,
    latents: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        latents = latents.detach()
        latents = latents.permute(0, 2, 1, 3, 4)

        scaling_factor = getattr(vae, "scaling_factor", None)
        if scaling_factor is not None:
            if isinstance(scaling_factor, torch.Tensor):
                latents = latents / scaling_factor.to(
                    latents.device,
                    latents.dtype,
                )
            else:
                latents = latents / float(scaling_factor)

        shift_factor = getattr(vae, "shift_factor", None)
        if shift_factor is not None:
            if isinstance(shift_factor, torch.Tensor):
                latents = latents + shift_factor.to(
                    latents.device,
                    latents.dtype,
                )
            else:
                latents = latents + float(shift_factor)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=latents.is_cuda,
        ):
            video = vae.decode(latents)
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.detach().cpu().float().permute(0, 2, 1, 3, 4)
        return (video * 255).numpy().astype(np.uint8)


def _maybe_add_video_artifact(
    tracker: Any,
    videos: list[Any],
    *,
    vae: Any,
    source_dict: dict[str, Any],
    latent_key: str,
    caption: str,
) -> None:
    latents = source_dict.get(latent_key)
    if not isinstance(latents, torch.Tensor):
        return
    video = _decode_latent_video(vae, latents)
    artifact = tracker.video(
        video,
        caption=caption,
        fps=24,
        format="mp4",
    )
    if artifact is not None:
        videos.append(artifact)


@dataclass(slots=True)
class TrainLoopState:
    step: int
    accum_iter: int


class Trainer:

    def __init__(
        self,
        training_config: TrainingConfig,
        *,
        config: dict[str, Any] | None = None,
        callback_configs: dict[str, dict[str, Any]]
        | None = None,
    ) -> None:
        self.training_config = training_config
        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self.global_rank = self.world_group.rank
        self.local_rank = self.world_group.local_rank
        self.tracker = build_tracker(
            training_config.tracker,
            training_config.checkpoint,
            config=config,
        )
        self.callbacks = CallbackDict(
            callback_configs or {},
            training_config,
        )

    def _should_log_train_artifacts(
        self,
        step: int,
    ) -> bool:
        validation_cb = self.callbacks.get_callback("validation")
        every_steps = getattr(validation_cb, "every_steps", None)
        if every_steps is None:
            return False
        every_steps = int(every_steps)
        if every_steps <= 0:
            return False
        return step % every_steps == 0

    def _log_train_artifacts(
        self,
        method: TrainingMethod,
        outputs: dict[str, Any],
        *,
        step: int,
    ) -> None:
        if self.global_rank != 0:
            return
        if not self._should_log_train_artifacts(step):
            return

        vae = getattr(method.student, "vae", None)
        if vae is None:
            return

        artifacts: dict[str, Any] = {}
        video_artifacts: list[Any] = []

        dmd_latent_dict = outputs.get("dmd_latent_vis_dict")
        if isinstance(dmd_latent_dict, dict) and dmd_latent_dict:
            _maybe_add_video_artifact(
                self.tracker,
                video_artifacts,
                latent_key="generator_pred_video",
                caption="generator",
                vae=vae,
                source_dict=dmd_latent_dict,
            )
            _maybe_add_video_artifact(
                self.tracker,
                video_artifacts,
                latent_key="real_score_pred_video",
                caption="real_score",
                vae=vae,
                source_dict=dmd_latent_dict,
            )
            _maybe_add_video_artifact(
                self.tracker,
                video_artifacts,
                latent_key="faker_score_pred_video",
                caption="fake_score",
                vae=vae,
                source_dict=dmd_latent_dict,
            )

            for scalar_key in ("generator_timestep", "dmd_timestep"):
                value = dmd_latent_dict.get(scalar_key)
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    artifacts[scalar_key] = float(value.detach().item())

        fake_score_latent_dict = outputs.get("fake_score_latent_vis_dict")
        if isinstance(fake_score_latent_dict, dict) and fake_score_latent_dict:
            _maybe_add_video_artifact(
                self.tracker,
                video_artifacts,
                latent_key="generator_pred_video",
                caption="critic_generator",
                vae=vae,
                source_dict=fake_score_latent_dict,
            )
            value = fake_score_latent_dict.get("fake_score_timestep")
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                artifacts["fake_score_timestep"] = float(
                    value.detach().item()
                )

        if video_artifacts:
            artifacts["train_visualization"] = video_artifacts

        if artifacts:
            self.tracker.log_artifacts(artifacts, step)

    def _iter_dataloader(self, dataloader: Any) -> Iterator[dict[str, Any]]:
        data_iter = iter(dataloader)
        while True:
            batch = next(data_iter, None)
            if batch is None:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            yield batch

    def run(
        self,
        method: TrainingMethod,
        *,
        dataloader: Any,
        max_steps: int,
        start_step: int = 0,
        checkpoint_manager: Any | None = None,
    ) -> None:
        tc = self.training_config
        grad_accum = max(
            1,
            int(tc.loop.gradient_accumulation_steps or 1),
        )

        method.set_tracker(self.tracker)
        method.on_train_start()

        resume_from_checkpoint = (tc.checkpoint.resume_from_checkpoint or "")
        if checkpoint_manager is not None:
            resumed_step = (checkpoint_manager.maybe_resume(resume_from_checkpoint=(resume_from_checkpoint)))
            if resumed_step is not None:
                start_step = int(resumed_step)
                _maybe_log_resume_fingerprint(
                    method,
                    global_rank=self.global_rank,
                    step=start_step,
                )
        self.callbacks.on_train_start(
            method,
            iteration=start_step,
        )
        self.callbacks.on_validation_begin(
            method,
            iteration=start_step,
        )
        method.optimizers_zero_grad(start_step)

        data_stream = self._iter_dataloader(dataloader)
        progress = tqdm(
            range(start_step + 1, max_steps + 1),
            total=max_steps,
            initial=start_step,
            desc="Steps",
            disable=self.local_rank > 0,
        )
        for step in progress:
            t0 = time.perf_counter()

            loss_sums: dict[str, float] = {}
            metric_sums: dict[str, float] = {}
            last_outputs: dict[str, Any] = {}
            for accum_iter in range(grad_accum):
                batch = next(data_stream)
                loss_map, outputs, step_metrics = method.single_train_step(
                    batch,
                    step,
                )
                last_outputs = outputs

                method.backward(
                    loss_map,
                    outputs,
                    grad_accum_rounds=grad_accum,
                )

                for k, v in loss_map.items():
                    if isinstance(v, torch.Tensor):
                        loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().item())
                for k, v in step_metrics.items():
                    if k in loss_sums:
                        raise ValueError(f"Metric key {k!r} collides "
                                         "with loss key. Use a "
                                         "different name (e.g. prefix "
                                         "with 'train/').")
                    metric_sums[k] = metric_sums.get(k, 0.0) + _coerce_log_scalar(
                        v,
                        where=("method.single_train_step()"
                               f".metrics[{k!r}]"),
                    )

            self.callbacks.on_before_optimizer_step(
                method,
                iteration=step,
            )
            method.optimizers_schedulers_step(step)
            method.optimizers_zero_grad(step)

            metrics = {k: v / grad_accum for k, v in loss_sums.items()}
            metrics.update({k: v / grad_accum for k, v in metric_sums.items()})
            metrics["step_time_sec"] = (time.perf_counter() - t0)
            metrics["vsa_sparsity"] = float(tc.vsa_sparsity)
            if self.global_rank == 0 and metrics:
                self.tracker.log(metrics, step)
            self._log_train_artifacts(
                method,
                last_outputs,
                step=step,
            )

            self.callbacks.on_training_step_end(
                method,
                metrics,
                iteration=step,
            )

            if checkpoint_manager is not None:
                checkpoint_manager.maybe_save(step)

            self.callbacks.on_validation_begin(
                method,
                iteration=step,
            )
            self.callbacks.on_validation_end(
                method,
                iteration=step,
            )
            if checkpoint_manager is not None:
                validation_cb = self.callbacks.get_callback("validation")
                latest_mf_metric: float | None = None
                get_latest_metric = getattr(
                    validation_cb,
                    "get_latest_metric",
                    None,
                )
                if callable(get_latest_metric):
                    latest_mf_metric = get_latest_metric(
                        "mf_angle_err_mean",
                        step=step,
                    )

                checkpoint_manager.maybe_save_best(
                    step=step,
                    metric_value=latest_mf_metric,
                    metric_name="mf_angle_err_mean",
                    start_step=int(
                        tc.checkpoint.best_checkpoint_start_step
                        or 0
                    ),
                    top_k=int(
                        tc.checkpoint.best_checkpoint_top_k
                        or 1
                    ),
                )

        self.callbacks.on_train_end(
            method,
            iteration=max_steps,
        )

        if checkpoint_manager is not None:
            checkpoint_manager.save_final(max_steps)

        self.tracker.finish()
