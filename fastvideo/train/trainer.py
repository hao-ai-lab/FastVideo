# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from tqdm.auto import tqdm

import fastvideo.envs as envs
from fastvideo.distributed import get_sp_group, get_world_group
from fastvideo.profiler import get_or_create_profiler
from fastvideo.train.callbacks.callback import CallbackDict
from fastvideo.train.methods.base import TrainingMethod
from fastvideo.train.utils.tracking import build_tracker

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )


def _coerce_log_scalar(
    value: Any,
    *,
    where: str,
) -> float | torch.Tensor:
    """Coerce *value* to a loggable scalar.

    GPU tensors stay on device so we avoid a
    ``cudaDeviceSynchronize`` per accumulation step.
    The caller must materialize them to float when logging.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor at {where}, "
                             f"got shape={tuple(value.shape)}")
        return value.detach()
    if isinstance(value, float | int):
        return float(value)
    raise TypeError(f"Expected a scalar (float/int/Tensor) at "
                    f"{where}, got {type(value).__name__}")


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
        # PyTorch profiler. No-op unless FASTVIDEO_TORCH_PROFILER_DIR is set.
        # Selected regions emit traces; see fastvideo/profiler.py for the
        # FASTVIDEO_TORCH_PROFILE_REGIONS list (e.g.
        # `profiler_region_training_train_one_step`).
        self.profiler_controller = get_or_create_profiler(envs.FASTVIDEO_TORCH_PROFILER_DIR)

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
        # Make the broad train region explicit so the controller's
        # `_active_region_depth` is back to zero before `stop()` runs.
        # `stop()` is in the `finally` so failed training runs still
        # flush/cleanup the profiler trace.
        try:
            with self.profiler_controller.region("profiler_region_training_train"):
                self._run_impl(
                    method,
                    dataloader=dataloader,
                    max_steps=max_steps,
                    start_step=start_step,
                    checkpoint_manager=checkpoint_manager,
                )
        finally:
            if self.profiler_controller.has_profiler:
                self.profiler_controller.stop()

    def _run_impl(
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
        self.callbacks.on_train_start(
            method,
            iteration=start_step,
        )

        resume_from_checkpoint = (tc.checkpoint.resume_from_checkpoint or "")
        if checkpoint_manager is not None:
            if resume_from_checkpoint:
                method.seed_optimizer_state_for_resume()
            resumed_step = (checkpoint_manager.maybe_resume(resume_from_checkpoint=(resume_from_checkpoint)))
            if resumed_step is not None:
                start_step = int(resumed_step)
        self.callbacks.on_validation_begin(
            method,
            iteration=start_step,
        )
        method.optimizers_zero_grad(start_step)

        data_stream = self._iter_dataloader(dataloader)

        # Restore the RNG snapshot LAST — after dcp.load,
        # after iter(dataloader), after everything that may
        # have advanced the RNG as a side-effect.
        if (checkpoint_manager is not None and resume_from_checkpoint):
            checkpoint_manager.load_rng_snapshot(resume_from_checkpoint, )
        progress = tqdm(
            range(start_step + 1, max_steps + 1),
            initial=start_step,
            desc="Steps",
            disable=self.local_rank > 0,
        )
        for step in progress:
            # Per-step profiler region. No-op unless
            # FASTVIDEO_TORCH_PROFILE_REGIONS includes
            # `profiler_region_training_train_one_step`.
            with self.profiler_controller.region("profiler_region_training_train_one_step"):
                t0 = time.perf_counter()

                # Accumulate on GPU during grad-accum; materialise
                # to CPU once per step right before logging.
                loss_sums: dict[str, float | torch.Tensor] = {}
                metric_sums: dict[str, float | torch.Tensor] = {}
                for accum_iter in range(grad_accum):
                    batch = next(data_stream)
                    loss_map, outputs, step_metrics = (method.single_train_step(
                        batch,
                        step,
                    ))

                    method.backward(
                        loss_map,
                        outputs,
                        grad_accum_rounds=grad_accum,
                    )

                    for k, v in loss_map.items():
                        if isinstance(v, torch.Tensor):
                            prev = loss_sums.get(k, 0.0)
                            loss_sums[k] = prev + v.detach()
                    for k, v in step_metrics.items():
                        if k in loss_sums:
                            raise ValueError(f"Metric key {k!r} collides "
                                             "with loss key. Use a "
                                             "different name (e.g. prefix "
                                             "with 'train/').")
                        prev = metric_sums.get(k, 0.0)
                        metric_sums[k] = (prev + _coerce_log_scalar(
                            v,
                            where=("method.single_train_step()"
                                   f".metrics[{k!r}]"),
                        ))

                self.callbacks.on_before_optimizer_step(
                    method,
                    iteration=step,
                )
                method.optimizers_schedulers_step(step)
                method.optimizers_zero_grad(step)

                # Single CPU sync point: materialise GPU tensors
                # to float right before logging.
                metrics = {k: float(v) / grad_accum for k, v in loss_sums.items()}
                metrics.update({k: float(v) / grad_accum for k, v in metric_sums.items()})
                metrics["step_time_sec"] = (time.perf_counter() - t0)
                metrics["vsa_sparsity"] = float(tc.vsa_sparsity)
                if self.global_rank == 0 and metrics:
                    self.tracker.log(metrics, step)

                self.callbacks.on_training_step_end(
                    method,
                    metrics,
                    iteration=step,
                )

                if checkpoint_manager is not None:
                    with self.profiler_controller.region("profiler_region_training_save_checkpoint"):
                        checkpoint_manager.maybe_save(step)

                with self.profiler_controller.region("profiler_region_training_validation"):
                    self.callbacks.on_validation_begin(
                        method,
                        iteration=step,
                    )
                    self.callbacks.on_validation_end(
                        method,
                        iteration=step,
                    )

            # Advance the torch.profiler schedule once per outer step so
            # `wait/warmup/active` cycles flush traces via on_trace_ready.
            # No-op when profiling is disabled.
            prof = self.profiler_controller.profiler
            if prof is not None:
                prof.step()

        self.callbacks.on_train_end(
            method,
            iteration=max_steps,
        )

        if checkpoint_manager is not None:
            checkpoint_manager.save_final(max_steps)

        self.tracker.finish()
