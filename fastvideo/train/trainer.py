# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from fastvideo.distributed import get_sp_group, get_world_group
from fastvideo.train.callbacks.callback import CallbackDict
from fastvideo.train.methods.base import TrainingMethod
from fastvideo.train.utils.tracking import build_tracker

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )


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


@dataclass(slots=True)
class TrainLoopState:
    step: int
    accum_iter: int
    current_vsa_sparsity: float


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

    def _iter_dataloader(self, dataloader: Any) -> Iterator[dict[str, Any]]:
        data_iter = iter(dataloader)
        while True:
            batch = next(data_iter, None)
            if batch is None:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            yield batch

    def _get_current_vsa_sparsity(self, step: int) -> float:
        tc = self.training_config
        vsa_sparsity = tc.vsa.sparsity
        vsa_decay_rate = tc.vsa.decay_rate
        vsa_decay_interval_steps = (tc.vsa.decay_interval_steps)
        if vsa_decay_interval_steps > 1:
            current_decay_times = min(
                step // vsa_decay_interval_steps,
                int(vsa_sparsity // vsa_decay_rate),
            )
            return current_decay_times * vsa_decay_rate
        return vsa_sparsity

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
        self.callbacks.on_train_start(
            method, iteration=start_step,
        )

        resume_from_checkpoint = (tc.checkpoint.resume_from_checkpoint or "")
        if checkpoint_manager is not None:
            if resume_from_checkpoint:
                method.seed_optimizer_state_for_resume()
            resumed_step = (checkpoint_manager.maybe_resume(resume_from_checkpoint=(resume_from_checkpoint)))
            if resumed_step is not None:
                start_step = int(resumed_step)
        self.callbacks.on_validation_begin(
            method, iteration=start_step,
        )
        method.optimizers_zero_grad(start_step)

        data_stream = self._iter_dataloader(dataloader)

        # Restore the RNG snapshot LAST — after dcp.load,
        # after iter(dataloader), after everything that may
        # have advanced the RNG as a side-effect.
        if (
            checkpoint_manager is not None
            and resume_from_checkpoint
        ):
            checkpoint_manager.load_rng_snapshot(
                resume_from_checkpoint,
            )
        progress = tqdm(
            range(start_step + 1, max_steps + 1),
            initial=start_step,
            desc="Steps",
            disable=self.local_rank > 0,
        )
        for step in progress:
            t0 = time.perf_counter()
            current_vsa_sparsity = (self._get_current_vsa_sparsity(step))

            loss_sums: dict[str, float] = {}
            metric_sums: dict[str, float] = {}
            for accum_iter in range(grad_accum):
                batch = next(data_stream)
                loss_map, outputs, step_metrics = method.single_train_step(
                    batch,
                    step,
                    current_vsa_sparsity=(current_vsa_sparsity),
                )

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
                method, iteration=step,
            )
            method.optimizers_schedulers_step(step)
            method.optimizers_zero_grad(step)

            metrics = {k: v / grad_accum for k, v in loss_sums.items()}
            metrics.update({k: v / grad_accum for k, v in metric_sums.items()})
            metrics["step_time_sec"] = (time.perf_counter() - t0)
            metrics["vsa_sparsity"] = float(current_vsa_sparsity)
            if self.global_rank == 0 and metrics:
                self.tracker.log(metrics, step)

            self.callbacks.on_training_step_end(
                method, metrics, iteration=step,
            )

            if checkpoint_manager is not None:
                checkpoint_manager.maybe_save(step)

            self.callbacks.on_validation_begin(
                method, iteration=step,
            )
            self.callbacks.on_validation_end(
                method, iteration=step,
            )

        self.callbacks.on_train_end(
            method, iteration=max_steps,
        )

        if checkpoint_manager is not None:
            checkpoint_manager.save_final(max_steps)

        self.tracker.finish()
