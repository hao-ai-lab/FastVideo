# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from tqdm.auto import tqdm

from fastvideo.distributed import get_sp_group, get_world_group
from fastvideo.fastvideo_args import TrainingArgs
from fastvideo.training.trackers import BaseTracker, DummyTracker


@dataclass(slots=True)
class TrainLoopState:
    step: int
    accum_iter: int
    current_vsa_sparsity: float


class DistillTrainer:

    def __init__(self,
                 training_args: TrainingArgs,
                 *,
                 tracker: BaseTracker
                 | None = None) -> None:
        self.training_args = training_args
        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self.global_rank = self.world_group.rank
        self.local_rank = self.world_group.local_rank
        self.tracker = tracker or DummyTracker()

    def _iter_dataloader(self, dataloader: Any) -> Iterator[dict[str, Any]]:
        data_iter = iter(dataloader)
        while True:
            batch = next(data_iter, None)
            if batch is None:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            yield batch

    def _get_current_vsa_sparsity(self, step: int) -> float:
        # Keep behavior close to existing pipelines.
        vsa_sparsity = self.training_args.VSA_sparsity
        vsa_decay_rate = self.training_args.VSA_decay_rate
        vsa_decay_interval_steps = self.training_args.VSA_decay_interval_steps
        if vsa_decay_interval_steps > 1:
            current_decay_times = min(step // vsa_decay_interval_steps,
                                      int(vsa_sparsity // vsa_decay_rate))
            return current_decay_times * vsa_decay_rate
        return vsa_sparsity

    def run(
        self,
        method: torch.nn.Module,
        *,
        dataloader: Any,
        max_steps: int,
        start_step: int = 0,
        checkpoint_manager: Any | None = None,
    ) -> None:
        grad_accum = max(
            1, int(self.training_args.gradient_accumulation_steps or 1))

        if hasattr(method, "on_train_start"):
            method.on_train_start()  # type: ignore[attr-defined]

        resume_from_checkpoint = getattr(self.training_args,
                                         "resume_from_checkpoint", "") or ""
        if checkpoint_manager is not None:
            resumed_step = checkpoint_manager.maybe_resume(
                resume_from_checkpoint=resume_from_checkpoint)
            if resumed_step is not None:
                start_step = int(resumed_step)

        validation_interval = int(self.training_args.validation_steps or 0)
        if (getattr(self.training_args, "log_validation", False)
                and validation_interval > 0
                and hasattr(method, "log_validation")):
            method.log_validation(start_step)  # type: ignore[attr-defined]

        if hasattr(method, "optimizers_zero_grad"):
            method.optimizers_zero_grad(
                start_step)  # type: ignore[attr-defined]

        data_stream = self._iter_dataloader(dataloader)
        progress = tqdm(
            range(start_step + 1, max_steps + 1),
            initial=start_step,
            desc="Steps",
            disable=self.local_rank > 0,
        )
        for step in progress:
            t0 = time.perf_counter()
            current_vsa_sparsity = self._get_current_vsa_sparsity(step)

            loss_sums: dict[str, float] = {}
            for accum_iter in range(grad_accum):
                batch = next(data_stream)
                if hasattr(method, "single_train_step"):
                    loss_map, outputs = method.single_train_step(  # type: ignore[attr-defined]
                        batch,
                        step,
                        current_vsa_sparsity=current_vsa_sparsity,
                    )
                else:
                    raise AttributeError(
                        "method must implement single_train_step()")

                if hasattr(method, "backward"):
                    method.backward(  # type: ignore[attr-defined]
                        loss_map,
                        outputs,
                        grad_accum_rounds=grad_accum,
                    )
                else:
                    total_loss = loss_map["total_loss"] / grad_accum
                    total_loss.backward()

                for k, v in loss_map.items():
                    if isinstance(v, torch.Tensor):
                        loss_sums[k] = loss_sums.get(k, 0.0) + float(
                            v.detach().item())

            if hasattr(method, "optimizers_schedulers_step"):
                method.optimizers_schedulers_step(
                    step)  # type: ignore[attr-defined]
            if hasattr(method, "optimizers_zero_grad"):
                method.optimizers_zero_grad(step)  # type: ignore[attr-defined]

            metrics = {k: v / grad_accum for k, v in loss_sums.items()}
            metrics["step_time_sec"] = time.perf_counter() - t0
            if self.global_rank == 0 and metrics:
                self.tracker.log(metrics, step)

            if checkpoint_manager is not None:
                checkpoint_manager.maybe_save(step)

            if (getattr(self.training_args, "log_validation", False)
                    and validation_interval > 0
                    and step % validation_interval == 0
                    and hasattr(method, "log_validation")):
                method.log_validation(step)  # type: ignore[attr-defined]

        if checkpoint_manager is not None:
            checkpoint_manager.save_final(max_steps)

        self.tracker.finish()
