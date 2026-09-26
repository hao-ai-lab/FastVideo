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
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.utils.performance import (
    TrainingPerformanceMonitor,
    infer_peak_bf16_tflops,
)
from fastvideo.train.utils.tracking import build_tracker

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )


def _verify_master_weight_precision(method: TrainingMethod, tc: TrainingConfig) -> None:
    """Refuse low-precision trainable masters for methods that require FP32.

    Optimizer steps applied in-place to bf16/fp16 parameters round away
    updates below ~half an ulp of each weight's magnitude; O(1)-magnitude
    parameters (norm gains) freeze entirely at typical distillation learning
    rates, and ``zeros_like``-allocated optimizer state inherits the same
    starved dtype. FP32 sharded masters (``training.dit_precision: fp32``)
    fix both; ordinary FSDP groups still compute in BF16.
    """
    if not getattr(method, "requires_fp32_master_weights", False):
        return
    if bool(getattr(tc.model, "allow_low_precision_master_weights", False)):
        return
    offenders: list[str] = []
    for role, model in getattr(method, "_role_models", {}).items():
        if not getattr(model, "_trainable", False):
            continue
        transformer = getattr(model, "transformer", None)
        if transformer is None:
            continue
        for name, param in transformer.named_parameters():
            if param.requires_grad and param.dtype != torch.float32:
                offenders.append(f"{role}:{name} ({param.dtype})")
                break
    if offenders:
        raise RuntimeError(f"{type(method).__name__} requires fp32 trainable master weights, got {offenders}. "
                           "bf16/fp16 parameter storage silently rounds away optimizer updates below ~half an ulp "
                           "per weight (norm-scale parameters freeze completely). Set training.dit_precision: fp32 "
                           "(FP32 sharded masters; ordinary groups compute in BF16), or acknowledge the effect "
                           "explicitly with training.model.allow_low_precision_master_weights: true.")


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
        self.performance_monitor = TrainingPerformanceMonitor()

    def _iter_dataloader(self,
                         dataloader: Any,
                         *,
                         initial_iterator: Iterator[dict[str, Any]] | None = None) -> Iterator[dict[str, Any]]:
        data_iter = initial_iterator if initial_iterator is not None else iter(dataloader)
        while True:
            batch = next(data_iter, None)
            if batch is None:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            yield batch

    def _run_method_validation(
        self,
        method: TrainingMethod,
        iteration: int,
    ) -> None:
        hook = getattr(method, "on_validation_begin", None)
        if hook is None:
            return
        validation_metrics: dict[str, LogScalar] = hook(iteration)
        validation_metrics = {
            k: float(_coerce_log_scalar(v, where=(f"method.on_validation_begin().metrics[{k!r}]")))
            for k, v in validation_metrics.items()
        }
        if self.global_rank == 0 and validation_metrics:
            self.tracker.log(validation_metrics, iteration)

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

        performance_enabled = bool(tc.performance.enabled)
        peak_tflops = tc.performance.peak_tflops_per_gpu
        if performance_enabled:
            if peak_tflops is None and torch.cuda.is_available():
                peak_tflops = infer_peak_bf16_tflops(torch.cuda.get_device_name(torch.cuda.current_device()))
            self.performance_monitor.attach(getattr(method, "_role_models", {}), )

        method.set_tracker(self.tracker)
        _verify_master_weight_precision(method, tc)
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
                if tc.checkpoint.reset_lr_on_resume:
                    method.apply_configured_lrs()
        if checkpoint_manager is not None:
            checkpoint_manager.maybe_save_inference(start_step,
                                                    validation_scheduled=self.callbacks.will_run_validation(start_step))
        self.callbacks.on_validation_begin(
            method,
            iteration=start_step,
        )
        self._run_method_validation(method, start_step)
        method.optimizers_zero_grad(start_step)

        data_stream = self._iter_dataloader(dataloader,
                                            initial_iterator=iter(dataloader) if resume_from_checkpoint else None)

        # Restore the RNG snapshot LAST — after dcp.load,
        # after iter(dataloader), after everything that may
        # have advanced the RNG as a side-effect.
        if (checkpoint_manager is not None and resume_from_checkpoint):
            checkpoint_manager.load_rng_snapshot(resume_from_checkpoint, )
        # Pre-loop forwards (teacher cache generation, step-0 validation) are
        # not part of a training step; drop them so a long cache build cannot
        # accumulate per-forward records.
        if performance_enabled:
            self.performance_monitor.reset()
        progress = tqdm(
            range(start_step + 1, max_steps + 1),
            initial=start_step,
            total=max_steps,
            desc="Steps",
            disable=self.local_rank > 0,
        )
        # Allow method-specific optimization flow (e.g. DiffusionNFT).
        method_manages_optimization = bool(method.manages_optimization())
        for step in progress:
            t0 = time.perf_counter()
            if performance_enabled:
                self.performance_monitor.reset()

            # Accumulate on GPU during grad-accum; materialise
            # to CPU once per step right before logging.
            loss_sums: dict[str, float | torch.Tensor] = {}
            metric_sums: dict[str, float | torch.Tensor] = {}
            if method_manages_optimization:
                loss_map, outputs, step_metrics = method.managed_train_step(
                    data_stream,
                    step,
                )
                for k, v in loss_map.items():
                    if isinstance(v, torch.Tensor):
                        loss_sums[k] = v.detach()
                for k, v in step_metrics.items():
                    if k in loss_sums:
                        raise ValueError(f"Metric key {k!r} collides "
                                         "with loss key. Use a "
                                         "different name (e.g. prefix "
                                         "with 'train/').")
                    metric_sums[k] = _coerce_log_scalar(
                        v,
                        where=("method.managed_train_step()"
                               f".metrics[{k!r}]"),
                    )
            else:
                use_no_sync = grad_accum > 1 and method.supports_gradient_accumulation_no_sync(step)
                for accum_iter in range(grad_accum):
                    if use_no_sync:
                        method.set_requires_gradient_sync(step, accum_iter == grad_accum - 1)
                    elif grad_accum > 1 and tc.loop.fsdp_defer_backward_finalize:
                        method.set_is_last_backward(step, accum_iter == grad_accum - 1)
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

            if not method_manages_optimization:
                self.callbacks.on_before_optimizer_step(
                    method,
                    iteration=step,
                )
                method.optimizers_schedulers_step(step)
                self.callbacks.on_after_optimizer_step(method, iteration=step)
                method.optimizers_zero_grad(step)

            # Single CPU sync point: materialise GPU tensors
            # to float right before logging.
            divisor = 1 if method_manages_optimization else grad_accum
            metrics = {k: float(v) / divisor for k, v in loss_sums.items()}
            metrics.update({k: float(v) / divisor for k, v in metric_sums.items()})
            step_time_sec = time.perf_counter() - t0
            metrics["step_time_sec"] = step_time_sec
            metrics["vsa_sparsity"] = float(tc.vsa_sparsity)
            if performance_enabled:
                metrics.update(
                    self.performance_monitor.metrics(
                        step_time_sec=step_time_sec,
                        local_batch_size=tc.data.train_batch_size,
                        grad_accum=(1 if method_manages_optimization else grad_accum),
                        world_size=getattr(
                            self.world_group,
                            "world_size",
                            tc.distributed.num_gpus,
                        ),
                        sp_size=getattr(
                            self.sp_group,
                            "world_size",
                            tc.distributed.sp_size,
                        ),
                        peak_tflops_per_gpu=peak_tflops,
                    ))
            if self.global_rank == 0 and metrics:
                self.tracker.log(metrics, step)

            self.callbacks.on_training_step_end(
                method,
                metrics,
                iteration=step,
            )

            if checkpoint_manager is not None and step != max_steps:
                # The final step is saved by ``save_final`` after validation
                # so the terminal checkpoint captures the post-validation
                # callback RNG state instead of being deduped against a
                # pre-validation interval save at the same step.
                checkpoint_manager.maybe_save(step)
                checkpoint_manager.maybe_save_inference(step,
                                                        validation_scheduled=self.callbacks.will_run_validation(step))

            self.callbacks.on_validation_begin(
                method,
                iteration=step,
            )
            self._run_method_validation(method, step)
            self.callbacks.on_validation_end(
                method,
                iteration=step,
            )

        self.callbacks.on_train_end(
            method,
            iteration=max_steps,
        )

        if checkpoint_manager is not None:
            checkpoint_manager.save_final(max_steps)

        self.performance_monitor.close()
        self.tracker.finish()
