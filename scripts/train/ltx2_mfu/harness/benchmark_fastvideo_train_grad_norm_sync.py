#!/usr/bin/env python3
"""Scratch A/B for deferring grad-norm scalar materialization past AdamW launch."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import statistics
import time
from typing import Any

import torch
import torch.distributed as dist


EXPECTED_STEPS = 30
WARMUP_STEPS = 10
MFU_NUMERATOR = 14.444115

_step_times: list[float] = []
_step_starts: list[float] = []
_synced_end: float | None = None
_grad_norm_records: list[dict[str, Any]] = []
_pending_norm_logs: list[tuple[Any, str, torch.Tensor, int]] = []
_semantic_counts: Counter[str] = Counter()
_optimizer_returned_iteration = 0
_original_log: Any = None


def _rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _log(self: Any, metrics: dict[str, Any], step: int) -> None:
    if _original_log is None:
        raise RuntimeError("tracker log hook was not initialized")
    _original_log(self, metrics, step)
    for key, raw_value in metrics.items():
        if not key.startswith("grad_norm/"):
            continue
        value = float(raw_value)
        after_optimizer = _optimizer_returned_iteration >= int(step)
        _semantic_counts["grad_norm_log_calls"] += 1
        _semantic_counts[
            "grad_norm_logs_after_optimizer" if after_optimizer else "grad_norm_logs_before_optimizer"
        ] += 1
        _grad_norm_records.append({
            "step": int(step),
            "key": key,
            "value": value,
            "after_optimizer": after_optimizer,
        })
    if "step_time_sec" in metrics and _rank_zero():
        print(
            "BF16_STEP " + json.dumps({
                "step": step,
                "step_time_sec": float(metrics["step_time_sec"]),
            }),
            flush=True,
        )


def _flush_deferred_norm_logs(iteration: int) -> None:
    global _pending_norm_logs
    pending = _pending_norm_logs
    _pending_norm_logs = []
    for tracker, key, norm, queued_iteration in pending:
        if queued_iteration != iteration:
            raise RuntimeError(
                f"deferred grad norm from step {queued_iteration} reached optimizer step {iteration}"
            )
        value = float(norm.item())
        _semantic_counts["deferred_norm_materializations"] += 1
        if value > 0.0:
            tracker.log({key: value}, iteration)
        else:
            _semantic_counts["nonpositive_norm_log_skips"] += 1


def _wall_intervals(starts: list[float], synced_end: float) -> list[float]:
    if not starts:
        raise RuntimeError("no training-step wall starts were recorded")
    return [next_start - start for start, next_start in zip(starts, starts[1:])] + [synced_end - starts[-1]]


def _series_digest(records: list[dict[str, Any]]) -> str:
    semantic_series = [(record["step"], record["key"], record["value"]) for record in records]
    return hashlib.sha256(json.dumps(semantic_series, separators=(",", ":")).encode()).hexdigest()


def _self_test() -> None:
    global _optimizer_returned_iteration, _pending_norm_logs
    events: list[Any] = ["optimizer"]

    class _Norm:
        def item(self) -> float:
            events.append("item")
            return 1.25

    class _Tracker:
        def log(self, metrics: dict[str, float], step: int) -> None:
            events.append(("log", metrics, step))

    _optimizer_returned_iteration = 3
    _pending_norm_logs = [(_Tracker(), "grad_norm/student", _Norm(), 3)]  # type: ignore[list-item]
    _flush_deferred_norm_logs(3)
    assert events == ["optimizer", "item", ("log", {"grad_norm/student": 1.25}, 3)]
    assert _wall_intervals([1.0, 2.0, 4.0], 7.0) == [1.0, 2.0, 3.0]
    assert not _pending_norm_logs
    print("SELF_TEST_OK")


def main() -> None:
    global _optimizer_returned_iteration, _original_log, _synced_end

    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--defer-grad-norm-materialization", action="store_true")
    parser.add_argument("--source-deferred-grad-norm", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args, overrides = parser.parse_known_args()
    if args.self_test:
        _self_test()
        return
    if not args.config:
        parser.error("--config is required unless --self-test is used")
    if args.defer_grad_norm_materialization and args.source_deferred_grad_norm:
        parser.error("scratch and source-deferred modes are mutually exclusive")

    from fastvideo.training.trackers import DummyTracker

    _original_log = DummyTracker.log
    DummyTracker.log = _log

    import fastvideo.train.callbacks.grad_clip as grad_clip_module
    import fastvideo.train.trainer as trainer_module
    from fastvideo.distributed import get_world_group
    from fastvideo.train.callbacks.grad_clip import GradNormClipCallback
    from fastvideo.train.entrypoint.train import main as train_main
    from fastvideo.train.methods.base import TrainingMethod
    from fastvideo.training.training_utils import clip_grad_norm_while_handling_failing_dtensor_cases

    original_clip_grad_norm = grad_clip_module.clip_grad_norm_if_needed
    original_method_optimizer_step = TrainingMethod.optimizers_schedulers_step
    original_trainer_init = trainer_module.Trainer.__init__
    original_trainer_iter = trainer_module.Trainer._iter_dataloader
    original_trainer_run = trainer_module.Trainer.run

    def _counted_clip_grad_norm(module: torch.nn.Module, max_grad_norm: float) -> float:
        _semantic_counts["clip_calls"] += 1
        return original_clip_grad_norm(module, max_grad_norm)

    def _deferred_before_optimizer(
        self: GradNormClipCallback,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        if self._max_grad_norm <= 0.0:
            return
        if _pending_norm_logs:
            raise RuntimeError("previous deferred grad norm was not flushed")
        tracker = getattr(method, "tracker", None)
        for name, module in method.get_grad_clip_targets(iteration).items():
            _semantic_counts["clip_calls"] += 1
            norm = clip_grad_norm_while_handling_failing_dtensor_cases(
                [parameter for parameter in module.parameters()],
                self._max_grad_norm,
                foreach=None,
            )
            if norm is None:
                _semantic_counts["missing_norms"] += 1
            elif self._log_grad_norms and tracker is not None:
                _pending_norm_logs.append((tracker, f"grad_norm/{name}", norm, iteration))
                _semantic_counts["deferred_norm_queues"] += 1
                _semantic_counts["pending_peak"] = max(
                    _semantic_counts["pending_peak"],
                    len(_pending_norm_logs),
                )

    if args.defer_grad_norm_materialization:
        GradNormClipCallback.on_before_optimizer_step = _deferred_before_optimizer
    else:
        grad_clip_module.clip_grad_norm_if_needed = _counted_clip_grad_norm

    def _optimizer_step(self: TrainingMethod, iteration: int) -> None:
        global _optimizer_returned_iteration
        _semantic_counts["optimizer_method_calls"] += 1
        original_method_optimizer_step(self, iteration)
        _optimizer_returned_iteration = iteration
        if args.defer_grad_norm_materialization:
            _flush_deferred_norm_logs(iteration)

    TrainingMethod.optimizers_schedulers_step = _optimizer_step

    trainer_module.build_tracker = lambda *_args, **_kwargs: DummyTracker()

    class _Recorder:
        def on_training_step_end(self, _method: Any, metrics: dict[str, Any], iteration: int = 0) -> None:
            del iteration
            _step_times.append(float(metrics["step_time_sec"]))

    def _trainer_init(self: Any, *init_args: Any, **init_kwargs: Any) -> None:
        original_trainer_init(self, *init_args, **init_kwargs)
        if int(self.training_config.loop.gradient_accumulation_steps or 1) != 1:
            raise RuntimeError("scratch grad-norm benchmark requires gradient accumulation 1")
        grad_callbacks = [
            callback for callback in self.callbacks._callbacks.values()
            if isinstance(callback, GradNormClipCallback)
        ]
        if len(grad_callbacks) != 1 or not grad_callbacks[0]._log_grad_norms:
            raise RuntimeError("scratch grad-norm benchmark requires one logging GradNormClipCallback")
        self.callbacks._callbacks.pop("validation", None)
        self.callbacks._callbacks["_benchmark_recorder"] = _Recorder()

    def _timed_iter(self: Any, dataloader: Any) -> Any:
        iterator = original_trainer_iter(self, dataloader)
        while True:
            _step_starts.append(time.perf_counter())
            yield next(iterator)

    def _trainer_run(self: Any, method: TrainingMethod, **kwargs: Any) -> Any:
        global _synced_end
        vae = getattr(getattr(method, "student", None), "vae", None)
        if vae is not None:
            method.student.vae = None
            del vae
            gc.collect()
            torch.cuda.empty_cache()
            if _rank_zero():
                print("BF16_SETUP " + json.dumps({"unused_vae": "unloaded"}), flush=True)
        result = original_trainer_run(self, method, **kwargs)
        torch.cuda.synchronize()
        _semantic_counts["final_cuda_sync_calls"] += 1
        _synced_end = time.perf_counter()
        return result

    trainer_module.Trainer.__init__ = _trainer_init
    trainer_module.Trainer._iter_dataloader = _timed_iter
    trainer_module.Trainer.run = _trainer_run

    train_main(argparse.Namespace(config=args.config, dry_run=False), overrides=overrides or None)
    if _synced_end is None:
        raise RuntimeError("final synchronized wall endpoint was not recorded")
    wall_intervals = _wall_intervals(_step_starts, _synced_end)
    if len(_step_times) != EXPECTED_STEPS or len(wall_intervals) != EXPECTED_STEPS:
        raise RuntimeError(
            f"expected {EXPECTED_STEPS} steps, got internal={len(_step_times)} wall={len(wall_intervals)}"
        )
    if _pending_norm_logs:
        raise RuntimeError(f"{len(_pending_norm_logs)} deferred grad norms remain after training")

    payload = {
        "internal_step_times": _step_times,
        "wall_intervals": wall_intervals,
        "end_to_end_wall_sec": _synced_end - _step_starts[0],
        "measured_window_wall_sec": _synced_end - _step_starts[WARMUP_STEPS],
        "semantics": dict(_semantic_counts),
        "grad_norm_records": _grad_norm_records,
    }
    world = get_world_group()
    payloads: list[dict[str, Any] | None] = [None] * world.world_size
    dist.all_gather_object(payloads, payload, group=world.cpu_group)
    if world.rank != 0:
        return
    rank_payloads = [rank_payload for rank_payload in payloads if rank_payload is not None]
    if len(rank_payloads) != world.world_size:
        raise RuntimeError("missing rank payload")

    internal_slowest = [
        max(rank_payload["internal_step_times"][index] for rank_payload in rank_payloads)
        for index in range(EXPECTED_STEPS)
    ]
    wall_slowest = [
        max(rank_payload["wall_intervals"][index] for rank_payload in rank_payloads)
        for index in range(EXPECTED_STEPS)
    ]
    internal_measured = internal_slowest[WARMUP_STEPS:]
    wall_measured = wall_slowest[WARMUP_STEPS:]
    internal_median = statistics.median(internal_measured)
    wall_median = statistics.median(wall_measured)

    semantics_by_rank = []
    expected_relation = (
        "grad_norm_logs_after_optimizer"
        if args.defer_grad_norm_materialization or args.source_deferred_grad_norm
        else "grad_norm_logs_before_optimizer"
    )
    for rank, rank_payload in enumerate(rank_payloads):
        semantics = rank_payload["semantics"]
        records = rank_payload["grad_norm_records"]
        if semantics.get("clip_calls") != EXPECTED_STEPS:
            raise RuntimeError(f"rank {rank} recorded {semantics.get('clip_calls')} clip calls")
        if semantics.get("optimizer_method_calls") != EXPECTED_STEPS:
            raise RuntimeError(f"rank {rank} recorded {semantics.get('optimizer_method_calls')} optimizer calls")
        if semantics.get("grad_norm_log_calls") != EXPECTED_STEPS:
            raise RuntimeError(f"rank {rank} recorded {semantics.get('grad_norm_log_calls')} grad-norm logs")
        if semantics.get(expected_relation) != EXPECTED_STEPS:
            raise RuntimeError(f"rank {rank} failed expected log ordering: {semantics}")
        if args.defer_grad_norm_materialization and semantics.get("deferred_norm_materializations") != EXPECTED_STEPS:
            raise RuntimeError(f"rank {rank} failed to materialize every deferred norm")
        semantics_by_rank.append({
            "rank": rank,
            "counts": semantics,
            "grad_norm_digest": _series_digest(records),
            "first_grad_norm": records[0],
            "last_grad_norm": records[-1],
        })

    print(
        "BF16_TIMING " + json.dumps({
            "internal_slowest_rank_sec": internal_slowest,
            "true_wall_slowest_rank_sec": wall_slowest,
            "final_cuda_sync": True,
        }, sort_keys=True),
        flush=True,
    )
    print(
        "BF16_SEMANTICS " + json.dumps({
            "defer_grad_norm_materialization": args.defer_grad_norm_materialization,
            "source_deferred_grad_norm": args.source_deferred_grad_norm,
            "by_rank": semantics_by_rank,
            "rank0_grad_norm_series": rank_payloads[0]["grad_norm_records"],
        }, sort_keys=True),
        flush=True,
    )
    print(
        "BF16_RESULT " + json.dumps({
            "defer_grad_norm_materialization": args.defer_grad_norm_materialization,
            "source_deferred_grad_norm": args.source_deferred_grad_norm,
            "world_size": world.world_size,
            "internal_median_step_sec": internal_median,
            "internal_model_mfu_percent": MFU_NUMERATOR / internal_median,
            "true_wall_median_step_sec": wall_median,
            "true_wall_model_mfu_percent": MFU_NUMERATOR / wall_median,
            "true_wall_sum_slowest_intervals_sec": sum(wall_measured),
            "true_wall_mean_slowest_interval_sec": statistics.mean(wall_measured),
            "true_measured_window_max_rank_sec": max(
                rank_payload["measured_window_wall_sec"] for rank_payload in rank_payloads
            ),
            "true_end_to_end_max_rank_sec": max(
                rank_payload["end_to_end_wall_sec"] for rank_payload in rank_payloads
            ),
            "samples_per_second_from_true_wall_median": world.world_size / wall_median,
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
