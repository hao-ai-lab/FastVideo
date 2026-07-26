#!/usr/bin/env python3
"""Scratch A/B for embedding one uniform LTX-2 T2V timestep per sample."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
import statistics
import time
from typing import Any

import torch
import torch.distributed as dist


EXPECTED_STEPS = 30
WARMUP_STEPS = 10
MFU_NUMERATOR = 14.444115
LOCAL_BATCH_SIZE = int(os.environ.get("FASTVIDEO_BENCH_LOCAL_BATCH_SIZE", "1"))
if LOCAL_BATCH_SIZE <= 0:
    raise ValueError("FASTVIDEO_BENCH_LOCAL_BATCH_SIZE must be positive")
GRAD_PROBE_STEP = WARMUP_STEPS

_step_times: list[float] = []
_step_starts: list[float] = []
_synced_end: float | None = None
_metrics_by_step: dict[int, dict[str, float]] = {}
_semantic_counts: Counter[str] = Counter()
_ada_grad_probe: dict[str, Any] | None = None
_peak_memory: dict[str, int] = {}
_optimizer_probe: dict[str, Any] = {}
_original_log: Any = None


def _rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _log(self: Any, metrics: dict[str, Any], step: int) -> None:
    if _original_log is None:
        raise RuntimeError("tracker log hook was not initialized")
    _original_log(self, metrics, step)
    record = _metrics_by_step.setdefault(int(step), {})
    for key, value in metrics.items():
        if key in {"total_loss", "finetune_loss", "step_time_sec"} or key.startswith("grad_norm/"):
            record[key] = float(value)
    if "step_time_sec" in metrics and _rank_zero():
        print(
            "BF16_STEP " + json.dumps({
                "step": int(step),
                "step_time_sec": float(metrics["step_time_sec"]),
            }),
            flush=True,
        )


def _wall_intervals(starts: list[float], synced_end: float) -> list[float]:
    if not starts:
        raise RuntimeError("no training-step wall starts were recorded")
    return [next_start - start for start, next_start in zip(starts, starts[1:])] + [synced_end - starts[-1]]


def _tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _capture_ada_grad_probe(method: Any, iteration: int) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for name, parameter in method.student.transformer.named_parameters():
        if "adaln_single" not in name:
            continue
        gradient = parameter.grad
        if gradient is None:
            parameters[name] = {"present": False}
            continue
        if isinstance(gradient, torch.distributed.tensor.DTensor):
            gradient = gradient.to_local()
        local = gradient.detach().contiguous().cpu()
        local_float = local.float()
        parameters[name] = {
            "present": True,
            "shape": list(local.shape),
            "dtype": str(local.dtype),
            "numel": local.numel(),
            "finite": bool(torch.isfinite(local).all()),
            "l2_norm": float(torch.linalg.vector_norm(local_float)),
            "max_abs": float(local_float.abs().max()),
            "mean": float(local_float.mean()),
            "sha256": _tensor_digest(local),
        }
    if not parameters:
        raise RuntimeError("found no adaln_single parameters for the gradient probe")
    if not all(item.get("finite", False) for item in parameters.values() if item.get("present")):
        raise RuntimeError("non-finite Ada gradient in probe")
    return {"step": int(iteration), "parameters": parameters}


def _capture_optimizer_probe(method: Any) -> dict[str, Any]:
    optimizers = list(method.get_optimizers(0))
    if len(optimizers) != 1:
        raise RuntimeError(f"expected one optimizer, got {len(optimizers)}")
    optimizer = optimizers[0]
    use_te_master = os.environ.get("FASTVIDEO_TE_FP32_MASTER") == "1"
    optimizer_class = f"{type(optimizer).__module__}.{type(optimizer).__name__}"
    if use_te_master:
        if type(optimizer).__name__ != "FusedAdam" or not type(optimizer).__module__.startswith(
            "transformer_engine."
        ):
            raise RuntimeError(f"TE master mode constructed the wrong optimizer: {optimizer_class}")
    elif not isinstance(optimizer, torch.optim.AdamW):
        raise RuntimeError(f"stock control constructed the wrong optimizer: {optimizer_class}")

    trainable_parameters = [
        parameter for parameter in method.student.transformer.parameters() if parameter.requires_grad
    ]
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    trainable_parameter_ids = [id(parameter) for parameter in trainable_parameters]
    optimizer_parameter_ids = [id(parameter) for parameter in optimizer_parameters]
    if len(optimizer_parameter_ids) != len(set(optimizer_parameter_ids)):
        raise RuntimeError("optimizer contains duplicate parameter objects")
    if set(trainable_parameter_ids) != set(optimizer_parameter_ids):
        raise RuntimeError(
            "optimizer parameter coverage differs from trainable transformer parameters: "
            f"trainable={len(trainable_parameter_ids)} optimizer={len(optimizer_parameter_ids)}"
        )

    param_dtypes: set[str] = set()
    state_dtypes: dict[str, set[str]] = {}
    missing_state: dict[str, int] = {}
    writeback_mismatches = torch.zeros((), dtype=torch.int64, device="cuda")
    parameter_count = 0
    parameter_numel = 0
    for parameter in trainable_parameters:
        if not isinstance(parameter, torch.distributed.tensor.DTensor):
            raise RuntimeError(f"registered parameter is not a DTensor shard: {type(parameter).__name__}")
        parameter_count += 1
        parameter_numel += parameter.numel()
        param_dtypes.add(str(parameter.dtype))
        state = optimizer.state.get(parameter)
        if state is None:
            raise RuntimeError("trainable optimizer parameter has no state")
        for name in ("master_param", "exp_avg", "exp_avg_sq"):
            state_tensor = state.get(name)
            if state_tensor is None:
                missing_state[name] = missing_state.get(name, 0) + 1
                continue
            if not isinstance(state_tensor, torch.distributed.tensor.DTensor):
                raise RuntimeError(f"optimizer state {name} is not a DTensor shard: {type(state_tensor).__name__}")
            if (
                state_tensor.shape != parameter.shape
                or state_tensor.stride() != parameter.stride()
                or state_tensor.placements != parameter.placements
                or state_tensor.device_mesh.device_type != parameter.device_mesh.device_type
                or not torch.equal(state_tensor.device_mesh.mesh, parameter.device_mesh.mesh)
                or state_tensor.to_local().shape != parameter.to_local().shape
                or state_tensor.to_local().stride() != parameter.to_local().stride()
            ):
                raise RuntimeError(
                    f"optimizer state {name} layout does not match its registered parameter"
                )
            state_dtypes.setdefault(name, set()).add(str(state_tensor.dtype))
        if use_te_master:
            master = state.get("master_param")
            if master is None:
                raise RuntimeError("TE optimizer parameter is missing master_param")
            local_parameter = parameter.to_local().detach()
            local_master = master.to_local().detach()
            if not torch.equal(local_parameter, local_master.to(local_parameter.dtype)):
                writeback_mismatches += 1

    expected_param_dtype = {"torch.bfloat16"} if use_te_master else {"torch.float32"}
    if param_dtypes != expected_param_dtype:
        raise RuntimeError(f"unexpected registered parameter dtypes: {param_dtypes}")
    if state_dtypes.get("exp_avg") != {"torch.float32"} or state_dtypes.get("exp_avg_sq") != {"torch.float32"}:
        raise RuntimeError(f"optimizer moments are not FP32: {state_dtypes}")
    if missing_state.get("exp_avg", 0) or missing_state.get("exp_avg_sq", 0):
        raise RuntimeError(f"missing optimizer moment states: {missing_state}")
    if use_te_master:
        if state_dtypes.get("master_param") != {"torch.float32"}:
            raise RuntimeError(f"optimizer masters are not FP32: {state_dtypes}")
        if missing_state.get("master_param", 0):
            raise RuntimeError(f"missing TE master states: {missing_state}")
    elif state_dtypes.get("master_param") or missing_state.get("master_param", 0) != parameter_count:
        raise RuntimeError(f"stock optimizer unexpectedly contains master states: {state_dtypes}, {missing_state}")

    if dist.is_initialized():
        dist.all_reduce(writeback_mismatches, op=dist.ReduceOp.SUM)
    total_writeback_mismatches = int(writeback_mismatches.item())
    if total_writeback_mismatches:
        raise RuntimeError(
            "registered BF16 parameters differ from rounded FP32 masters: "
            f"mismatches_across_ranks={total_writeback_mismatches}"
        )
    return {
        "class": optimizer_class,
        "parameter_count": parameter_count,
        "parameter_numel": parameter_numel,
        "parameter_dtypes": sorted(param_dtypes),
        "state_dtypes": {name: sorted(dtypes) for name, dtypes in sorted(state_dtypes.items())},
        "missing_state": missing_state,
        "master_writeback_mismatches_across_ranks": total_writeback_mismatches,
        "te_fp32_master": use_te_master,
    }


def _series_digest(metrics_by_step: dict[int, dict[str, float]]) -> str:
    series = [(step, sorted(metrics.items())) for step, metrics in sorted(metrics_by_step.items())]
    return hashlib.sha256(json.dumps(series, separators=(",", ":")).encode()).hexdigest()


def _self_test() -> None:
    assert _wall_intervals([1.0, 2.0, 4.0], 7.0) == [1.0, 2.0, 3.0]
    assert len(hashlib.sha256(b"test").hexdigest()) == 64
    print("SELF_TEST_OK")


def main() -> None:
    global _ada_grad_probe, _optimizer_probe, _original_log, _peak_memory, _synced_end

    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--singleton-timestep", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args, overrides = parser.parse_known_args()
    if args.self_test:
        _self_test()
        return
    if not args.config:
        parser.error("--config is required unless --self-test is used")

    from fastvideo.training.trackers import DummyTracker

    _original_log = DummyTracker.log
    DummyTracker.log = _log

    import fastvideo.train.trainer as trainer_module
    from fastvideo.distributed import get_world_group
    from fastvideo.train.callbacks.grad_clip import GradNormClipCallback
    from fastvideo.train.entrypoint.train import main as train_main
    from fastvideo.train.methods.base import TrainingMethod
    from fastvideo.train.models.ltx2 import LTX2Model

    original_build_kwargs = LTX2Model._build_distill_input_kwargs
    original_method_optimizer_step = TrainingMethod.optimizers_schedulers_step
    original_trainer_init = trainer_module.Trainer.__init__
    original_trainer_iter = trainer_module.Trainer._iter_dataloader
    original_trainer_run = trainer_module.Trainer.run

    def _build_distill_input_kwargs(self: LTX2Model, *build_args: Any, **build_kwargs: Any) -> dict[str, Any]:
        original_token_count = getattr(self, "_token_count", None)
        if original_token_count is None:
            raise RuntimeError("LTX-2 token count was not initialized before building transformer inputs")
        _semantic_counts[f"semantic_token_count_{int(original_token_count)}"] += 1
        if not args.singleton_timestep:
            result = original_build_kwargs(self, *build_args, **build_kwargs)
        else:
            if int(self.training_config.distributed.sp_size or 1) != 1:
                raise RuntimeError("singleton-timestep scratch path is restricted to SP=1")
            # The source method uses _token_count only to expand the uniform
            # per-sample sigma. Temporarily setting it to one reproduces the
            # proposed source path without editing the checkout.
            self._token_count = 1
            try:
                result = original_build_kwargs(self, *build_args, **build_kwargs)
            finally:
                self._token_count = original_token_count
        timestep = result.get("timestep")
        if not isinstance(timestep, torch.Tensor) or timestep.ndim != 2:
            raise RuntimeError(f"unexpected LTX-2 timestep: {type(timestep).__name__}")
        _semantic_counts[f"model_timestep_tokens_{int(timestep.shape[1])}"] += 1
        return result

    LTX2Model._build_distill_input_kwargs = _build_distill_input_kwargs

    def _optimizer_step(self: TrainingMethod, iteration: int) -> None:
        global _ada_grad_probe
        if int(iteration) == GRAD_PROBE_STEP:
            _ada_grad_probe = _capture_ada_grad_probe(self, iteration)
            _semantic_counts["ada_grad_probes"] += 1
        original_method_optimizer_step(self, iteration)

    TrainingMethod.optimizers_schedulers_step = _optimizer_step
    trainer_module.build_tracker = lambda *_args, **_kwargs: DummyTracker()

    class _Recorder:
        def on_training_step_end(self, _method: Any, metrics: dict[str, Any], iteration: int = 0) -> None:
            del iteration
            _step_times.append(float(metrics["step_time_sec"]))

    def _trainer_init(self: Any, *init_args: Any, **init_kwargs: Any) -> None:
        original_trainer_init(self, *init_args, **init_kwargs)
        if int(self.training_config.loop.gradient_accumulation_steps or 1) != 1:
            raise RuntimeError("scratch singleton benchmark requires gradient accumulation 1")
        if int(self.training_config.distributed.sp_size or 1) != 1:
            raise RuntimeError("scratch singleton benchmark requires SP=1")
        grad_callbacks = [
            callback for callback in self.callbacks._callbacks.values()
            if isinstance(callback, GradNormClipCallback)
        ]
        if len(grad_callbacks) != 1 or not grad_callbacks[0]._log_grad_norms:
            raise RuntimeError("scratch singleton benchmark requires one logging GradNormClipCallback")
        self.callbacks._callbacks.pop("validation", None)
        self.callbacks._callbacks["_benchmark_recorder"] = _Recorder()

    def _timed_iter(self: Any, dataloader: Any) -> Any:
        iterator = original_trainer_iter(self, dataloader)
        while True:
            _step_starts.append(time.perf_counter())
            yield next(iterator)

    def _trainer_run(self: Any, method: TrainingMethod, **kwargs: Any) -> Any:
        global _optimizer_probe, _peak_memory, _synced_end
        vae = getattr(getattr(method, "student", None), "vae", None)
        if vae is not None:
            method.student.vae = None
            del vae
            gc.collect()
            torch.cuda.empty_cache()
            if _rank_zero():
                print("BF16_SETUP " + json.dumps({"unused_vae": "unloaded"}), flush=True)
        torch.cuda.reset_peak_memory_stats()
        result = original_trainer_run(self, method, **kwargs)
        torch.cuda.synchronize()
        _synced_end = time.perf_counter()
        _peak_memory = {
            "allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        _optimizer_probe = _capture_optimizer_probe(method)
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
    if _ada_grad_probe is None or _semantic_counts["ada_grad_probes"] != 1:
        raise RuntimeError("Ada gradient probe did not run exactly once")

    payload = {
        "internal_step_times": _step_times,
        "wall_intervals": wall_intervals,
        "end_to_end_wall_sec": _synced_end - _step_starts[0],
        "measured_window_wall_sec": _synced_end - _step_starts[WARMUP_STEPS],
        "metrics_by_step": _metrics_by_step,
        "metric_series_digest": _series_digest(_metrics_by_step),
        "semantics": dict(_semantic_counts),
        "ada_grad_probe": _ada_grad_probe,
        "optimizer_probe": _optimizer_probe,
        "peak_memory": _peak_memory,
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

    expected_model_tokens = 1 if args.singleton_timestep else None
    semantics_by_rank = []
    for rank, rank_payload in enumerate(rank_payloads):
        semantics = rank_payload["semantics"]
        semantic_token_counts = {
            int(key.rsplit("_", 1)[1]): count
            for key, count in semantics.items()
            if key.startswith("semantic_token_count_")
        }
        model_token_counts = {
            int(key.rsplit("_", 1)[1]): count
            for key, count in semantics.items()
            if key.startswith("model_timestep_tokens_")
        }
        if sum(semantic_token_counts.values()) != EXPECTED_STEPS or len(semantic_token_counts) != 1:
            raise RuntimeError(f"rank {rank} recorded invalid semantic token counts: {semantic_token_counts}")
        if sum(model_token_counts.values()) != EXPECTED_STEPS:
            raise RuntimeError(f"rank {rank} recorded invalid timestep counts: {model_token_counts}")
        if expected_model_tokens is not None and model_token_counts != {expected_model_tokens: EXPECTED_STEPS}:
            raise RuntimeError(f"rank {rank} did not use singleton timesteps: {model_token_counts}")
        if expected_model_tokens is None and model_token_counts != semantic_token_counts:
            raise RuntimeError(
                f"rank {rank} control changed timestep shape: semantic={semantic_token_counts}, model={model_token_counts}"
            )
        semantics_by_rank.append({
            "rank": rank,
            "counts": semantics,
            "metric_series_digest": rank_payload["metric_series_digest"],
            "peak_memory": rank_payload["peak_memory"],
            "ada_grad_probe": rank_payload["ada_grad_probe"],
            "optimizer_probe": rank_payload["optimizer_probe"],
            "first_metrics": rank_payload["metrics_by_step"].get(1, {}),
            "last_metrics": rank_payload["metrics_by_step"].get(EXPECTED_STEPS, {}),
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
            "singleton_timestep": args.singleton_timestep,
            "gradient_probe_step": GRAD_PROBE_STEP,
            "by_rank": semantics_by_rank,
            "rank0_metric_series": rank_payloads[0]["metrics_by_step"],
        }, sort_keys=True),
        flush=True,
    )
    print(
        "BF16_RESULT " + json.dumps({
            "singleton_timestep": args.singleton_timestep,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "world_size": world.world_size,
            "internal_median_step_sec": internal_median,
            "internal_model_mfu_percent": MFU_NUMERATOR * LOCAL_BATCH_SIZE / internal_median,
            "true_wall_median_step_sec": wall_median,
            "true_wall_model_mfu_percent": MFU_NUMERATOR * LOCAL_BATCH_SIZE / wall_median,
            "true_wall_sum_slowest_intervals_sec": sum(wall_measured),
            "true_wall_mean_slowest_interval_sec": statistics.mean(wall_measured),
            "true_measured_window_max_rank_sec": max(
                rank_payload["measured_window_wall_sec"] for rank_payload in rank_payloads
            ),
            "true_end_to_end_max_rank_sec": max(
                rank_payload["end_to_end_wall_sec"] for rank_payload in rank_payloads
            ),
            "samples_per_second_from_true_wall_median": (
                world.world_size * LOCAL_BATCH_SIZE / wall_median
            ),
            "peak_allocated_max_rank_bytes": max(
                rank_payload["peak_memory"]["allocated_bytes"] for rank_payload in rank_payloads
            ),
            "peak_reserved_max_rank_bytes": max(
                rank_payload["peak_memory"]["reserved_bytes"] for rank_payload in rank_payloads
            ),
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
