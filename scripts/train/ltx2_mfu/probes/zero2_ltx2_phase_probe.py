#!/usr/bin/env python3
"""Disposable inclusive phase probe for the four-GB200 LTX-2 fixed arena."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("FASTVIDEO_FA4", "1")
os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")

import torch
import torch.distributed as dist


BASE_PATH = Path(__file__).with_name("zero2_ltx2_probe.py")
EXPECTED_BASE_SHA = "7ff05aafe045a53e754a88c4842fe7be33606440e2b1a8f18718f2793c24fff6"
EXPECTED_SHA = "7f139e2b28610063d2f30526ba8f0ccae5d88944"
AG_WAITS_PER_STEP = 49
GPU_CLOSURE_TOLERANCE_MS = 0.05


def _load_base() -> Any:
    digest = hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_BASE_SHA:
        raise RuntimeError(f"expected base probe {EXPECTED_BASE_SHA}, got {digest}")
    spec = importlib.util.spec_from_file_location("zero2_ltx2_probe_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base probe {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


class StepEvents:
    """All events for one step, allocated before warmup."""

    def __init__(self) -> None:
        self.gpu_start = _event()
        self.after_zero = _event()
        self.after_forward = _event()
        self.after_backward = _event()
        self.rs_wait_start = _event()
        self.rs_wait_end = _event()
        self.grad_copy_start = _event()
        self.grad_copy_end = _event()
        self.norm_clip_start = _event()
        self.norm_clip_end = _event()
        self.adamw_start = _event()
        self.adamw_end = _event()
        self.copy_gather_start = _event()
        self.copy_gather_end = _event()
        self.gpu_end = _event()
        self.ag_wait_pairs = tuple((_event(), _event()) for _ in range(AG_WAITS_PER_STEP))
        self.ag_wait_count = 0

    def all_events(self) -> tuple[torch.cuda.Event, ...]:
        boundaries = (
            self.gpu_start,
            self.after_zero,
            self.after_forward,
            self.after_backward,
            self.rs_wait_start,
            self.rs_wait_end,
            self.grad_copy_start,
            self.grad_copy_end,
            self.norm_clip_start,
            self.norm_clip_end,
            self.adamw_start,
            self.adamw_end,
            self.copy_gather_start,
            self.copy_gather_end,
            self.gpu_end,
        )
        return boundaries + tuple(event for pair in self.ag_wait_pairs for event in pair)


class ProfiledZero2(base.LTX2FixedArenaZero2):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._profile_events: StepEvents | None = None

    def begin_profile_step(self, events: StepEvents) -> None:
        if self._profile_events is not None:
            raise RuntimeError("previous profile step was not closed")
        events.ag_wait_count = 0
        self._profile_events = events

    def end_profile_step(self) -> None:
        if self._profile_events is None:
            raise RuntimeError("profile step was not started")
        self._profile_events = None

    def _wait_gather(self, bucket_id: int) -> None:
        work = self.buckets[bucket_id].gather_work
        if work is None:
            return
        events = self._profile_events
        if events is None:
            work.wait()
        else:
            index = events.ag_wait_count
            if index >= AG_WAITS_PER_STEP:
                raise RuntimeError(f"more than {AG_WAITS_PER_STEP} all-gather waits in one step")
            start, end = events.ag_wait_pairs[index]
            start.record()
            work.wait()
            end.record()
            events.ag_wait_count += 1
        self.buckets[bucket_id].gather_work = None

    @torch.no_grad()
    def step(self, max_grad_norm: float) -> torch.Tensor:
        events = self._profile_events
        if events is None:
            raise RuntimeError("profile events must be active during step")
        if not self._active:
            raise RuntimeError("step called without zero_grad/backward")
        self._launch_ready_buckets()
        if self._reduce_cursor != len(self._reduce_order):
            missing = sum(bucket.ready for bucket in self.buckets)
            raise RuntimeError(f"backward left {missing} parameter gradients unused")

        events.rs_wait_start.record()
        for bucket in self.buckets:
            bucket.reduce_work.wait()
        events.rs_wait_end.record()

        master_grads = [bucket.master_grad for bucket in self.buckets]
        reduced_shards = [bucket.grad.view(4, -1)[self.rank] for bucket in self.buckets]
        events.grad_copy_start.record()
        torch._foreach_copy_(master_grads, reduced_shards)
        events.grad_copy_end.record()

        events.norm_clip_start.record()
        norm_sq = torch.stack(torch._foreach_norm(master_grads, 2.0)).square().sum()
        dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM, group=self.group)
        total_norm = norm_sq.sqrt()
        torch._foreach_mul_(master_grads, (max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0))
        events.norm_clip_end.record()

        events.adamw_start.record()
        base.optim_f.adamw(
            [bucket.master for bucket in self.buckets],
            master_grads,
            [bucket.exp_avg for bucket in self.buckets],
            [bucket.exp_avg_sq for bucket in self.buckets],
            [],
            [bucket.step for bucket in self.buckets],
            foreach=False,
            capturable=True,
            differentiable=False,
            fused=True,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
            amsgrad=False,
            beta1=self.betas[0],
            beta2=self.betas[1],
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.eps,
            maximize=False,
        )
        events.adamw_end.record()

        local_param_shards = [bucket.flat.view(4, -1)[self.rank] for bucket in self.buckets]
        events.copy_gather_start.record()
        torch._foreach_copy_(local_param_shards, [bucket.master for bucket in self.buckets])
        for bucket, local_shard in zip(self.buckets, local_param_shards, strict=True):
            bucket.gather_work = dist.all_gather_into_tensor(
                bucket.flat,
                local_shard,
                group=self.group,
                async_op=True,
            )
        events.copy_gather_end.record()
        self._active = False
        return total_norm


def _elapsed(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


def _phase_record(events: StepEvents, host: dict[str, float | int]) -> dict[str, Any]:
    exposed_ag = sum(_elapsed(*events.ag_wait_pairs[index]) for index in range(events.ag_wait_count))
    forward_total = _elapsed(events.after_zero, events.after_forward)
    phases = {
        "data": float(host["data_ms"]),
        "exposed_all_gather_wait": exposed_ag,
        "forward_compute": forward_total - exposed_ag,
        "backward_overlapped_reduce_scatter": _elapsed(events.after_forward, events.after_backward),
        "exposed_reduce_scatter_wait": _elapsed(events.rs_wait_start, events.rs_wait_end),
        "grad_shard_copy": _elapsed(events.grad_copy_start, events.grad_copy_end),
        "grad_norm_clip": _elapsed(events.norm_clip_start, events.norm_clip_end),
        "adamw": _elapsed(events.adamw_start, events.adamw_end),
        "bf16_copy_all_gather_launch": _elapsed(events.copy_gather_start, events.copy_gather_end),
    }
    gpu_other = sum((
        _elapsed(events.gpu_start, events.after_zero),
        _elapsed(events.after_backward, events.rs_wait_start),
        _elapsed(events.rs_wait_end, events.grad_copy_start),
        _elapsed(events.grad_copy_end, events.norm_clip_start),
        _elapsed(events.norm_clip_end, events.adamw_start),
        _elapsed(events.adamw_end, events.copy_gather_start),
        _elapsed(events.copy_gather_end, events.gpu_end),
    ))
    gpu_span = _elapsed(events.gpu_start, events.gpu_end)
    phases["gpu_other"] = gpu_other
    gpu_phase_sum = sum(value for key, value in phases.items() if key != "data")
    phases["host_other"] = float(host["wall_ms"]) - phases["data"] - gpu_span
    phase_sum = sum(phases.values())
    return {
        "global_step": int(host["global_step"]),
        "wall_ms": float(host["wall_ms"]),
        "gpu_span_ms": gpu_span,
        "phase_ms": phases,
        "phase_sum_ms": phase_sum,
        "gpu_phase_sum_ms": gpu_phase_sum,
        "gpu_closure_error_ms": abs(gpu_span - gpu_phase_sum),
        "total_closure_error_ms": abs(float(host["wall_ms"]) - phase_sum),
        "all_gather_wait_count": events.ag_wait_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/train/configs/overfit_ltx2_t2v.yaml")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gate-ms", type=float, default=419.0)
    parser.add_argument("--expected-sha", default=EXPECTED_SHA)
    parser.add_argument("--max-event-overhead-percent", type=float, default=1.0)
    args, user_overrides = parser.parse_known_args()
    if args.warmup < 1 or args.steps < 1:
        raise ValueError("warmup and steps must be positive")

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.expected_sha and head != args.expected_sha:
        raise RuntimeError(f"expected FastVideo {args.expected_sha}, got {head}")

    fixed_overrides = [
        "--models.student.attention_backend", "FLASH_ATTN",
        "--models.student.enable_gradient_checkpointing_type", "null",
        "--training.model.enable_gradient_checkpointing_type", "null",
        "--training.model.enable_torch_compile", "true",
        "--training.distributed.reshard_after_forward", "true",
        "--training.distributed.reduce_dtype", "bf16",
        "--training.loop.gradient_accumulation_steps", "1",
        "--training.dit_precision", "bf16",
    ]

    from fastvideo.distributed import get_world_group, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.utils.config import load_run_config

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = load_run_config(args.config, overrides=[*user_overrides, *fixed_overrides])
    tc = cfg.training
    maybe_init_distributed_environment_and_model_parallel(tc.distributed.tp_size, tc.distributed.sp_size)
    world = get_world_group()
    group = world.device_group
    rank = world.rank
    base.check_in_place_collectives(group)

    import fastvideo.models.loader.fsdp_load as fsdp_load
    import fastvideo.train.methods.fine_tuning.finetune as finetune
    from fastvideo.train.utils.builder import build_from_config

    original_shard_model = fsdp_load.shard_model
    original_optimizer_builder = finetune.build_optimizer_and_scheduler
    fsdp_load.shard_model = lambda *_, **__: None
    finetune.build_optimizer_and_scheduler = lambda **_: (None, None)
    try:
        _, method, dataloader, _ = build_from_config(cfg)
    finally:
        fsdp_load.shard_model = original_shard_model
        finetune.build_optimizer_and_scheduler = original_optimizer_builder

    method.student.vae = None
    gc.collect()
    torch.cuda.empty_cache()

    zero2 = ProfiledZero2(
        method.student.transformer,
        lr=tc.optimizer.learning_rate,
        betas=tc.optimizer.betas,
        eps=1e-8,
        weight_decay=tc.optimizer.weight_decay,
        group=group,
    )
    method._student_optimizer = None
    method._student_lr_scheduler = None
    method.on_train_start()
    zero2.check_bindings()
    dist.barrier(group=group)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    total_steps = args.warmup + args.steps
    step_events = [StepEvents() for _ in range(total_steps)]
    events_per_step = len(step_events[0].all_events())
    calibration_events = tuple(_event() for _ in range(events_per_step))
    for event in calibration_events:
        event.record()

    if rank == 0:
        print(
            f"PHASE_SETUP buckets={len(zero2.buckets)} arena_numel={zero2.param_arena.numel()} "
            f"events_per_step={events_per_step} allocated_gib={torch.cuda.memory_allocated() / 2**30:.2f}",
            flush=True,
        )

    data_iter = iter(dataloader)
    host_records: list[dict[str, float | int]] = []
    last_loss = math.nan
    last_norm = math.nan
    finite = True
    for step, events in enumerate(step_events):
        wall_start = time.perf_counter()
        batch = next(data_iter, None)
        if batch is None:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        data_ms = (time.perf_counter() - wall_start) * 1000.0

        zero2.begin_profile_step(events)
        events.gpu_start.record()
        zero2.zero_grad()
        events.after_zero.record()
        loss_map, outputs, _ = method.single_train_step(batch, step + 1)
        events.after_forward.record()
        method.backward(loss_map, outputs, grad_accum_rounds=1)
        events.after_backward.record()
        grad_norm = zero2.step(args.max_grad_norm)
        events.gpu_end.record()
        zero2.end_profile_step()

        last_loss = float(loss_map["total_loss"].detach())
        last_norm = float(grad_norm)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        finite = finite and math.isfinite(last_loss) and math.isfinite(last_norm)
        phase = "warmup" if step < args.warmup else "measure"
        if rank == 0:
            print(
                f"PHASE_STEP phase={phase} step={step + 1}/{total_steps} wall_ms={wall_ms:.3f} "
                f"data_ms={data_ms:.3f} ag_waits={events.ag_wait_count} "
                f"loss={last_loss:.6g} grad_norm={last_norm:.6g}",
                flush=True,
            )
        if step >= args.warmup:
            host_records.append({
                "global_step": step + 1,
                "wall_ms": wall_ms,
                "data_ms": data_ms,
            })

    zero2.wait_all_parameters()
    zero2.check_bindings()
    replica_error = zero2.check_replicas_and_masters()
    finite_tensor = torch.tensor(int(finite), device=zero2.param_arena.device)
    dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN, group=group)
    torch.cuda.synchronize()
    if not bool(finite_tensor.item()):
        raise RuntimeError("a measured rank produced non-finite loss or gradient norm")

    measured_events = step_events[args.warmup:]
    records = [_phase_record(events, host) for events, host in zip(measured_events, host_records, strict=True)]
    bad_ag_steps = [record["global_step"] for record in records if record["all_gather_wait_count"] != AG_WAITS_PER_STEP]
    max_gpu_closure = max(record["gpu_closure_error_ms"] for record in records)
    if bad_ag_steps:
        raise RuntimeError(f"expected {AG_WAITS_PER_STEP} all-gather waits on steady steps, bad steps: {bad_ag_steps}")
    if max_gpu_closure > GPU_CLOSURE_TOLERANCE_MS:
        raise RuntimeError(
            f"GPU categories miss span by {max_gpu_closure:.6f} ms, limit {GPU_CLOSURE_TOLERANCE_MS:.3f} ms")

    calibration_ms = _elapsed(calibration_events[0], calibration_events[-1])
    calibration_ms *= events_per_step / (events_per_step - 1)
    payload = {
        "rank": rank,
        "records": records,
        "event_overhead_ms": calibration_ms,
        "loss": last_loss,
        "grad_norm": last_norm,
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    payloads: list[dict[str, Any] | None] = [None] * world.world_size
    dist.all_gather_object(payloads, payload, group=world.cpu_group)

    if rank == 0:
        gathered = [payload for payload in payloads if payload is not None]
        slowest_vectors: list[dict[str, Any]] = []
        for measured_index in range(args.steps):
            candidates = []
            for rank_payload in gathered:
                record = dict(rank_payload["records"][measured_index])
                record["rank"] = rank_payload["rank"]
                candidates.append(record)
            slowest_vectors.append(max(candidates, key=lambda record: record["wall_ms"]))

        ordered = sorted(slowest_vectors, key=lambda record: record["wall_ms"])
        median_vector = ordered[(len(ordered) - 1) // 2]
        wall_ms = median_vector["wall_ms"]
        max_event_overhead_ms = max(payload["event_overhead_ms"] for payload in gathered)
        event_overhead_percent = 100.0 * max_event_overhead_ms / wall_ms
        slowest_rank_counts = {
            str(candidate_rank): sum(vector["rank"] == candidate_rank for vector in slowest_vectors)
            for candidate_rank in range(world.world_size)
        }
        result = {
            "kind": "ltx2_fixed_arena_zero2_inclusive_phase_profile",
            "sha": head,
            "world_size": world.world_size,
            "warmup_steps": args.warmup,
            "measured_steps": args.steps,
            "median_policy": "median_low of per-step slowest-rank vectors",
            "median_slowest_rank_vector": median_vector,
            "median_max_rank_wall_ms": wall_ms,
            "samples_per_second": 4000.0 / wall_ms,
            "model_mfu_percent": 14.444115 / (wall_ms / 1000.0),
            "gate_ms": args.gate_ms,
            "gate_pass": wall_ms <= args.gate_ms,
            "all_gather_waits_per_steady_step": AG_WAITS_PER_STEP,
            "all_gather_wait_count_pass": all(
                vector["all_gather_wait_count"] == AG_WAITS_PER_STEP for vector in slowest_vectors),
            "max_gpu_closure_error_ms": max(
                vector["gpu_closure_error_ms"] for vector in slowest_vectors),
            "gpu_closure_tolerance_ms": GPU_CLOSURE_TOLERANCE_MS,
            "gpu_closure_pass": all(
                vector["gpu_closure_error_ms"] <= GPU_CLOSURE_TOLERANCE_MS for vector in slowest_vectors),
            "max_total_closure_error_ms": max(
                vector["total_closure_error_ms"] for vector in slowest_vectors),
            "events_per_step": events_per_step,
            "max_event_record_overhead_ms": max_event_overhead_ms,
            "max_event_record_overhead_percent": event_overhead_percent,
            "event_overhead_limit_percent": args.max_event_overhead_percent,
            "event_overhead_pass": event_overhead_percent <= args.max_event_overhead_percent,
            "slowest_rank_step_counts": slowest_rank_counts,
            "last_loss_by_rank": [payload["loss"] for payload in gathered],
            "last_grad_norm_by_rank": [payload["grad_norm"] for payload in gathered],
            "allocated_gib_by_rank": [payload["allocated_gib"] for payload in gathered],
            "peak_allocated_gib_by_rank": [payload["peak_allocated_gib"] for payload in gathered],
            "arena_numel": zero2.param_arena.numel(),
            "bucket_count": len(zero2.buckets),
            "replica_sample_max_error": replica_error,
        }
        print("PHASE_RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
