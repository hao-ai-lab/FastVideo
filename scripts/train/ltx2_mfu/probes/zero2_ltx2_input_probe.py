#!/usr/bin/env python3
"""Disposable four-GB200 fixed-arena ZeRO-2 timing probe for LTX-2."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("FASTVIDEO_FA4", "1")
os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import _functional as optim_f


EXPECTED_SHA = "7f139e2b28610063d2f30526ba8f0ccae5d88944"


@dataclass
class _Bucket:
    params: tuple[nn.Parameter, ...]
    flat: torch.Tensor
    grad: torch.Tensor
    master: torch.Tensor
    master_grad: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    step: torch.Tensor
    reduce_work: Any = None
    gather_work: Any = None
    ready: int = 0


class LTX2FixedArenaZero2:
    """Accumulation=1, four-rank ZeRO-2 with stable BF16 arenas."""

    def __init__(
        self,
        transformer: nn.Module,
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        group: Any,
    ) -> None:
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if self.world_size != 4:
            raise ValueError(f"probe requires four ranks, got {self.world_size}")

        blocks = list(transformer.model.transformer_blocks)
        block_params = [tuple(p for p in block.parameters() if p.requires_grad) for block in blocks]
        in_blocks = {id(p) for params in block_params for p in params}
        root_params = tuple(p for p in transformer.parameters() if p.requires_grad and id(p) not in in_blocks)
        groups = [root_params, *block_params]
        if len(blocks) != 48 or not root_params or any(not params for params in block_params):
            raise ValueError("expected one root bucket and 48 nonempty LTX-2 block buckets")

        all_params = [p for params in groups for p in params]
        expected = [p for p in transformer.parameters() if p.requires_grad]
        if len({id(p) for p in all_params}) != len(all_params):
            raise ValueError("a trainable parameter appears in multiple buckets")
        if {id(p) for p in all_params} != {id(p) for p in expected}:
            raise ValueError("buckets do not cover every trainable parameter exactly once")
        if any(p.dtype != torch.float32 or not p.is_cuda or not p.is_contiguous() for p in all_params):
            raise ValueError("probe requires contiguous CUDA FP32 source parameters")

        padded_sizes = [((sum(p.numel() for p in params) + 3) // 4) * 4 for params in groups]
        device = all_params[0].device
        self.param_arena = torch.zeros(sum(padded_sizes), dtype=torch.bfloat16, device=device)
        self.grad_arena = torch.zeros_like(self.param_arena)
        self.buckets: list[_Bucket] = []
        self._bindings: list[tuple[nn.Parameter, int, int]] = []
        self.master_precision_delta_max = 0.0

        arena_offset = 0
        for params, padded_size in zip(groups, padded_sizes, strict=True):
            flat = self.param_arena.narrow(0, arena_offset, padded_size)
            grad = self.grad_arena.narrow(0, arena_offset, padded_size)
            shard_size = padded_size // self.world_size
            shard_begin = self.rank * shard_size
            shard_end = shard_begin + shard_size
            master = torch.zeros(shard_size, dtype=torch.float32, device=device)
            param_offset = 0
            with torch.no_grad():
                for param in params:
                    size = param.numel()
                    param_flat = flat.narrow(0, param_offset, size)
                    param_grad = grad.narrow(0, param_offset, size)
                    source = param.detach().reshape(-1)
                    param_flat.copy_(source)

                    overlap_begin = max(param_offset, shard_begin)
                    overlap_end = min(param_offset + size, shard_end)
                    if overlap_begin < overlap_end:
                        source_begin = overlap_begin - param_offset
                        master_begin = overlap_begin - shard_begin
                        count = overlap_end - overlap_begin
                        master_slice = master.narrow(0, master_begin, count)
                        master_slice.copy_(source.narrow(0, source_begin, count))
                        precision_delta = (master_slice - master_slice.bfloat16().float()).abs().max()
                        self.master_precision_delta_max = max(self.master_precision_delta_max,
                                                              float(precision_delta))

                    param.data = param_flat.view_as(param)
                    param.grad = param_grad.view_as(param)
                    self._bindings.append((param, param.data_ptr(), param.grad.data_ptr()))
                    param_offset += size

            self.buckets.append(
                _Bucket(
                    params=params,
                    flat=flat,
                    grad=grad,
                    master=master,
                    master_grad=torch.empty_like(master),
                    exp_avg=torch.zeros_like(master),
                    exp_avg_sq=torch.zeros_like(master),
                    step=torch.zeros((), dtype=torch.float32, device=device),
                ))
            arena_offset += padded_size

        self.lr = float(lr)
        self.betas = betas
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._reduce_order = [*range(len(self.buckets) - 1, 0, -1), 0]
        self._reduce_cursor = 0
        self._seen: set[int] = set()
        self._active = False
        self._handles = []

        for bucket_id, bucket in enumerate(self.buckets):
            for param in bucket.params:
                self._handles.append(
                    param.register_post_accumulate_grad_hook(
                        lambda p, bucket_id=bucket_id: self._mark_ready(bucket_id, p)))

        self._handles.append(transformer.register_forward_pre_hook(lambda *_: self._wait_gather(0)))
        for bucket_id, block in enumerate(blocks, start=1):
            self._handles.append(
                block.register_forward_pre_hook(lambda *_, bucket_id=bucket_id: self._wait_gather(bucket_id)))

    def zero_grad(self) -> None:
        if self._active:
            raise RuntimeError("zero_grad called before the preceding step completed")
        self.grad_arena.zero_()
        self._seen.clear()
        self._reduce_cursor = 0
        for bucket in self.buckets:
            bucket.ready = len(bucket.params)
            bucket.reduce_work = None
        self._active = True

    @torch.no_grad()
    def step(self, max_grad_norm: float) -> torch.Tensor:
        if not self._active:
            raise RuntimeError("step called without zero_grad/backward")
        self._launch_ready_buckets()
        if self._reduce_cursor != len(self._reduce_order):
            missing = sum(bucket.ready for bucket in self.buckets)
            raise RuntimeError(f"backward left {missing} parameter gradients unused")
        for bucket in self.buckets:
            bucket.reduce_work.wait()

        master_grads = [bucket.master_grad for bucket in self.buckets]
        reduced_shards = [bucket.grad.view(4, -1)[self.rank] for bucket in self.buckets]
        torch._foreach_copy_(master_grads, reduced_shards)

        norm_sq = torch.stack(torch._foreach_norm(master_grads, 2.0)).square().sum()
        dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM, group=self.group)
        total_norm = norm_sq.sqrt()
        torch._foreach_mul_(master_grads, (max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0))

        optim_f.adamw(
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

        local_param_shards = [bucket.flat.view(4, -1)[self.rank] for bucket in self.buckets]
        torch._foreach_copy_(local_param_shards, [bucket.master for bucket in self.buckets])
        for bucket, local_shard in zip(self.buckets, local_param_shards, strict=True):
            bucket.gather_work = dist.all_gather_into_tensor(
                bucket.flat,
                local_shard,
                group=self.group,
                async_op=True,
            )
        self._active = False
        return total_norm

    def wait_all_parameters(self) -> None:
        for bucket_id in range(len(self.buckets)):
            self._wait_gather(bucket_id)

    def check_bindings(self) -> None:
        for param, param_ptr, grad_ptr in self._bindings:
            if param.data_ptr() != param_ptr or param.grad is None or param.grad.data_ptr() != grad_ptr:
                raise RuntimeError("parameter or gradient arena pointer changed")

    @torch.no_grad()
    def check_replicas_and_masters(self) -> float:
        self.wait_all_parameters()
        count = min(4096, self.param_arena.numel())
        idx = _sample_indices(self.param_arena.numel(), count, self.param_arena.device)
        lo = self.param_arena[idx].float()
        hi = lo.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=self.group)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=self.group)
        replica_error = float((hi - lo).abs().max())
        for bucket in self.buckets:
            local = bucket.flat.view(4, -1)[self.rank]
            sample_idx = _sample_indices(local.numel(), min(32, local.numel()), local.device)
            if not torch.equal(local[sample_idx], bucket.master[sample_idx].to(torch.bfloat16)):
                raise RuntimeError("gathered BF16 shard does not match its FP32 master")
        if replica_error != 0.0:
            raise RuntimeError(f"replicated parameters diverged by {replica_error}")
        return replica_error

    def _mark_ready(self, bucket_id: int, param: nn.Parameter) -> None:
        if not self._active:
            raise RuntimeError("gradient arrived outside an active step")
        key = id(param)
        if key in self._seen:
            raise RuntimeError("probe does not support multiple backward calls per step")
        self._seen.add(key)
        self.buckets[bucket_id].ready -= 1
        self._launch_ready_buckets()

    def _launch_ready_buckets(self) -> None:
        while self._reduce_cursor < len(self._reduce_order):
            bucket = self.buckets[self._reduce_order[self._reduce_cursor]]
            if bucket.ready:
                return
            recv = bucket.grad.view(4, -1)[self.rank]
            bucket.reduce_work = dist.reduce_scatter_tensor(
                recv,
                bucket.grad,
                op=dist.ReduceOp.AVG,
                group=self.group,
                async_op=True,
            )
            self._reduce_cursor += 1

    def _wait_gather(self, bucket_id: int) -> None:
        work = self.buckets[bucket_id].gather_work
        if work is not None:
            work.wait()
            self.buckets[bucket_id].gather_work = None


def _sample_indices(numel: int, count: int, device: torch.device) -> torch.Tensor:
    """Evenly spaced int64 indices without float precision loss on 13B arenas."""
    if count <= 1:
        return torch.zeros((count, ), dtype=torch.int64, device=device)
    return torch.arange(count, dtype=torch.int64, device=device).mul_(numel - 1).div_(count - 1,
                                                                                      rounding_mode="floor")


def check_in_place_collectives(group: Any) -> None:
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if world_size != 4:
        raise ValueError(f"probe requires four ranks, got {world_size}")
    device = torch.device("cuda", torch.cuda.current_device())
    buf = torch.full((world_size * 32,), rank + 1, dtype=torch.bfloat16, device=device)
    recv = buf.view(world_size, -1)[rank]
    dist.reduce_scatter_tensor(recv, buf, op=dist.ReduceOp.AVG, group=group)
    torch.testing.assert_close(recv.float(), torch.full_like(recv.float(), 2.5))
    buf.zero_()
    send = buf.view(world_size, -1)[rank]
    send.fill_(rank)
    dist.all_gather_into_tensor(buf, send, group=group)
    expected = torch.arange(world_size, device=device).repeat_interleave(send.numel()).to(torch.bfloat16)
    torch.testing.assert_close(buf, expected)


def _events() -> tuple[torch.cuda.Event, ...]:
    return tuple(torch.cuda.Event(enable_timing=True) for _ in range(5))


def _median_max_rank(payloads: list[dict[str, Any]], key: str) -> float:
    per_step = zip(*(payload[key] for payload in payloads), strict=True)
    return statistics.median(max(values) for values in per_step)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/train/configs/overfit_ltx2_t2v.yaml")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gate-ms", type=float, default=419.0)
    parser.add_argument("--expected-sha", default=EXPECTED_SHA)
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
        "--training.dit_precision", "fp32",
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
    check_in_place_collectives(group)

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

    # Preprocessed fine-tuning never uses the VAE, and validation is intentionally absent.
    method.student.vae = None
    gc.collect()
    torch.cuda.empty_cache()

    zero2 = LTX2FixedArenaZero2(
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
    if rank == 0:
        print(
            f"ZERO2_SETUP buckets={len(zero2.buckets)} arena_numel={zero2.param_arena.numel()} "
            f"allocated_gib={torch.cuda.memory_allocated() / 2**30:.2f}",
            flush=True,
        )

    data_iter = iter(dataloader)
    records: list[dict[str, Any]] = []
    finite = True
    total_steps = args.warmup + args.steps
    for step in range(total_steps):
        wall_start = time.perf_counter()
        batch = next(data_iter, None)
        if batch is None:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        data_ms = (time.perf_counter() - wall_start) * 1000.0

        start, after_zero, after_forward, after_backward, after_step = _events()
        start.record()
        zero2.zero_grad()
        after_zero.record()
        loss_map, outputs, _ = method.single_train_step(batch, step + 1)
        after_forward.record()
        method.backward(loss_map, outputs, grad_accum_rounds=1)
        after_backward.record()
        grad_norm = zero2.step(args.max_grad_norm)
        after_step.record()
        loss = float(loss_map["total_loss"].detach())
        norm = float(grad_norm)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        finite = finite and torch.isfinite(torch.tensor((loss, norm))).all().item()
        if rank == 0:
            phase = "warmup" if step < args.warmup else "measure"
            print(
                f"ZERO2_STEP phase={phase} step={step + 1}/{total_steps} "
                f"wall_ms={wall_ms:.3f} loss={loss:.6g} grad_norm={norm:.6g}",
                flush=True,
            )
        if step >= args.warmup:
            records.append({
                "wall_ms": wall_ms,
                "data_ms": data_ms,
                "zero_ms": start.elapsed_time(after_zero),
                "forward_ms": after_zero.elapsed_time(after_forward),
                "backward_ms": after_forward.elapsed_time(after_backward),
                "update_ms": after_backward.elapsed_time(after_step),
                "gpu_ms": start.elapsed_time(after_step),
                "loss": loss,
                "grad_norm": norm,
            })

    zero2.wait_all_parameters()
    torch.cuda.synchronize()
    zero2.check_bindings()
    replica_error = zero2.check_replicas_and_masters()
    finite_tensor = torch.tensor(int(bool(finite)), device=zero2.param_arena.device)
    dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN, group=group)
    if not bool(finite_tensor.item()):
        raise RuntimeError("a measured rank produced non-finite loss or gradient norm")

    payload = {
        key: [record[key] for record in records]
        for key in ("wall_ms", "data_ms", "zero_ms", "forward_ms", "backward_ms", "update_ms", "gpu_ms")
    }
    payload["loss"] = records[-1]["loss"]
    payload["grad_norm"] = records[-1]["grad_norm"]
    payload["allocated_gib"] = torch.cuda.memory_allocated() / 2**30
    payload["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    payloads: list[dict[str, Any] | None] = [None] * world.world_size
    dist.all_gather_object(payloads, payload, group=world.cpu_group)

    if rank == 0:
        gathered = [payload for payload in payloads if payload is not None]
        wall_ms = _median_max_rank(gathered, "wall_ms")
        result = {
            "kind": "ltx2_fixed_arena_zero2",
            "sha": head,
            "world_size": world.world_size,
            "warmup_steps": args.warmup,
            "measured_steps": args.steps,
            "median_max_rank_wall_ms": wall_ms,
            "median_max_rank_gpu_ms": _median_max_rank(gathered, "gpu_ms"),
            "median_phase_ms": {
                key.removesuffix("_ms"): _median_max_rank(gathered, key)
                for key in ("data_ms", "zero_ms", "forward_ms", "backward_ms", "update_ms")
            },
            "samples_per_second": 4000.0 / wall_ms,
            "model_mfu_percent": 14.444115 / (wall_ms / 1000.0),
            "gate_ms": args.gate_ms,
            "gate_pass": wall_ms <= args.gate_ms,
            "last_loss_by_rank": [payload["loss"] for payload in gathered],
            "last_grad_norm_by_rank": [payload["grad_norm"] for payload in gathered],
            "allocated_gib_by_rank": [payload["allocated_gib"] for payload in gathered],
            "peak_allocated_gib_by_rank": [payload["peak_allocated_gib"] for payload in gathered],
            "arena_numel": zero2.param_arena.numel(),
            "bucket_count": len(zero2.buckets),
            "master_precision_delta_max": zero2.master_precision_delta_max,
            "replica_sample_max_error": replica_error,
        }
        print("ZERO2_RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
