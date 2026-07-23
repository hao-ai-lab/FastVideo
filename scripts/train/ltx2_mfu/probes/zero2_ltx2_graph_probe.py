#!/usr/bin/env python3
"""Scratch two-step CUDA-graph gate for the fixed-arena LTX-2 probe."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import time
import traceback
from typing import Any

os.environ.setdefault("FASTVIDEO_FA4", "1")
os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")

import torch
import torch.distributed as dist

from zero2_ltx2_probe import (
    EXPECTED_SHA,
    LTX2FixedArenaZero2,
    _median_max_rank,
    _sample_indices,
    check_in_place_collectives,
)


BASELINE_MS = 403.724612435326


def _stage_cuda_tensors(value):
    if torch.is_tensor(value):
        return value.cuda() if value.device.type == "cpu" else value
    if isinstance(value, dict):
        return {key: _stage_cuda_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stage_cuda_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_stage_cuda_tensors(item) for item in value)
    return value


def _install_graphable_sigma_sampler(student):
    """Avoid two host-created CUDA scalars in the otherwise graph-safe sampler."""
    import math
    import types

    from fastvideo.train.models.ltx2 import ltx2 as model

    def sample(self, batch_size, token_count, device, generator):
        slope = ((model._SIGMA_MAX_SHIFT - model._SIGMA_MIN_SHIFT) /
                 (model._SIGMA_MAX_TOKENS - model._SIGMA_MIN_TOKENS))
        mu = slope * float(token_count) + (model._SIGMA_MIN_SHIFT - slope * model._SIGMA_MIN_TOKENS)
        normal = torch.randn(
            (batch_size,), generator=generator, device=device, dtype=torch.float32
        ) * model._SIGMA_STD + mu
        sigmas = torch.sigmoid(normal)
        lo = 1.0 / (1.0 + math.exp(-(mu + model._SIGMA_Z_LO * model._SIGMA_STD)))
        hi = 1.0 / (1.0 + math.exp(-(mu + model._SIGMA_Z_HI * model._SIGMA_STD)))
        raw = (sigmas - lo) / (hi - lo)
        stretched = torch.where(raw >= model._SIGMA_EPS, raw, 2 * model._SIGMA_EPS - raw).clamp(0.0, 1.0)
        if self._timestep_uniform_prob > 0.0:
            prob = torch.rand((batch_size,), generator=generator, device=device)
            uniform = torch.rand((batch_size,), generator=generator, device=device) * (
                1.0 - model._SIGMA_EPS) + model._SIGMA_EPS
            stretched = torch.where(prob > self._timestep_uniform_prob, stretched, uniform)
        return stretched

    student._sample_ltx2_sigmas = types.MethodType(sample, student)


def _install_graphable_video_patchifier(transformer, output_shape, device):
    """Reuse a device-resident patch-size constant during graph capture."""
    import types

    from fastvideo.models.dits import ltx2 as model

    patchifier = transformer.patchifier
    original = patchifier.get_patch_grid_bounds
    expected = original(output_shape, device=device)
    patch_size_delta = torch.tensor(
        patchifier.patch_size,
        device=device,
        dtype=expected.dtype,
    ).view(3, 1, 1, 1)

    def get_patch_grid_bounds(self, output_shape, device=None):
        device = device or patch_size_delta.device
        grid_coords = torch.meshgrid(
            torch.arange(0, output_shape.frames, self._patch_size[0], device=device),
            torch.arange(0, output_shape.height, self._patch_size[1], device=device),
            torch.arange(0, output_shape.width, self._patch_size[2], device=device),
            indexing="ij",
        )
        patch_starts = torch.stack(grid_coords, dim=0)
        latent_coords = torch.stack((patch_starts, patch_starts + patch_size_delta), dim=-1)
        return model.repeat(
            latent_coords,
            "c f h w bounds -> b c (f h w) bounds",
            b=output_shape.batch,
            bounds=2,
        )

    patchifier.get_patch_grid_bounds = types.MethodType(get_patch_grid_bounds, patchifier)
    actual = patchifier.get_patch_grid_bounds(output_shape, device=device)
    max_error = int((actual - expected).abs().max())
    if not torch.equal(actual, expected):
        raise RuntimeError(f"graphable patchifier changed eager output (max error {max_error})")
    return {"equal": True, "max_error": max_error, "shape": list(actual.shape)}


def _run_step(zero2, method, batch, iteration, max_grad_norm, *, final_join):
    zero2.zero_grad()
    loss_map, outputs, _ = method.single_train_step(batch, iteration)
    method.backward(loss_map, outputs, grad_accum_rounds=1)
    grad_norm = zero2.step(max_grad_norm)
    if final_join:
        zero2.wait_all_parameters()
    return loss_map["total_loss"], grad_norm


def _noise_sample(student):
    noise = student._graph_probe_noise
    sigma = student._graph_probe_sigmas
    return tuple(noise.flatten()[:4].float().cpu().tolist()), tuple(sigma.flatten()[:4].float().cpu().tolist())


def _time_eager_pair(zero2, method, batch, iteration, max_grad_norm, steps_per_replay, group):
    dist.barrier(group=group)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    begin.record()
    loss = norm = None
    for substep in range(steps_per_replay):
        loss, norm = _run_step(
            zero2,
            method,
            batch,
            iteration * steps_per_replay + substep,
            max_grad_norm,
            final_join=substep == steps_per_replay - 1,
        )
    end.record()
    end.synchronize()
    return {
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0 / steps_per_replay,
        "gpu_ms": begin.elapsed_time(end) / steps_per_replay,
        "loss": float(loss.detach()),
        "grad_norm": float(norm.detach()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/train/configs/overfit_ltx2_t2v.yaml")
    parser.add_argument("--eager-warmup", type=int, default=5)
    parser.add_argument("--eager-steps", type=int, default=5)
    parser.add_argument("--replay-warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--steps-per-replay", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--baseline-ms", type=float, default=BASELINE_MS)
    parser.add_argument("--gate-saving-ms", type=float, default=32.0)
    parser.add_argument("--expected-sha", default=EXPECTED_SHA)
    args, user_overrides = parser.parse_known_args()
    if min(args.eager_warmup, args.eager_steps, args.replay_warmup, args.replays) < 1:
        raise ValueError("all warmup/measurement counts must be positive")

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
    _install_graphable_sigma_sampler(method.student)
    if method.cuda_generator is None:
        raise RuntimeError("training method did not initialize its CUDA generator")

    original_prepare = method.student._prepare_dit_inputs

    def tracked_prepare(training_batch, generator):
        result = original_prepare(training_batch, generator)
        method.student._graph_probe_noise = result.noise
        method.student._graph_probe_sigmas = result.sigmas
        return result

    method.student._prepare_dit_inputs = tracked_prepare
    static_batch = _stage_cuda_tensors(next(iter(dataloader)))
    raw_latents = static_batch["vae_latent"][:, :, :tc.data.num_latent_t]
    from fastvideo.models.dits.ltx2 import VideoLatentShape
    patchifier_check = _install_graphable_video_patchifier(
        method.student.transformer,
        VideoLatentShape.from_torch_shape(raw_latents.shape),
        raw_latents.device,
    )
    if rank == 0:
        print("PATCHIFIER_EQUIVALENCE " + json.dumps(patchifier_check, sort_keys=True), flush=True)
    zero2.check_bindings()
    dist.barrier(group=group)
    torch.cuda.synchronize()

    # Compile/warm every path and establish a same-process, same-join eager control.
    for iteration in range(args.eager_warmup):
        _run_step(
            zero2,
            method,
            static_batch,
            iteration,
            args.max_grad_norm,
            final_join=True,
        )
    zero2.wait_all_parameters()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    eager_records = [
        _time_eager_pair(
            zero2,
            method,
            static_batch,
            iteration,
            args.max_grad_norm,
            args.steps_per_replay,
            group,
        ) for iteration in range(args.eager_steps)
    ]
    eager_peak_gib = torch.cuda.max_memory_allocated() / 2**30
    zero2.wait_all_parameters()
    torch.cuda.synchronize()

    sample_idx = _sample_indices(zero2.param_arena.numel(), 4096, zero2.param_arena.device)
    params_before = zero2.param_arena[sample_idx].clone()
    step_before = float(zero2.buckets[0].step)
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier(group=group)
    torch.cuda.synchronize()
    pre_capture_allocated_gib = torch.cuda.memory_allocated() / 2**30
    pre_capture_reserved_gib = torch.cuda.memory_reserved() / 2**30
    torch.cuda.reset_peak_memory_stats()

    graph = torch.cuda.CUDAGraph()
    if not hasattr(graph, "register_generator_state"):
        raise RuntimeError("this PyTorch lacks CUDAGraph.register_generator_state")
    graph.register_generator_state(method.cuda_generator)
    capture_stream = torch.cuda.Stream()
    capture_started = time.perf_counter()
    try:
        with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
            static_losses = []
            static_norms = []
            static_noises = []
            static_sigmas = []
            for substep in range(args.steps_per_replay):
                loss, norm = _run_step(
                    zero2,
                    method,
                    static_batch,
                    1000 + substep,
                    args.max_grad_norm,
                    final_join=substep == args.steps_per_replay - 1,
                )
                static_losses.append(loss)
                static_norms.append(norm)
                static_noises.append(method.student._graph_probe_noise)
                static_sigmas.append(method.student._graph_probe_sigmas)
    except Exception as exc:
        traceback.print_exc()
        blocker = {
            "kind": "ltx2_fixed_arena_graph_blocker",
            "rank": rank,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "steps_per_replay": args.steps_per_replay,
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        print("GRAPH_BLOCKER " + json.dumps(blocker, sort_keys=True), flush=True)
        return

    capture_seconds = time.perf_counter() - capture_started
    torch.cuda.synchronize()
    dist.barrier(group=group)
    captured_allocated_gib = torch.cuda.memory_allocated() / 2**30
    captured_reserved_gib = torch.cuda.memory_reserved() / 2**30

    warm_replay_rng = []
    for _ in range(args.replay_warmup):
        graph.replay()
        torch.cuda.synchronize()
        warm_replay_rng.append((
            tuple(static_noises[-1].flatten()[:4].float().cpu().tolist()),
            tuple(static_sigmas[-1].flatten()[:4].float().cpu().tolist()),
            float(static_losses[-1]),
        ))

    records = []
    for _ in range(args.replays):
        dist.barrier(group=group)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        records.append({
            "wall_ms": (time.perf_counter() - wall_start) * 1000.0 / args.steps_per_replay,
            "gpu_ms": begin.elapsed_time(end) / args.steps_per_replay,
            "loss": float(static_losses[-1]),
            "grad_norm": float(static_norms[-1]),
            "noise": tuple(static_noises[-1].flatten()[:4].float().cpu().tolist()),
            "sigma": tuple(static_sigmas[-1].flatten()[:4].float().cpu().tolist()),
        })

    torch.cuda.synchronize()
    zero2.check_bindings()
    replica_error = zero2.check_replicas_and_masters()
    params_after = zero2.param_arena[sample_idx]
    parameter_sample_max_change = float((params_after - params_before).abs().max())
    step_after = float(zero2.buckets[0].step)
    graph_peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30
    graph_peak_reserved_gib = torch.cuda.max_memory_reserved() / 2**30

    finite = all(
        torch.isfinite(torch.tensor((record["loss"], record["grad_norm"]))).all().item()
        for record in records)
    noise_varies = len({record["noise"] for record in records}) > 1
    sigma_varies = len({record["sigma"] for record in records}) > 1
    loss_varies = len({record["loss"] for record in records}) > 1
    expected_step_delta = args.steps_per_replay * (1 + args.replay_warmup + args.replays)
    local_checks = {
        "finite": bool(finite),
        "noise_varies": noise_varies,
        "sigma_varies": sigma_varies,
        "loss_varies": loss_varies,
        "parameter_sample_max_change": parameter_sample_max_change,
        "optimizer_step_delta": step_after - step_before,
        "expected_optimizer_step_delta": expected_step_delta,
    }
    checks: list[dict[str, Any] | None] = [None] * world.world_size
    dist.all_gather_object(checks, local_checks, group=world.cpu_group)

    eager_payload = {
        "wall_ms": [record["wall_ms"] for record in eager_records],
        "gpu_ms": [record["gpu_ms"] for record in eager_records],
    }
    graph_payload = {
        "wall_ms": [record["wall_ms"] for record in records],
        "gpu_ms": [record["gpu_ms"] for record in records],
    }
    payload = {
        "eager": eager_payload,
        "graph": graph_payload,
        "last_loss": records[-1]["loss"],
        "last_grad_norm": records[-1]["grad_norm"],
        "checks": local_checks,
        "eager_peak_gib": eager_peak_gib,
        "pre_capture_allocated_gib": pre_capture_allocated_gib,
        "pre_capture_reserved_gib": pre_capture_reserved_gib,
        "captured_allocated_gib": captured_allocated_gib,
        "captured_reserved_gib": captured_reserved_gib,
        "graph_peak_allocated_gib": graph_peak_allocated_gib,
        "graph_peak_reserved_gib": graph_peak_reserved_gib,
    }
    payloads: list[dict[str, Any] | None] = [None] * world.world_size
    dist.all_gather_object(payloads, payload, group=world.cpu_group)

    if rank == 0:
        gathered = [item for item in payloads if item is not None]
        eager = [item["eager"] for item in gathered]
        graphed = [item["graph"] for item in gathered]
        eager_wall_ms = _median_max_rank(eager, "wall_ms")
        graph_wall_ms = _median_max_rank(graphed, "wall_ms")
        result = {
            "kind": "ltx2_fixed_arena_cuda_graph",
            "sha": head,
            "world_size": world.world_size,
            "static_batch": True,
            "steps_per_replay": args.steps_per_replay,
            "eager_warmup": args.eager_warmup,
            "eager_measured_pairs": args.eager_steps,
            "warmup_replays": args.replay_warmup,
            "measured_replays": args.replays,
            "measured_training_steps": args.replays * args.steps_per_replay,
            "capture_seconds": capture_seconds,
            "same_join_eager_wall_ms": eager_wall_ms,
            "same_join_eager_gpu_ms": _median_max_rank(eager, "gpu_ms"),
            "graph_wall_ms_per_step": graph_wall_ms,
            "graph_gpu_ms_per_step": _median_max_rank(graphed, "gpu_ms"),
            "saving_vs_same_join_eager_ms": eager_wall_ms - graph_wall_ms,
            "baseline_ms": args.baseline_ms,
            "saving_vs_baseline_ms": args.baseline_ms - graph_wall_ms,
            "gate_saving_ms": args.gate_saving_ms,
            "gate_pass_vs_baseline": args.baseline_ms - graph_wall_ms >= args.gate_saving_ms,
            "model_mfu_percent": 14.444115 / (graph_wall_ms / 1000.0),
            "last_loss_by_rank": [item["last_loss"] for item in gathered],
            "last_grad_norm_by_rank": [item["last_grad_norm"] for item in gathered],
            "checks_by_rank": checks,
            "warm_replay_rng_samples_rank0": warm_replay_rng,
            "replica_sample_max_error": replica_error,
            "memory_gib_by_rank": [{
                key: item[key]
                for key in (
                    "eager_peak_gib",
                    "pre_capture_allocated_gib",
                    "pre_capture_reserved_gib",
                    "captured_allocated_gib",
                    "captured_reserved_gib",
                    "graph_peak_allocated_gib",
                    "graph_peak_reserved_gib",
                )
            } for item in gathered],
        }
        print("GRAPH_RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
