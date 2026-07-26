#!/usr/bin/env python3
"""Four-GB200 LTX-2 packed BF16 DP/TP timing gate."""

import gc
import json
import os
import statistics

import torch
import torch.distributed as dist


LAYERS = (
    ("self_qkv", "column", "video", 4290, 4096, 12288),
    ("self_out", "row", "video", 4290, 4096, 4096),
    ("cross_q", "column", "video", 4290, 4096, 4096),
    ("cross_kv", "column", "text", 1024, 4096, 8192),
    ("cross_out", "row", "video", 4290, 4096, 4096),
    ("ffn_up", "column", "video", 4290, 4096, 16384),
    ("ffn_down", "row", "video", 4290, 16384, 4096),
)
BLOCK_COUNT = 48
CURRENT_STEP_MS = 403.725
TARGET_STEP_MS = 288.882
GB200_CAPACITY_GIB = 189471 / 1024
FIXED_ARENA_PEAK_GIB = 149.792064
FIXED_ARENA_STEADY_GIB = 97.234253


def emit(kind, **fields):
    if dist.get_rank() == 0:
        print(json.dumps({"kind": kind, **fields}, sort_keys=True), flush=True)


def timed(fn, sync_group, warmup=3, samples=7, inner=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=sync_group, device_ids=[torch.cuda.current_device()])
    values = []
    for _ in range(samples):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) / inner)
    return statistics.median(values)


def sanity(tp, group, tp_rank):
    if tp == 1:
        return 0.0
    torch.manual_seed(7)
    x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(12, 16, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(8, 12, device="cuda", dtype=torch.bfloat16)
    k_width, n_width = 16 // tp, 12 // tp
    row_fwd = x[:, tp_rank * k_width:(tp_rank + 1) * k_width] @ w[:, tp_rank * k_width:(tp_rank + 1) * k_width].T
    dist.all_reduce(row_fwd, group=group)
    col_dx = dy[:, tp_rank * n_width:(tp_rank + 1) * n_width] @ w[tp_rank * n_width:(tp_rank + 1) * n_width]
    dist.all_reduce(col_dx, group=group)
    return max(float((row_fwd - x @ w.T).abs().max()), float((col_dx - dy @ w).abs().max()))


def case(topology, tp, batch, group, tp_rank, layer):
    name, parallel, modality, base_m, k, n = layer
    m = base_m * batch
    local_k = k // tp if parallel == "row" else k
    local_n = n // tp if parallel == "column" else n
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(20260721 + LAYERS.index(layer))
    x = torch.empty(m, local_k, device="cuda", dtype=torch.bfloat16).normal_()
    w = torch.empty(local_n, local_k, device="cuda", dtype=torch.bfloat16).normal_(0, 0.02)
    dy = torch.empty(m, local_n, device="cuda", dtype=torch.bfloat16).normal_(0, 0.01)
    y = torch.empty(m, local_n, device="cuda", dtype=torch.bfloat16)
    dx = torch.empty(m, local_k, device="cuda", dtype=torch.bfloat16)
    dw = torch.empty(local_n, local_k, device="cuda", dtype=torch.bfloat16)
    collective_tensor = dx if parallel == "column" else y
    zeros = torch.zeros_like(collective_tensor) if tp > 1 else None

    def fwd():
        return torch.mm(x, w.T, out=y)

    def dgrad():
        return torch.mm(dy, w, out=dx)

    def wgrad():
        return torch.mm(dy.T, x, out=dw)

    def allreduce():
        assert zeros is not None
        dist.all_reduce(zeros, group=group)

    def fwd_collective():
        torch.mm(x, w.T, out=y)
        if tp > 1 and parallel == "row":
            dist.all_reduce(y, group=group)
        return y

    def dgrad_collective():
        torch.mm(dy, w, out=dx)
        if tp > 1 and parallel == "column":
            dist.all_reduce(dx, group=group)
        return dx

    local = {
        "fwd_ms": timed(fwd, group),
        "dgrad_ms": timed(dgrad, group),
        "wgrad_ms": timed(wgrad, group),
        "allreduce_ms": timed(allreduce, group) if tp > 1 else 0.0,
        "fwd_collective_ms": timed(fwd_collective, group) if tp > 1 else 0.0,
        "dgrad_collective_ms": timed(dgrad_collective, group) if tp > 1 else 0.0,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    row = None
    if dist.get_rank() == 0:
        row = {
            "topology": topology,
            "name": name,
            "count": BLOCK_COUNT,
            "parallel": parallel,
            "modality": modality,
            "logical_shape": [m, k, n],
            "local_gemm_shape": [m, local_k, local_n],
            "allreduce_shape": list(collective_tensor.shape) if tp > 1 else None,
            "allreduce_bytes": collective_tensor.numel() * 2 if tp > 1 else 0,
            **{key: max(rank_row[key] for rank_row in gathered) for key in local},
        }
        if tp == 1:
            row["fwd_collective_ms"] = row["fwd_ms"]
            row["dgrad_collective_ms"] = row["dgrad_ms"]
        print(json.dumps({"kind": "case", **row}, sort_keys=True), flush=True)
    del x, w, dy, y, dx, dw, zeros
    return row


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 4
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False

    tp2_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    topologies = (
        ("dp4_b1", 1, 1, dist.group.WORLD, 0),
        ("tp2_dp2_b2", 2, 2, tp2_groups[rank // 2], rank % 2),
        ("tp4_b4", 4, 4, dist.group.WORLD, rank),
    )
    emit(
        "environment",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(local_rank),
        capability=torch.cuda.get_device_capability(local_rank),
        world_size=world,
        warmup=3,
        samples=7,
        inner=3,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
    )

    summaries = {}
    for topology, tp, batch, group, tp_rank in topologies:
        errors = [None] * world
        dist.all_gather_object(errors, sanity(tp, group, tp_rank))
        emit("sanity", topology=topology, max_abs=max(errors))
        rows = [case(topology, tp, batch, group, tp_rank, layer) for layer in LAYERS]
        if rank != 0:
            continue
        compute_ms = sum((r["fwd_ms"] + r["dgrad_ms"] + r["wgrad_ms"]) * r["count"] for r in rows)
        allreduce_ms = sum(r["allreduce_ms"] * r["count"] for r in rows)
        end_to_end_ms = sum(
            ((r["fwd_ms"] if r["parallel"] == "column" else r["fwd_collective_ms"])
             + (r["dgrad_collective_ms"] if r["parallel"] == "column" else r["dgrad_ms"])
             + r["wgrad_ms"]) * r["count"] for r in rows)
        payload_bytes = sum(r["allreduce_bytes"] * r["count"] for r in rows)
        ring_wire_bytes = payload_bytes * (2 * (tp - 1) / tp) if tp > 1 else 0
        flops = sum(3 * 2 * (r["logical_shape"][0] * r["logical_shape"][1] * r["logical_shape"][2] // tp) * r["count"] for r in rows)
        summaries[topology] = {
            "tp": tp,
            "dp": world // tp,
            "local_batch": batch,
            "compute_ms": compute_ms,
            "allreduce_only_ms": allreduce_ms,
            "ideal_overlap_lower_bound_ms": max(compute_ms, allreduce_ms),
            "sequential_compute_collective_ms": end_to_end_ms,
            "logical_flops_per_rank": flops,
            "effective_compute_tflops": flops / compute_ms / 1e9,
            "allreduce_payload_gib": payload_bytes / 2**30,
            "ring_wire_lower_bound_gib_per_rank": ring_wire_bytes / 2**30,
            "max_microbench_allocated_gib": max(r["peak_allocated_gib"] for r in rows),
        }
        emit("topology", topology=topology, **summaries[topology])

    if rank == 0:
        baseline = summaries["dp4_b1"]["compute_ms"]
        required_segment = baseline - (CURRENT_STEP_MS - TARGET_STEP_MS)
        for topology in ("tp2_dp2_b2", "tp4_b4"):
            row = summaries[topology]
            row["required_projection_segment_ms"] = required_segment
            row["projected_step_compute_only_ms"] = CURRENT_STEP_MS - baseline + row["compute_ms"]
            row["projected_step_ideal_overlap_ms"] = CURRENT_STEP_MS - baseline + row["ideal_overlap_lower_bound_ms"]
            row["projected_step_sequential_ms"] = CURRENT_STEP_MS - baseline + row["sequential_compute_collective_ms"]
            row["can_clear_target_compute_only"] = row["projected_step_compute_only_ms"] <= TARGET_STEP_MS
            row["can_clear_target_ideal_overlap"] = row["projected_step_ideal_overlap_ms"] <= TARGET_STEP_MS
        state_floor = {
            "dp4_b1": 24.292 * 2 + 36.438,
            "tp2_dp2_b2": (24.292 / 2) * 2 + 36.438,
            "tp4_b4": (24.292 / 4) * 2 + 36.438,
        }
        emit(
            "gate",
            current_step_ms=CURRENT_STEP_MS,
            target_step_ms=TARGET_STEP_MS,
            saving_required_ms=CURRENT_STEP_MS - TARGET_STEP_MS,
            dp4_projection_baseline_ms=baseline,
            allreduces_per_block={"video": 6, "text": 1},
            gb200_capacity_gib=GB200_CAPACITY_GIB,
            fixed_arena_peak_gib=FIXED_ARENA_PEAK_GIB,
            fixed_arena_steady_gib=FIXED_ARENA_STEADY_GIB,
            state_floor_gib_per_rank=state_floor,
            optimistic_peak_if_non_state_unchanged_gib={
                name: FIXED_ARENA_PEAK_GIB - state_floor["dp4_b1"] + floor
                for name, floor in state_floor.items()
            },
            conservative_peak_if_all_non_state_scales_with_batch_gib={
                name: floor + (FIXED_ARENA_PEAK_GIB - state_floor["dp4_b1"]) * summaries[name]["local_batch"]
                for name, floor in state_floor.items()
            },
            summaries=summaries,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
