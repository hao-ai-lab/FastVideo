# SPDX-License-Identifier: Apache-2.0
"""Measure MiniMax-H3 production VAE stage memory with CPU offload enabled."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--revision-label", required=True)
    parser.add_argument("--operation", choices=("encode", "decode"), required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def _git(source_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_snapshot(model_root: Path) -> str | None:
    parts = model_root.resolve().parts
    try:
        return parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError):
        return None


def _make_layout(rows, latent_shape):
    import torch

    from fastvideo.pipelines.basic.minimax_h3.packing import MiniMaxH3PackedLayout

    empty = torch.empty(0, dtype=torch.long, device=rows.device)
    return MiniMaxH3PackedLayout(
        sequence_length=rows.shape[0],
        position_ids=empty,
        token_tags=empty,
        video_indices=empty,
        audio_indices=empty,
        text_indices=empty,
        num_condition_video_rows=0,
        num_condition_audio_rows=0,
        num_video_latent_frames=latent_shape[2],
        latent_height=latent_shape[3],
        latent_width=latent_shape[4],
        num_audio_latents=0,
    )


def _build_operation(args, vae, device):
    import numpy as np
    import torch

    from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3Config
    from fastvideo.pipelines.basic.minimax_h3.packing import patchify_video_latents, video_latent_num_frames
    from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_decoding import MiniMaxH3VideoDecodingStage
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import (
        MINIMAX_H3_LAYOUT_KEY,
        MiniMaxH3LatentPreparationStage,
    )
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    patch_size = MiniMaxH3Config().arch_config.patch_size
    transformer = SimpleNamespace(patch_size=patch_size)
    runtime_args = SimpleNamespace(output_type="pil", pin_cpu_memory=False, vae_cpu_offload=True)

    if args.operation == "encode":
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("The encode benchmark is single-rank; use one process.")
        frames = np.random.default_rng(args.seed).integers(
            0,
            256,
            size=(args.num_frames, args.height, args.width, 3),
            dtype=np.uint8,
        )
        stage = MiniMaxH3LatentPreparationStage(
            transformer=transformer,
            vae=vae,
            audio_vae=None,
            scheduler=None,
            ref2va=True,
        )

        def run_once():
            reference = MiniMaxH3PreparedReference(media_type="video", frames=frames)
            vae.to(device)
            try:
                return stage._encode_visual_rows([reference], device)[0]
            finally:
                vae.to("cpu")

        return run_once, {
            "input": "NumPy PCG64 CPU uint8 RGB pixels",
            "input_shape": [1, 3, args.num_frames, args.height, args.width],
            "boundary": "before VAE CPU-to-GPU transfer through post-encode VAE CPU offload",
        }

    latent_shape = (
        1,
        vae.latent_channels,
        video_latent_num_frames(args.num_frames),
        args.height // vae.spatial_compression_ratio,
        args.width // vae.spatial_compression_ratio,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=torch.float32)
    rows = patchify_video_latents(latents, patch_size)
    layout = _make_layout(rows, latent_shape)
    stage = MiniMaxH3VideoDecodingStage(vae, transformer)

    def run_once():
        batch = ForwardBatch(data_type="video", latents=rows, raw_latent_shape=latent_shape)
        batch.extra[MINIMAX_H3_LAYOUT_KEY] = layout
        return stage.forward(batch, runtime_args).output

    return run_once, {
        "input": "PyTorch Philox normalized FP32 latents",
        "input_shape": list(latent_shape),
        "boundary": "MiniMaxH3VideoDecodingStage.forward including VAE CPU-to-GPU transfer and CPU offload",
    }


def _measure(run_once, device, warmups: int, repetitions: int) -> dict:
    import torch

    for _ in range(warmups):
        with torch.inference_mode():
            result = run_once()
        torch.cuda.synchronize(device)
        del result
        gc.collect()
        torch.cuda.empty_cache()

    records = []
    for repetition in range(repetitions):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        start_allocated = torch.cuda.memory_allocated(device)
        start_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.inference_mode():
            result = run_once()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        records.append({
            "repetition": repetition,
            "elapsed_seconds": elapsed,
            "start_allocated_bytes": start_allocated,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) - start_allocated,
            "start_reserved_bytes": start_reserved,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "incremental_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) - start_reserved,
            "output_shape": list(result.shape),
        })
        del result
    return {
        "repetitions": records,
        "median_elapsed_seconds": statistics.median(record["elapsed_seconds"] for record in records),
        "max_peak_allocated_bytes": max(record["peak_allocated_bytes"] for record in records),
        "max_incremental_peak_allocated_bytes": max(
            record["incremental_peak_allocated_bytes"] for record in records),
        "max_peak_reserved_bytes": max(record["peak_reserved_bytes"] for record in records),
        "max_incremental_peak_reserved_bytes": max(
            record["incremental_peak_reserved_bytes"] for record in records),
    }


def main() -> None:
    args = _parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions must be positive.")
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))

    import torch
    import torch.distributed as dist

    import fastvideo
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.models.loader.component_loader import VAELoader

    imported_root = Path(fastvideo.__file__).resolve().parents[1]
    if imported_root != source_root:
        raise RuntimeError(f"Imported FastVideo from {imported_root}, expected {source_root}.")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29673")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    world_size = int(os.environ["WORLD_SIZE"])
    maybe_init_distributed_environment_and_model_parallel(1, world_size)
    rank = dist.get_rank()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    component_dir = args.model_root.resolve() / "vae"
    config_path = component_dir / "config.json"
    index_path = component_dir / "diffusion_pytorch_model.safetensors.index.json"
    for path in (config_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    loader_args = SimpleNamespace(
        pipeline_config=MiniMaxH3PipelineConfig(),
        model_paths={},
        vae_cpu_offload=True,
    )
    vae = VAELoader().load(str(component_dir), loader_args)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in vae.parameters())
    run_once, workload = _build_operation(args, vae, device)
    rank_result = _measure(run_once, device, args.warmups, args.repetitions)
    rank_result.update({
        "rank": rank,
        "device_index": device.index,
        "device_name": torch.cuda.get_device_name(device),
        "device_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
    })
    rank_results = [None] * world_size
    dist.all_gather_object(rank_results, rank_result)

    if rank == 0:
        result = {
            "schema_version": 1,
            "exit_status": 0,
            "revision_label": args.revision_label,
            "source_root": str(source_root),
            "source_git_head": _git(source_root, "rev-parse", "HEAD"),
            "source_tracked_status": _git(source_root, "status", "--short", "--untracked-files=no"),
            "benchmark_script_sha256": _sha256(Path(__file__).resolve()),
            "model_root": str(args.model_root.resolve()),
            "model_snapshot": _model_snapshot(args.model_root),
            "model_config_sha256": _sha256(config_path),
            "model_index_sha256": _sha256(index_path),
            "vae_parameter_bytes": parameter_bytes,
            "operation": args.operation,
            "vae_cpu_offload": True,
            "pin_cpu_memory": False,
            "warmups": args.warmups,
            "measurement_repetitions": args.repetitions,
            "seed": args.seed,
            "workload": workload,
            "world_size": world_size,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF")
            or os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "rank_results": rank_results,
            "sum_rank_local_max_peak_allocated_bytes": sum(
                item["max_peak_allocated_bytes"] for item in rank_results),
            "sum_rank_local_max_incremental_peak_allocated_bytes": sum(
                item["max_incremental_peak_allocated_bytes"] for item in rank_results),
            "sum_rank_local_max_peak_reserved_bytes": sum(
                item["max_peak_reserved_bytes"] for item in rank_results),
            "sum_rank_local_max_incremental_peak_reserved_bytes": sum(
                item["max_incremental_peak_reserved_bytes"] for item in rank_results),
        }
        print("MINIMAX_H3_VAE_MEMORY=" + json.dumps(result, sort_keys=True), flush=True)

    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
