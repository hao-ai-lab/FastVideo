# SPDX-License-Identifier: Apache-2.0
"""Run one config-driven inference benchmark under an nsys capture range."""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STAGE_METRIC_MAP: dict[str, str] = {
    "TextEncodingStage": "text_encoder_time_s",
    "DenoisingStage": "dit_time_s",
    "DmdDenoisingStage": "dit_time_s",
    "DecodingStage": "vae_decode_time_s",
}

_CUDART: ctypes.CDLL | None = None


def _load_cudart() -> ctypes.CDLL:
    global _CUDART
    if _CUDART is not None:
        return _CUDART

    errors: list[str] = []
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            lib = ctypes.CDLL(name)
        except OSError as exc:
            errors.append(f"{name}: {exc}")
            continue
        lib.cudaProfilerStart.restype = ctypes.c_int
        lib.cudaProfilerStop.restype = ctypes.c_int
        _CUDART = lib
        return lib

    raise RuntimeError("Unable to load CUDA runtime for cudaProfilerApi capture range: " + "; ".join(errors))


def _cuda_profiler_start() -> None:
    err = _load_cudart().cudaProfilerStart()
    if err != 0:
        raise RuntimeError(f"cudaProfilerStart failed with CUDA error code {err}")


def _cuda_profiler_stop() -> None:
    err = _load_cudart().cudaProfilerStop()
    if err != 0:
        raise RuntimeError(f"cudaProfilerStop failed with CUDA error code {err}")


def _worker_cuda_profiler_start(worker_wrapper: Any) -> dict[str, Any]:
    _cuda_profiler_start()
    return {
        "pid": os.getpid(),
        "rank": getattr(worker_wrapper, "rpc_rank", None),
        "event": "cudaProfilerStart",
    }


def _worker_cuda_profiler_stop(worker_wrapper: Any) -> dict[str, Any]:
    import torch

    torch.cuda.synchronize()
    _cuda_profiler_stop()
    return {
        "pid": os.getpid(),
        "rank": getattr(worker_wrapper, "rpc_rank", None),
        "event": "cudaProfilerStop",
    }


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _remap_init_kwargs(init_kwargs: dict[str, Any]) -> dict[str, Any]:
    remapped = dict(init_kwargs)
    text_enc_prec = remapped.pop("text_encoder_precisions", None)
    if text_enc_prec is not None:
        remapped["text_encoder_precisions"] = tuple(text_enc_prec)
    return remapped


def _extract_component_times(result: dict[str, Any]) -> dict[str, float | None]:
    component_times: dict[str, float | None] = {
        "text_encoder_time_s": None,
        "dit_time_s": None,
        "vae_decode_time_s": None,
    }
    logging_info = result.get("logging_info")
    if logging_info is None:
        return component_times
    if isinstance(logging_info, Mapping):
        stages: dict[str, Any] = logging_info.get("stages", {}) or {}
    else:
        stages = getattr(logging_info, "stages", {}) or {}
    for stage_name, stage_data in stages.items():
        metric_key = STAGE_METRIC_MAP.get(stage_name)
        if metric_key is None:
            continue
        elapsed = stage_data.get("execution_time")
        if elapsed is None:
            continue
        existing = component_times[metric_key]
        component_times[metric_key] = elapsed if existing is None else existing + elapsed
    return component_times


def _run_generation(
    generator: Any,
    prompt: str,
    generation_kwargs: dict[str, Any],
) -> tuple[float, float, dict[str, float | None]]:
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    result = generator.generate_video(prompt, **generation_kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_memory_mb = result.get("peak_memory_mb", 0.0) or 0.0
    return elapsed, peak_memory_mb, _extract_component_times(result)


def _shutdown_executor(generator: Any | None) -> None:
    from fastvideo.worker.multiproc_executor import MultiprocExecutor

    if generator is None:
        return
    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()


def _start_capture(generator: Any) -> list[dict[str, Any]]:
    import cloudpickle
    from fastvideo.worker.multiproc_executor import MultiprocExecutor

    _cuda_profiler_start()
    responses: list[dict[str, Any]] = []
    if isinstance(generator.executor, MultiprocExecutor):
        responses = generator.executor.collective_rpc(
            cloudpickle.dumps(_worker_cuda_profiler_start))
    return responses


def _stop_capture(generator: Any) -> list[dict[str, Any]]:
    import cloudpickle
    import torch
    from fastvideo.worker.multiproc_executor import MultiprocExecutor

    responses: list[dict[str, Any]] = []
    if isinstance(generator.executor, MultiprocExecutor):
        responses = generator.executor.collective_rpc(
            cloudpickle.dumps(_worker_cuda_profiler_stop))
    torch.cuda.synchronize()
    _cuda_profiler_stop()
    return responses


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def run_profile(args: argparse.Namespace) -> None:
    global logger

    import torch

    from fastvideo import VideoGenerator
    from fastvideo.logger import init_logger

    logger = init_logger(__name__)

    cfg = _load_config(args.config)
    run_config = cfg.get("run_config", {})
    required_gpus = int(run_config.get("required_gpus", 1))
    available_gpus = torch.cuda.device_count()
    if available_gpus < required_gpus:
        raise RuntimeError(f"Need {required_gpus} CUDA GPUs for {cfg['benchmark_id']}, found {available_gpus}")

    model_info = cfg["model"]
    init_kwargs = _remap_init_kwargs(cfg.get("init_kwargs", {}))
    generation_kwargs = dict(cfg.get("generation_kwargs", {}))
    generation_kwargs["output_path"] = str(args.output_dir)
    prompt = cfg.get("test_prompts", ["A cinematic video."])[0]
    num_warmup = int(run_config.get("num_warmup_runs", 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

    generator: VideoGenerator | None = None
    capture_started = False
    worker_start_events: list[dict[str, Any]] = []
    worker_stop_events: list[dict[str, Any]] = []
    try:
        logger.info("Loading model %s with init kwargs: %s", model_info["model_path"], init_kwargs)
        generator = VideoGenerator.from_pretrained(
            model_path=model_info["model_path"],
            **init_kwargs,
        )

        for i in range(num_warmup):
            logger.info("Warmup run %d/%d", i + 1, num_warmup)
            _run_generation(generator, prompt, generation_kwargs)

        logger.info("Starting cudaProfilerApi capture range")
        worker_start_events = _start_capture(generator)
        capture_started = True
        elapsed, peak_mb, component_times = _run_generation(generator, prompt, generation_kwargs)
    finally:
        if capture_started and generator is not None:
            logger.info("Stopping cudaProfilerApi capture range")
            worker_stop_events = _stop_capture(generator)
        _shutdown_executor(generator)

    device_names = [torch.cuda.get_device_name(i) for i in range(required_gpus)]
    throughput_fps = None
    num_frames = generation_kwargs.get("num_frames")
    if isinstance(num_frames, (int, float)) and elapsed > 0:
        throughput_fps = num_frames / elapsed

    summary = {
        "benchmark_id": cfg["benchmark_id"],
        "model_short_name": model_info.get("model_short_name", ""),
        "model_path": model_info["model_path"],
        "config_path": str(args.config),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_names": device_names,
        "available_gpus": available_gpus,
        "required_gpus": required_gpus,
        "num_warmup_runs": num_warmup,
        "num_profiled_runs": 1,
        "profiled_generation_time_s": round(elapsed, 3),
        "throughput_fps": round(throughput_fps, 3) if throughput_fps is not None else None,
        "peak_memory_mb": round(peak_mb, 1),
        "component_times_s": component_times,
        "generation_kwargs": generation_kwargs,
        "worker_start_events": worker_start_events,
        "worker_stop_events": worker_stop_events,
    }
    _write_summary(args.summary_path, summary)
    logger.info("Profile summary written to %s", args.summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".buildkite/performance-benchmarks/tests/wan-t2v-1.3b.json"),
        help="Benchmark JSON config to profile.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated videos.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        required=True,
        help="Path for the profile summary JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_profile(parse_args())
