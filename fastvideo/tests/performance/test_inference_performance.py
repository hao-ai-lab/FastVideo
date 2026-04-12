# SPDX-License-Identifier: Apache-2.0
"""Config-driven inference performance tests.

Benchmark configs live in .buildkite/performance-benchmarks/tests/*.json.
Each JSON file defines model params, generation kwargs, run config, and
per-device thresholds.  This test module auto-discovers all configs and
parametrizes a single test function over them.
"""
import glob
import json
import os
import time
from datetime import datetime, timezone

import torch
import pytest

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger
from fastvideo.worker.multiproc_executor import MultiprocExecutor

logger = init_logger(__name__)

# -- Config discovery -------------------------------------------------------

_BENCHMARKS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    ".buildkite",
    "performance-benchmarks",
    "tests",
)


def _discover_benchmarks():
    """Glob benchmark JSON configs and return list of (id, config) tuples."""
    pattern = os.path.join(_BENCHMARKS_DIR, "*.json")
    configs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            cfg = json.load(f)
        configs.append(cfg)
    return configs


_BENCHMARK_CONFIGS = _discover_benchmarks()

# -- Helpers ----------------------------------------------------------------


def _get_thresholds(cfg):
    """Return thresholds dict for the current GPU from config."""
    device_name = torch.cuda.get_device_name()
    thresholds = cfg.get("thresholds", {})
    for gpu_key, thresh in thresholds.items():
        if gpu_key in device_name:
            logger.info("Using thresholds for %s: %s", gpu_key, thresh)
            return thresh
    default = thresholds.get("default", {
        "max_generation_time_s": 120.0,
        "max_peak_memory_mb": 30000.0,
    })
    logger.warning("No thresholds for device '%s', using defaults", device_name)
    return default


def _shutdown_executor(generator):
    if generator is None:
        return
    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()


def _run_generation(generator, prompt, generation_kwargs):
    """Run a single generation, return (elapsed_s, peak_memory_mb)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = generator.generate_video(prompt, **generation_kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_memory_mb = result.get("peak_memory_mb", 0.0) or 0.0
    return elapsed, peak_memory_mb


def _write_results(results):
    """Write JSON results to the results directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    bid = results.get("benchmark_id", "unknown")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"perf_{bid}_{ts}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Performance results written to %s", filepath)


# -- Test -------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg",
    _BENCHMARK_CONFIGS,
    ids=[c["benchmark_id"] for c in _BENCHMARK_CONFIGS],
)
def test_inference_performance(cfg):
    """Measure generation latency and peak GPU memory,
    assert against device-aware thresholds."""
    run_config = cfg.get("run_config", {})
    required_gpus = run_config.get("required_gpus", 1)
    available = torch.cuda.device_count()
    if available < required_gpus:
        pytest.skip(f"Need {required_gpus} GPUs, only {available} available")

    model_info = cfg["model"]
    init_kwargs = dict(cfg.get("init_kwargs", {}))
    gen_kwargs = dict(cfg.get("generation_kwargs", {}))
    prompts = cfg.get("test_prompts", ["A cinematic video."])
    prompt = prompts[0]

    num_warmup = run_config.get("num_warmup_runs", 1)
    num_measure = run_config.get("num_measurement_runs", 3)
    thresholds = _get_thresholds(cfg)

    # Remap JSON keys to VideoGenerator kwargs
    text_enc_prec = init_kwargs.pop("text_encoder_precisions", None)
    if text_enc_prec is not None:
        init_kwargs["text_encoder_precisions"] = tuple(text_enc_prec)

    # Output directory for generated videos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "generated_videos",
                              cfg["benchmark_id"])
    os.makedirs(output_dir, exist_ok=True)
    gen_kwargs["output_path"] = output_dir

    generator = None
    try:
        generator = VideoGenerator.from_pretrained(
            model_path=model_info["model_path"],
            **init_kwargs,
        )

        for i in range(num_warmup):
            logger.info("Warmup run %d/%d", i + 1, num_warmup)
            _run_generation(generator, prompt, gen_kwargs)

        times = []
        peak_memories = []
        for i in range(num_measure):
            logger.info("Measurement run %d/%d", i + 1, num_measure)
            elapsed, peak_mb = _run_generation(generator, prompt, gen_kwargs)
            logger.info("  Time: %.2fs, Peak memory: %.0fMB", elapsed, peak_mb)
            times.append(elapsed)
            peak_memories.append(peak_mb)
    finally:
        _shutdown_executor(generator)

    avg_time = sum(times) / len(times)
    max_peak_memory = max(peak_memories)
    device_name = torch.cuda.get_device_name()

    results = {
        "benchmark_id": cfg["benchmark_id"],
        "model_short_name": model_info.get("model_short_name", ""),
        "device": device_name,
        "num_gpus": init_kwargs.get("num_gpus", 1),
        "num_warmup_runs": num_warmup,
        "num_measurement_runs": num_measure,
        "avg_generation_time_s": round(avg_time, 3),
        "individual_times_s": [round(t, 3) for t in times],
        "max_peak_memory_mb": round(max_peak_memory, 1),
        "individual_peak_memories_mb": [round(m, 1) for m in peak_memories],
        "thresholds": thresholds,
        "commit": os.environ.get("BUILDKITE_COMMIT", ""),
        "pr_number": os.environ.get("BUILDKITE_PULL_REQUEST", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "Performance results: avg_time=%.2fs, "
        "max_peak_memory=%.0fMB", avg_time, max_peak_memory)
    _write_results(results)

    max_time = thresholds["max_generation_time_s"]
    max_mem = thresholds["max_peak_memory_mb"]

    assert avg_time <= max_time, (
        f"Average generation time {avg_time:.2f}s exceeds "
        f"threshold {max_time:.1f}s for {device_name}")

    assert max_peak_memory <= max_mem, (
        f"Peak memory {max_peak_memory:.0f}MB exceeds "
        f"threshold {max_mem:.0f}MB for {device_name}")
