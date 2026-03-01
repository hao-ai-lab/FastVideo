# SPDX-License-Identifier: Apache-2.0
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

REQUIRED_GPUS = 2

NUM_WARMUP_RUNS = 1
NUM_MEASUREMENT_RUNS = 3

WAN_T2V_PARAMS = {
    "num_gpus":
    2,
    "model_path":
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "height":
    480,
    "width":
    832,
    "num_frames":
    45,
    "num_inference_steps":
    4,
    "guidance_scale":
    3,
    "embedded_cfg_scale":
    6,
    "flow_shift":
    7.0,
    "seed":
    1024,
    "sp_size":
    2,
    "tp_size":
    1,
    "vae_sp":
    True,
    "fps":
    24,
    "neg_prompt":
    "Bright tones, overexposed, static, blurred details, subtitles, "
    "style, works, paintings, images, static, overall gray, worst quality, "
    "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards",
    "text-encoder-precision": ("fp32", ),
}

TEST_PROMPT = (
    "Will Smith casually eats noodles, his relaxed demeanor contrasting "
    "with the energetic background of a bustling street food market. "
    "The scene captures a mix of humor and authenticity. "
    "Mid-shot framing, vibrant lighting.")

# Device-aware thresholds: {gpu_name_substring: {metric: value}}
# Initial values are generous placeholders — calibrate after first CI run.
DEVICE_THRESHOLDS = {
    "L40S": {
        "max_generation_time_s": 60.0,
        "max_peak_memory_mb": 20000.0,
    },
}

# Fallback for unknown GPUs (very generous so test still runs)
DEFAULT_THRESHOLDS = {
    "max_generation_time_s": 120.0,
    "max_peak_memory_mb": 30000.0,
}


def _get_thresholds() -> dict:
    device_name = torch.cuda.get_device_name()
    for gpu_key, thresholds in DEVICE_THRESHOLDS.items():
        if gpu_key in device_name:
            logger.info("Using thresholds for %s: %s", gpu_key, thresholds)
            return thresholds
    logger.warning("No thresholds for device '%s', using defaults", device_name)
    return DEFAULT_THRESHOLDS


def _shutdown_executor(generator):
    if generator is None:
        return
    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()


def _run_generation(generator, prompt, generation_kwargs):
    """Run a single generation and return (elapsed_seconds, peak_memory_mb)."""
    for device_id in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_id)

    torch.cuda.synchronize()
    start = time.perf_counter()
    generator.generate_video(prompt, **generation_kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_memory_bytes = max(
        torch.cuda.max_memory_allocated(i)
        for i in range(torch.cuda.device_count()))
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    return elapsed, peak_memory_mb


def _write_results(results: dict) -> None:
    """Write JSON results to the results directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"perf_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Performance results written to %s", filepath)


@pytest.mark.parametrize("model_id", ["Wan2.1-T2V-1.3B-Diffusers"])
def test_inference_performance(model_id):
    """Measure generation latency and peak GPU memory, assert against
    device-aware thresholds."""
    params = WAN_T2V_PARAMS
    thresholds = _get_thresholds()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "generated_videos", model_id)
    os.makedirs(output_dir, exist_ok=True)

    init_kwargs = {
        "num_gpus": params["num_gpus"],
        "flow_shift": params["flow_shift"],
        "sp_size": params["sp_size"],
        "tp_size": params["tp_size"],
    }
    if params.get("vae_sp"):
        init_kwargs["vae_sp"] = True
        init_kwargs["vae_tiling"] = True
    if "text-encoder-precision" in params:
        init_kwargs["text_encoder_precisions"] = params[
            "text-encoder-precision"]

    generation_kwargs = {
        "num_inference_steps": params["num_inference_steps"],
        "output_path": output_dir,
        "height": params["height"],
        "width": params["width"],
        "num_frames": params["num_frames"],
        "guidance_scale": params["guidance_scale"],
        "embedded_cfg_scale": params["embedded_cfg_scale"],
        "seed": params["seed"],
        "fps": params["fps"],
    }
    if "neg_prompt" in params:
        generation_kwargs["neg_prompt"] = params["neg_prompt"]

    generator = None
    try:
        generator = VideoGenerator.from_pretrained(
            model_path=params["model_path"], **init_kwargs)

        # Warmup runs (discard results, primes CUDA kernels)
        for i in range(NUM_WARMUP_RUNS):
            logger.info("Warmup run %d/%d", i + 1, NUM_WARMUP_RUNS)
            _run_generation(generator, TEST_PROMPT, generation_kwargs)

        # Measurement runs
        times = []
        peak_memories = []
        for i in range(NUM_MEASUREMENT_RUNS):
            logger.info("Measurement run %d/%d", i + 1, NUM_MEASUREMENT_RUNS)
            elapsed, peak_mb = _run_generation(generator, TEST_PROMPT,
                                               generation_kwargs)
            logger.info("  Time: %.2fs, Peak memory: %.0fMB", elapsed, peak_mb)
            times.append(elapsed)
            peak_memories.append(peak_mb)

    finally:
        _shutdown_executor(generator)

    avg_time = sum(times) / len(times)
    max_peak_memory = max(peak_memories)
    device_name = torch.cuda.get_device_name()

    results = {
        "model_id": model_id,
        "device": device_name,
        "num_gpus": params["num_gpus"],
        "num_warmup_runs": NUM_WARMUP_RUNS,
        "num_measurement_runs": NUM_MEASUREMENT_RUNS,
        "avg_generation_time_s": round(avg_time, 3),
        "individual_times_s": [round(t, 3) for t in times],
        "max_peak_memory_mb": round(max_peak_memory, 1),
        "individual_peak_memories_mb": [round(m, 1) for m in peak_memories],
        "thresholds": thresholds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "Performance results: avg_time=%.2fs, "
        "max_peak_memory=%.0fMB", avg_time, max_peak_memory)
    _write_results(results)

    # Assert against thresholds
    max_time = thresholds["max_generation_time_s"]
    max_mem = thresholds["max_peak_memory_mb"]

    assert avg_time <= max_time, (
        f"Average generation time {avg_time:.2f}s exceeds threshold "
        f"{max_time:.1f}s for {device_name}")

    assert max_peak_memory <= max_mem, (
        f"Peak memory {max_peak_memory:.0f}MB exceeds threshold "
        f"{max_mem:.0f}MB for {device_name}")
