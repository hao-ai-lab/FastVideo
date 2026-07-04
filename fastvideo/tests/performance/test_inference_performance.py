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
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import torch
import pytest

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger
from fastvideo.tests.performance.identity import (
    build_recipe_from_benchmark_config,
    environment_fingerprint,
    environment_metadata,
    hardware_profile,
    hardware_profile_id,
    recipe_fingerprint,
    resolved_revision_from_model_path,
    software_profile,
    software_profile_id,
)
from fastvideo.worker.multiproc_executor import MultiprocExecutor

logger = init_logger(__name__)

STAGE_METRIC_MAP: dict[str, str] = {
    "TextEncodingStage": "text_encoder_time_s",
    "DenoisingStage": "dit_time_s",
    "DmdDenoisingStage": "dit_time_s",
    "DecodingStage": "vae_decode_time_s",
}
V2_CONFIG_SCHEMA_VERSION = 2
V2_REQUIRED_IDENTITY_FIELDS = (
    "workload_id",
    "variant_id",
    "benchmark_version",
)
V2_OPTIONAL_METADATA_FIELDS = (
    "recipe",
    "metric_threshold_policy",
    "quality_metadata",
)

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


def _has_v2_fields(cfg):
    v2_fields = V2_REQUIRED_IDENTITY_FIELDS + V2_OPTIONAL_METADATA_FIELDS
    return any(field in cfg for field in v2_fields)


def _is_v2_config(cfg):
    return cfg.get("config_schema_version") == V2_CONFIG_SCHEMA_VERSION


def _validate_non_empty_string(value, field, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: v2 identity field {field!r} must be a non-empty string")


def _validate_integer(value, field, path):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: v2 identity field {field!r} must be an integer")


def _validate_benchmark_config(cfg, path="<memory>"):
    missing_common = [field for field in ("benchmark_id",) if field not in cfg]
    if missing_common:
        raise ValueError(f"{path}: missing required benchmark config fields: {', '.join(missing_common)}")

    schema_version = cfg.get("config_schema_version")
    if schema_version is None:
        if _has_v2_fields(cfg):
            raise ValueError(f"{path}: v2 benchmark identity fields require config_schema_version=2")
        return

    if schema_version != V2_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported benchmark config_schema_version={schema_version!r}")

    missing_v2 = [field for field in V2_REQUIRED_IDENTITY_FIELDS if field not in cfg]
    if missing_v2:
        raise ValueError(f"{path}: missing required v2 identity fields: {', '.join(missing_v2)}")

    _validate_non_empty_string(cfg["workload_id"], "workload_id", path)
    _validate_non_empty_string(cfg["variant_id"], "variant_id", path)
    _validate_integer(cfg["benchmark_version"], "benchmark_version", path)

    for field in V2_OPTIONAL_METADATA_FIELDS:
        if field in cfg and not isinstance(cfg[field], Mapping):
            raise ValueError(f"{path}: optional v2 metadata field {field!r} must be an object")


def _config_identity_metadata(cfg):
    if not _is_v2_config(cfg):
        return {}
    metadata = {
        "config_schema_version": cfg["config_schema_version"],
        "workload_id": cfg["workload_id"],
        "variant_id": cfg["variant_id"],
        "benchmark_version": cfg["benchmark_version"],
    }
    for field in V2_OPTIONAL_METADATA_FIELDS:
        if field in cfg:
            metadata[field] = cfg[field]
    return metadata


def _benchmark_display_id(cfg):
    return cfg["benchmark_id"]


def _discover_benchmarks():
    """Glob benchmark JSON configs and return list of (id, config) tuples."""
    pattern = os.path.join(_BENCHMARKS_DIR, "*.json")
    configs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            cfg = json.load(f)
        _validate_benchmark_config(cfg, path)
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


def _extract_component_times(result: dict) -> dict[str, float | None]:
    component_times: dict[str, float | None] = {
        "text_encoder_time_s": None,
        "dit_time_s": None,
        "vae_decode_time_s": None,
    }
    logging_info = result.get("logging_info")
    if logging_info is None:
        return component_times
    if isinstance(logging_info, Mapping):
        stages: dict = logging_info.get("stages", {}) or {}
    else:
        stages: dict = getattr(logging_info, "stages", {}) or {}
    if not stages:
        return component_times
    logger.info("Discovered pipeline stages: %s", list(stages.keys()))
    for stage_name, stage_data in stages.items():
        if not isinstance(stage_data, Mapping):
            logger.debug("Skipping malformed stage '%s' data: %r", stage_name, stage_data)
            continue
        stage_class = stage_data.get("stage_class", stage_name)
        metric_key = STAGE_METRIC_MAP.get(stage_class)
        if metric_key is None:
            logger.debug("Unmapped stage '%s' class '%s' (%.3fs)",
                         stage_name,
                         stage_class,
                         stage_data.get("execution_time", 0))
            continue
        elapsed = stage_data.get("execution_time")
        if elapsed is None:
            continue
        existing = component_times[metric_key]
        component_times[metric_key] = (
            elapsed if existing is None else existing + elapsed)
    return component_times


def _avg_component(all_component_times: list[dict], key: str) -> float | None:
    vals = [r[key] for r in all_component_times if r.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _run_generation(generator, prompt, generation_kwargs):
    """Run a single generation, return (elapsed_s, peak_memory_mb, component_times)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = generator.generate_video(prompt, **generation_kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_memory_mb = result.get("peak_memory_mb", 0.0) or 0.0
    component_times = _extract_component_times(result)
    return elapsed, peak_memory_mb, component_times

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


def _backend_name(value) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _collect_worker_identity(worker) -> dict[str, Any]:
    pipeline = getattr(worker, "pipeline", None)
    model_path = getattr(pipeline, "model_path", None)
    backends: set[str] = set()

    modules = getattr(pipeline, "modules", None)
    if isinstance(modules, Mapping):
        module_iter = modules.values()
    elif isinstance(pipeline, torch.nn.Module):
        module_iter = (pipeline,)
    else:
        module_iter = ()

    for module in module_iter:
        if isinstance(module, torch.nn.Module):
            for submodule in module.modules():
                backend = getattr(submodule, "backend", None)
                if backend is not None:
                    backends.add(_backend_name(backend))
        else:
            backend = getattr(module, "backend", None)
            if backend is not None:
                backends.add(_backend_name(backend))

    return {
        "resolved_attention_backends": sorted(backends),
        "resolved_model_path": model_path,
    }


def _single_or_list(values: set[str]) -> str | list[str] | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    return ordered


def _runtime_identity_from_generator(generator) -> dict[str, Any]:
    worker_records = generator.executor.collective_rpc(_collect_worker_identity)
    resolved_backends: set[str] = set()
    resolved_revisions: set[str] = set()

    for record in worker_records:
        resolved_backends.update(record.get("resolved_attention_backends") or [])
        revision = resolved_revision_from_model_path(record.get("resolved_model_path"))
        if revision is not None:
            resolved_revisions.add(revision)

    return {
        "resolved_attention_backend": _single_or_list(resolved_backends),
        "resolved_model_revision": _single_or_list(resolved_revisions),
    }


def _benchmark_identity_fields(cfg):
    return {
        key: cfg[key]
        for key in ("workload_id", "variant_id", "benchmark_version")
        if cfg.get(key) is not None
    }


def _build_identity_fields(cfg, init_kwargs, prompt, runtime_identity):
    recipe = build_recipe_from_benchmark_config(
        cfg,
        resolved_attention_backend=runtime_identity.get("resolved_attention_backend"),
        resolved_model_revision=runtime_identity.get("resolved_model_revision"),
        measured_prompts=[prompt],
    )
    num_gpus = init_kwargs.get("num_gpus", cfg.get("run_config", {}).get("required_gpus", 1))
    hw_profile = hardware_profile(num_gpus=num_gpus)
    sw_profile = software_profile()
    env_metadata = environment_metadata(
        hardware=hw_profile,
        software=sw_profile,
    )
    return {
        "recipe": recipe,
        "recipe_fingerprint": recipe_fingerprint(recipe),
        "hardware_profile": hw_profile,
        "hardware_profile_id": hardware_profile_id(hw_profile),
        "software_profile": sw_profile,
        "software_profile_id": software_profile_id(sw_profile),
        "environment_metadata": env_metadata,
        "environment_fingerprint": environment_fingerprint(env_metadata),
        **_benchmark_identity_fields(cfg),
    }


# -- Test -------------------------------------------------------------------

def _run_benchmark(cfg):
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
        runtime_identity = _runtime_identity_from_generator(generator)

        for i in range(num_warmup):
            logger.info("Warmup run %d/%d", i + 1, num_warmup)
            _run_generation(generator, prompt, gen_kwargs)

        times = []
        peak_memories = []
        all_component_times = []
        for i in range(num_measure):
            logger.info("Measurement run %d/%d", i + 1, num_measure)
            elapsed, peak_mb, component_times = _run_generation(
                generator,
                prompt,
                gen_kwargs,
            )
            logger.info("  Time: %.2fs, Peak memory: %.0fMB", elapsed, peak_mb)
            times.append(elapsed)
            peak_memories.append(peak_mb)
            all_component_times.append(component_times)
    finally:
        _shutdown_executor(generator)

    avg_time = sum(times) / len(times)
    max_peak_memory = max(peak_memories)
    device_name = torch.cuda.get_device_name()
    num_frames = gen_kwargs.get("num_frames")
    throughput_fps = (1.0 / avg_time) if avg_time > 0 else None
    if isinstance(num_frames, (int, float)) and avg_time > 0:
        throughput_fps = num_frames / avg_time

    results = {
        "benchmark_id": cfg["benchmark_id"],
        **_config_identity_metadata(cfg),
        "model_short_name": model_info.get("model_short_name", ""),
        "device": device_name,
        "num_gpus": init_kwargs.get("num_gpus", 1),
        "num_warmup_runs": num_warmup,
        "num_measurement_runs": num_measure,
        "avg_generation_time_s": round(avg_time, 3),
        "individual_times_s": [round(t, 3) for t in times],
        "throughput_fps": round(throughput_fps, 3)
        if throughput_fps is not None else None,
        "max_peak_memory_mb": round(max_peak_memory, 1),
        "individual_peak_memories_mb": [round(m, 1) for m in peak_memories],
        "thresholds": thresholds,
        "regression_thresholds": cfg.get("regression_thresholds", {}),
        "commit": os.environ.get("BUILDKITE_COMMIT", ""),
        "pr_number": os.environ.get("BUILDKITE_PULL_REQUEST", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text_encoder_time_s": _avg_component(all_component_times,
                                              "text_encoder_time_s"),
        "dit_time_s": _avg_component(all_component_times, "dit_time_s"),
        "vae_decode_time_s": _avg_component(all_component_times,
                                            "vae_decode_time_s"),
        **_build_identity_fields(cfg, init_kwargs, prompt, runtime_identity),
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

    component_thresholds = {
        "text_encoder_time_s": thresholds.get("max_text_encoder_time_s"),
        "dit_time_s": thresholds.get("max_dit_time_s"),
        "vae_decode_time_s": thresholds.get("max_vae_decode_time_s"),
    }
    for metric, max_val in component_thresholds.items():
        if max_val is None:
            continue
        actual = results[metric]
        if actual is not None:
            assert actual <= max_val, (
                f"{metric} {actual:.3f}s exceeds threshold {max_val:.3f}s "
                f"for {device_name}")


@pytest.mark.parametrize(
    "cfg",
    _BENCHMARK_CONFIGS,
    ids=[_benchmark_display_id(c) for c in _BENCHMARK_CONFIGS],
)
def test_inference_performance(cfg):
    """Measure generation latency, peak GPU memory, and component-level timings
    (text encoder, DiT, VAE decode). Assert each against device-aware thresholds.
    """

    original_env = os.environ.get("FASTVIDEO_STAGE_LOGGING")
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
    try:
        _run_benchmark(cfg)
    finally:
        if original_env is None:
            os.environ.pop("FASTVIDEO_STAGE_LOGGING", None)
        else:
            os.environ["FASTVIDEO_STAGE_LOGGING"] = original_env
