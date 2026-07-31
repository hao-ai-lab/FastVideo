#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic FastVideo generation launcher driven by workload manifests.

This script is the FastVideo half of MotionKernel Workstream 1. It loads a
versioned workload YAML/JSON (schema_version 1), resolves the model through
FastVideo's registries, runs one mode per process, and writes a structured
result JSON that MotionKernel can validate and resume over.

Example::

    python examples/inference/optimizations/generation_launcher.py \\
        --workload /path/to/workloads/wan_t2v_1.3b_480p.yaml \\
        --mode native \\
        --output-dir /tmp/wan_ab

    python examples/inference/optimizations/generation_launcher.py \\
        --workload /path/to/workloads/wan_t2v_1.3b_480p.yaml \\
        --mode optimized \\
        --output-dir /tmp/wan_ab

Modes run in separate processes so environment variables (for example
``FASTVIDEO_WAN_FUSIONS``) do not leak across baseline and candidate runs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

RESULT_SCHEMA_VERSION = 1
WORKLOAD_SCHEMA_VERSION = 1


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        import yaml

        raw = yaml.safe_load(text)
    elif suffix == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(
            f"unsupported workload extension {suffix!r}; use .yaml/.yml/.json"
        )
    if not isinstance(raw, Mapping):
        raise ValueError("workload top level must be an object")
    return dict(raw)


def load_workload_dict(path: Path) -> dict[str, Any]:
    """Lightweight structural validation for schema_version 1 workloads."""
    raw = _load_mapping(path)
    version = raw.get("schema_version")
    if version != WORKLOAD_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported workload schema_version {version!r}; "
            f"expected {WORKLOAD_SCHEMA_VERSION}"
        )
    for key in ("workload_id", "model", "task", "sampling"):
        if key not in raw:
            raise ValueError(f"workload missing required field {key!r}")
    model = raw["model"]
    if not isinstance(model, Mapping) or not model.get("model_id"):
        raise ValueError("workload.model.model_id is required")
    if raw.get("prompt") is None and raw.get("prompt_file") is None:
        raise ValueError("workload requires prompt or prompt_file")
    if raw.get("prompt") is not None and raw.get("prompt_file") is not None:
        raise ValueError("provide only one of prompt or prompt_file")
    return raw


def resolve_prompt(workload: Mapping[str, Any], *, base_dir: Path) -> str:
    if workload.get("prompt") is not None:
        text = str(workload["prompt"]).strip()
        if not text:
            raise ValueError("workload.prompt must be non-empty")
        return text
    prompt_file = Path(str(workload["prompt_file"]))
    if not prompt_file.is_absolute():
        prompt_file = base_dir / prompt_file
    text = prompt_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt_file is empty: {prompt_file}")
    return text


def build_request(workload: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    sampling = dict(workload.get("sampling") or {})
    # FastVideo SamplingConfig does not consume these workload-only fields.
    sampling.pop("dtype", None)
    sampling.pop("attention_backend", None)
    measurement = dict(workload.get("measurement") or {})
    return {
        "prompt": resolve_prompt(workload, base_dir=base_dir),
        "sampling": sampling,
        "output": {
            "save_video": bool(measurement.get("save_video", False)),
            "return_frames": bool(measurement.get("save_frames", True)),
        },
    }


def build_generator_kwargs(
    workload: Mapping[str, Any],
    *,
    model_override: str | None = None,
) -> tuple[str, dict[str, Any]]:
    model = dict(workload["model"])
    model_id = model_override or str(model["model_id"])
    runtime = dict(workload.get("runtime") or {})
    kwargs: dict[str, Any] = {
        "num_gpus": int(runtime.get("num_gpus", 1)),
        "use_fsdp_inference": bool(runtime.get("use_fsdp_inference", False)),
        "dit_cpu_offload": bool(runtime.get("dit_cpu_offload", False)),
        "vae_cpu_offload": bool(runtime.get("vae_cpu_offload", False)),
        "text_encoder_cpu_offload": bool(
            runtime.get("text_encoder_cpu_offload", True)
        ),
        "image_encoder_cpu_offload": bool(
            runtime.get("image_encoder_cpu_offload", True)
        ),
        "pin_cpu_memory": bool(runtime.get("pin_cpu_memory", False)),
    }
    if model.get("revision"):
        kwargs["revision"] = model["revision"]
    if model.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    for key in ("distributed_executor_backend", "tp_size", "sp_size"):
        if runtime.get(key) is not None:
            kwargs[key] = runtime[key]
    return model_id, kwargs


def apply_mode_env(workload: Mapping[str, Any], mode: str) -> dict[str, str]:
    """Apply mode-specific environment variables and return the applied map."""
    mode_env = dict(workload.get("mode_env") or {})
    if mode in {"optimized", "fused", "candidate"}:
        key = "optimized" if "optimized" in mode_env else "fused"
        selected = dict(mode_env.get(key) or mode_env.get("optimized") or {})
    else:
        selected = dict(mode_env.get(mode) or {})
    for name, value in selected.items():
        os.environ[str(name)] = str(value)
    return {str(k): str(v) for k, v in selected.items()}


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            env["gpu_capability"] = f"{major}.{minor}"
    except Exception as exc:  # pragma: no cover - defensive
        env["torch_error"] = str(exc)
    try:
        import fastvideo

        env["fastvideo"] = getattr(fastvideo, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover
        env["fastvideo_error"] = str(exc)
    return env


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _normalize_mode(mode: str) -> str:
    if mode in {"fused", "candidate"}:
        return "optimized"
    return mode


def run_generation(
    *,
    workload: Mapping[str, Any],
    workload_path: Path,
    mode: str,
    output_dir: Path,
    model_override: str | None = None,
) -> dict[str, Any]:
    applied_env = apply_mode_env(workload, mode)
    attention_backend = (workload.get("sampling") or {}).get("attention_backend")
    if attention_backend:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = str(attention_backend)

    result_mode = _normalize_mode(mode)
    measurement = dict(workload.get("measurement") or {})
    warmups = int(measurement.get("warmups", 1))
    runs = int(measurement.get("runs", 2))
    save_frames = bool(measurement.get("save_frames", True))

    request = build_request(workload, base_dir=workload_path.parent)
    model_id, generator_kwargs = build_generator_kwargs(
        workload, model_override=model_override
    )
    environment = collect_environment()
    environment["mode_env"] = applied_env
    environment["mode_requested"] = mode

    log_path = output_dir / f"{result_mode}.log"
    frames_path: Path | None = None
    timings: list[float] = []
    generation_times: list[float | None] = []
    peak_memory: list[float | None] = []
    failure_reason: str | None = None
    status = "ok"

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(model_id, **generator_kwargs)
        try:
            for _ in range(warmups):
                generator.generate(request)
            for run_index in range(runs):
                if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available():
                    import torch

                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                start = time.perf_counter()
                result = generator.generate(request)
                if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available():
                    import torch

                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
                generation_times.append(getattr(result, "generation_time", None))
                peak_memory.append(getattr(result, "peak_memory_mb", None))
                if save_frames and run_index == 0 and getattr(result, "frames", None) is not None:
                    import numpy as np

                    frames_path = output_dir / f"{result_mode}_frames.npy"
                    np.save(frames_path, np.asarray(result.frames))
        finally:
            generator.shutdown()
    except Exception as exc:  # noqa: BLE001 - record full failure for resume
        status = "failed"
        failure_reason = f"{type(exc).__name__}: {exc}"
        log_path.write_text(traceback.format_exc(), encoding="utf-8")

    payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "mode": result_mode,
        "status": status,
        "workload_id": workload["workload_id"],
        "model_id": model_id,
        "request": {
            # Keep sampling/output metadata only; omit full prompt text from
            # durable results when a prompt_file was used.
            "sampling": request["sampling"],
            "output": request["output"],
            "prompt_source": (
                "prompt_file" if workload.get("prompt_file") else "inline"
            ),
        },
        "warmups": warmups,
        "runs": runs if status == "ok" else len(timings),
        "wall_seconds": timings,
        "median_wall_seconds": (
            statistics.median(timings) if timings else None
        ),
        "generation_seconds": generation_times,
        "peak_memory_mb": peak_memory,
        "environment": environment,
        "stage": "generate",
    }
    if frames_path is not None:
        payload["frames_path"] = str(frames_path)
    if log_path.is_file():
        payload["log_path"] = str(log_path)
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason

    result_path = output_dir / f"{result_mode}_result.json"
    write_json(result_path, payload)
    # Compatibility alias used by the historical Wan A/B script naming.
    if result_mode == "optimized" and mode == "fused":
        write_json(output_dir / "fused_result.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one FastVideo generation mode from a workload manifest"
    )
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("native", "optimized", "fused", "candidate"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        help="Optional model_id override (still resolved through FastVideo)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate workload and print the planned request without loading models",
    )
    args = parser.parse_args(argv)

    try:
        workload = load_workload_dict(args.workload)
    except Exception as exc:  # noqa: BLE001
        print(f"WORKLOAD_LOAD: FAIL\n{exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        request = build_request(workload, base_dir=args.workload.parent)
        model_id, generator_kwargs = build_generator_kwargs(
            workload, model_override=args.model
        )
        plan = {
            "workload_id": workload["workload_id"],
            "mode": args.mode,
            "model_id": model_id,
            "generator_kwargs": generator_kwargs,
            "request": {
                "sampling": request["sampling"],
                "output": request["output"],
                "prompt_chars": len(request["prompt"]),
            },
            "mode_env": apply_mode_env(workload, args.mode),
        }
        # dry-run must not permanently mutate the parent environment beyond
        # what the caller expects; mode env was applied above for visibility.
        print(json.dumps(plan, indent=2))
        return 0

    payload = run_generation(
        workload=workload,
        workload_path=args.workload,
        mode=args.mode,
        output_dir=args.output_dir,
        model_override=args.model,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
