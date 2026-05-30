# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from fastvideo import VideoGenerator

DEFAULT_PROMPTS = (
    "A small robot sketches a city skyline at sunrise, cinematic lighting.",
    "A glass teapot steams on a wooden table while rain falls outside.",
)


def _build_init_kwargs(args: argparse.Namespace, *, dynamic: bool) -> dict[str, Any]:
    return {
        "num_gpus": args.num_gpus,
        "sp_size": args.sp_size,
        "tp_size": args.tp_size,
        "use_fsdp_inference": args.use_fsdp_inference,
        "dit_cpu_offload": False,
        "dit_layerwise_offload": False,
        "flow_shift": args.flow_shift,
        "text_encoder_precisions": ("fp32",),
        "output_type": "latent",
        "batching_mode": "dynamic" if dynamic else "disabled",
        "batching_max_size": args.batch_size if dynamic else 1,
        "batching_delay_ms": 0.0,
    }


def _request_kwargs(args: argparse.Namespace, prompt_index: int) -> dict[str, Any]:
    return {
        "prompt": args.prompts[prompt_index],
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "embedded_cfg_scale": args.embedded_cfg_scale,
        "seed": args.seed + prompt_index,
        "fps": 24,
        "save_video": False,
        "return_frames": True,
        "output_path": str(Path(args.output_dir) / f"request_{prompt_index}.mp4"),
    }


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_sequential(generator: VideoGenerator, args: argparse.Namespace) -> tuple[list[dict[str, Any]], float]:
    _sync()
    start = time.perf_counter()
    results = []
    for index in range(args.batch_size):
        kwargs = _request_kwargs(args, index)
        prompt = kwargs.pop("prompt")
        results.append(generator.generate_video(prompt=prompt, **kwargs))
    _sync()
    return results, time.perf_counter() - start


def _run_dynamic(generator: VideoGenerator, args: argparse.Namespace) -> tuple[list[dict[str, Any]], float]:
    if not hasattr(generator, "generate_video_batch"):
        raise RuntimeError("VideoGenerator.generate_video_batch is unavailable in this checkout")
    requests = [_request_kwargs(args, index) for index in range(args.batch_size)]
    _sync()
    start = time.perf_counter()
    results = generator.generate_video_batch(requests)
    _sync()
    return results, time.perf_counter() - start


def _tensor_metrics(sequential: list[dict[str, Any]], dynamic: list[dict[str, Any]]) -> dict[str, Any]:
    per_request = []
    for index, (seq_result, dyn_result) in enumerate(zip(sequential, dynamic, strict=True)):
        seq = seq_result["samples"].detach().cpu().to(torch.float32)
        dyn = dyn_result["samples"].detach().cpu().to(torch.float32)
        diff = (seq - dyn).abs()
        per_request.append({
            "index": index,
            "shape": list(seq.shape),
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
            "allclose_atol_1e_5": bool(torch.allclose(seq, dyn, atol=1e-5, rtol=1e-5)),
            "allclose_atol_1e_4": bool(torch.allclose(seq, dyn, atol=1e-4, rtol=1e-4)),
        })
    return {
        "per_request": per_request,
        "max_abs_diff": max(item["max_abs_diff"] for item in per_request),
        "mean_abs_diff": sum(item["mean_abs_diff"] for item in per_request) / len(per_request),
        "allclose_atol_1e_5": all(item["allclose_atol_1e_5"] for item in per_request),
        "allclose_atol_1e_4": all(item["allclose_atol_1e_4"] for item in per_request),
    }


def run_parity(args: argparse.Namespace) -> dict[str, Any]:
    generator = VideoGenerator.from_pretrained(args.model_path, **_build_init_kwargs(args, dynamic=True))
    try:
        sequential, sequential_s = _run_sequential(generator, args)
        dynamic, dynamic_s = _run_dynamic(generator, args)
        metrics = _tensor_metrics(sequential, dynamic)
    finally:
        generator.shutdown()
    return {
        "mode": "parity",
        "model_path": args.model_path,
        "num_gpus": args.num_gpus,
        "shape": {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
        },
        "batch_size": args.batch_size,
        "sequential_time_s": sequential_s,
        "dynamic_time_s": dynamic_s,
        "speedup": sequential_s / dynamic_s if dynamic_s > 0 else None,
        "tensor_metrics": metrics,
    }


def run_benchmark(args: argparse.Namespace, *, dynamic: bool) -> dict[str, Any]:
    generator = VideoGenerator.from_pretrained(args.model_path, **_build_init_kwargs(args, dynamic=dynamic))
    run = _run_dynamic if dynamic else _run_sequential
    try:
        for _ in range(args.warmup_runs):
            run(generator, args)
        times = []
        for _ in range(args.measurement_runs):
            _results, elapsed = run(generator, args)
            times.append(elapsed)
    finally:
        generator.shutdown()
    avg = sum(times) / len(times)
    return {
        "mode": "dynamic" if dynamic else "sequential",
        "model_path": args.model_path,
        "num_gpus": args.num_gpus,
        "batch_size": args.batch_size,
        "measurement_runs": args.measurement_runs,
        "times_s": times,
        "avg_time_s": avg,
        "requests_per_second": args.batch_size / avg if avg > 0 else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("parity", "sequential", "dynamic"), required=True)
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--sp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--use-fsdp-inference", action="store_true")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--embedded-cfg-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measurement-runs", type=int, default=3)
    parser.add_argument("--output-dir", default="/tmp/fastvideo_dynamic_batching")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.prompts) < args.batch_size:
        raise ValueError("--prompts must contain at least --batch-size prompts")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "parity":
        result = run_parity(args)
    else:
        result = run_benchmark(args, dynamic=args.mode == "dynamic")
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
