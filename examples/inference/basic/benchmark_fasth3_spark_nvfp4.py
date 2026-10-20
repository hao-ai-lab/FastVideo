# SPDX-License-Identifier: Apache-2.0
"""Time a resident FastH3 eight-forward Spark recipe with the release prompts.

One process loads the model, runs one excluded ceramics warmup, then times
each prompt at least twice. Every call writes a video. This script does not alter
the V2 schedule, VSA sparsity, or video resolution.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from copy import deepcopy
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.api.parser import load_raw_config, parse_config
from fastvideo.api.schema import RunConfig

PROMPT_IDS = ("latency-ceramics-005", "latency-harbor-005")


def _stage_metrics(result: object) -> dict[str, dict]:
    logging_info = getattr(result, "logging_info", None)
    stages = getattr(logging_info, "stages", None)
    if isinstance(logging_info, dict):
        stages = logging_info.get("stages", stages)
    if not isinstance(stages, dict):
        return {}
    return {name: metrics for name, metrics in stages.items() if isinstance(metrics, dict)}


def _stage_seconds(metrics: dict[str, dict]) -> dict[str, float]:
    return {name: float(stage["execution_time"]) for name, stage in metrics.items()
            if stage.get("execution_time") is not None}


def _peak_mb(metrics: dict[str, dict], key: str) -> float | None:
    values = [float(stage[key]) for stage in metrics.values() if stage.get(key) is not None]
    return max(values) if values else None


def _stage_total(stages: dict[str, float], fragment: str, exclude: str | None = None) -> float | None:
    matches = [
        seconds for name, seconds in stages.items()
        if fragment in name.lower() and (exclude is None or exclude not in name.lower())
    ]
    return sum(matches) if matches else None


def _request(base: RunConfig, prompt: str, frames: int, width: int, height: int,
             output: Path):
    request = deepcopy(base.request)
    request.prompt = prompt
    request.inputs.prompt_path = None
    request.sampling.num_frames = frames
    request.sampling.width = width
    request.sampling.height = height
    request.output.output_path = str(output)
    request.output.save_video = True
    request.output.return_frames = False
    return request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    if args.frames not in (124, 243):
        parser.error("use 124 frames for roughly five seconds or 243 for roughly ten seconds")
    if args.repeats < 2:
        parser.error("the release protocol requires at least two timed calls")

    config = parse_config(RunConfig, load_raw_config(args.config))
    if args.model_path:
        config.generator.model_path = str(args.model_path)
    if config.request.sampling.num_inference_steps != 9:
        parser.error("the V2 contract requires nine sigma points for eight DiT forwards")
    if config.generator.engine.offload.lazy_module_load is not False:
        parser.error("the resident recipe requires lazy_module_load: false")
    contract = Path(config.generator.model_path) / "fastvideo_inference.json"
    if not contract.is_file():
        parser.error(f"missing trained V2 schedule: {contract}")
    inference = json.loads(contract.read_text())
    if inference.get("num_inference_steps") != 9 or inference.get("transformer_forwards") != 8:
        parser.error("the checkpoint is not the trained V2 eight-forward schedule")

    prompts = json.loads(args.prompts.read_text())
    if any(prompt_id not in prompts for prompt_id in PROMPT_IDS):
        parser.error(f"prompt JSON must contain {', '.join(PROMPT_IDS)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generator = VideoGenerator.from_config(config.generator)
    try:
        for prompt_index, prompt_id in enumerate(PROMPT_IDS):
            times = []
            first_index = 0 if prompt_index == 0 else 1
            for index in range(first_index, args.repeats + 1):
                warmup = index == 0
                label = "warmup" if warmup else f"run-{index:02d}"
                requested_path = args.output_dir / f"{prompt_id}-{args.width}x{args.height}-{args.frames}-{label}.mp4"
                request = _request(config, prompts[prompt_id], args.frames, args.width, args.height,
                                   requested_path)
                started = time.perf_counter()
                result = generator.generate(request)
                wall = time.perf_counter() - started
                output = Path(result.video_path) if result.video_path else requested_path
                if not output.is_file():
                    raise RuntimeError(f"generation returned without an MP4: {output}")
                metrics = _stage_metrics(result)
                stages = _stage_seconds(metrics)
                row = {
                    "prompt_id": prompt_id,
                    "warmup": warmup,
                    "frames": args.frames,
                    "width": args.width,
                    "height": args.height,
                    "e2e_seconds": round(wall, 3),
                    "denoise_seconds": _stage_total(stages, "denois"),
                    "decode_seconds": _stage_total(stages, "decod", exclude="postdecode"),
                    "postprocess_seconds": _stage_total(stages, "postdecode"),
                    "peak_memory_mb": _peak_mb(metrics, "peak_allocated_mb"),
                    "peak_reserved_mb": _peak_mb(metrics, "peak_reserved_mb"),
                    "result_peak_memory_mb": result.peak_memory_mb,
                    "stages": stages,
                    "mp4": str(output),
                }
                print(json.dumps(row, sort_keys=True), flush=True)
                if not warmup:
                    times.append(wall)
            print(json.dumps({"prompt_id": prompt_id, "timed_runs": len(times),
                              "median_e2e_seconds": round(statistics.median(times), 3)},
                             sort_keys=True), flush=True)
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
