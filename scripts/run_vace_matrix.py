#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the official Wan-VACE resolution × parameter matrix on local GPUs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.api.presets import get_preset
from fastvideo.models.wan.pipeline_config import WanVACE1_3B_Config, WanVACE14B_Config

ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIR = ROOT / "outputs" / "vace_matrix"
ASSETS_DIR = MATRIX_DIR / "assets"
RESULTS_JSON = MATRIX_DIR / "results.json"

# Reuse the example's prompt and reference-asset download so both stay in sync.
_EXAMPLE_SPEC = importlib.util.spec_from_file_location("basic_wan_vace",
                                                       ROOT / "examples/inference/basic/basic_wan_vace.py")
assert _EXAMPLE_SPEC is not None and _EXAMPLE_SPEC.loader is not None
_EXAMPLE = importlib.util.module_from_spec(_EXAMPLE_SPEC)
_EXAMPLE_SPEC.loader.exec_module(_EXAMPLE)

PRESET_1_3B = get_preset("wan_vace_1_3b", "wan")
PRESET_14B = get_preset("wan_vace_14b", "wan")
CONFIG_1_3B = WanVACE1_3B_Config()
CONFIG_14B = WanVACE14B_Config()

PROMPT = _EXAMPLE.PROMPT
NEGATIVE_PROMPT = PRESET_1_3B.defaults["negative_prompt"]
MATRIX_SEED = 0

RUNS = (
    {
        "name": "vace_1_3b_480",
        "model_path": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        "preset": PRESET_1_3B,
        "config": CONFIG_1_3B,
        "height": 480,
        "width": 832,
        "dit_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "vae_cpu_offload": False,
    },
    {
        "name": "vace_14b_480",
        "model_path": "Wan-AI/Wan2.1-VACE-14B-diffusers",
        "preset": PRESET_14B,
        "config": CONFIG_14B,
        "height": 480,
        "width": 832,
        "dit_cpu_offload": False,
        "text_encoder_cpu_offload": False,
        "vae_cpu_offload": False,
    },
    {
        "name": "vace_14b_720",
        "model_path": "Wan-AI/Wan2.1-VACE-14B-diffusers",
        "preset": PRESET_14B,
        "config": CONFIG_14B,
        "height": 720,
        "width": 1280,
        "dit_cpu_offload": False,
        "text_encoder_cpu_offload": False,
        "vae_cpu_offload": False,
    },
)


def _count_frames(video_path: Path) -> int:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(probe.stdout.strip())


def _run_case(case: dict, references: list[str]) -> dict:
    preset = case["preset"]
    output_name = case["name"]
    output_path = MATRIX_DIR / f"{output_name}.mp4"
    if output_path.exists():
        output_path.unlink()
    started = time.perf_counter()
    generator = VideoGenerator.from_pretrained(
        case["model_path"],
        num_gpus=1,
        dit_cpu_offload=case["dit_cpu_offload"],
        text_encoder_cpu_offload=case["text_encoder_cpu_offload"],
        vae_cpu_offload=case["vae_cpu_offload"],
    )
    try:
        generator.generate_video(
            PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            references=references,
            height=case["height"],
            width=case["width"],
            num_frames=preset.defaults["num_frames"],
            fps=preset.defaults["fps"],
            guidance_scale=preset.defaults["guidance_scale"],
            num_inference_steps=preset.defaults["num_inference_steps"],
            conditioning_scale=1.0,
            output_path=str(output_path),
            save_video=True,
            seed=MATRIX_SEED,
        )
    finally:
        generator.shutdown()
    elapsed_s = time.perf_counter() - started
    if not output_path.is_file():
        raise FileNotFoundError(f"Expected output video at {output_path}")
    frame_count = _count_frames(output_path)
    expected_frames = preset.defaults["num_frames"]
    if frame_count != expected_frames:
        raise RuntimeError(f"{output_path} has {frame_count} frames, expected {expected_frames}")
    return {
        "name": output_name,
        "output_path": str(output_path),
        "elapsed_s": round(elapsed_s, 2),
        "frame_count": frame_count,
        "model_path": case["model_path"],
        "height": case["height"],
        "width": case["width"],
        "flow_shift": case["config"].flow_shift,
        "num_inference_steps": preset.defaults["num_inference_steps"],
        "guidance_scale": preset.defaults["guidance_scale"],
        "conditioning_scale": 1.0,
        "seed": MATRIX_SEED,
        "references": references,
        "official_case": "Wan2.1 generate.py vace EXAMPLE_PROMPT",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=[case["name"] for case in RUNS], default=None)
    args = parser.parse_args()

    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    references = _EXAMPLE._ensure_reference_assets(ASSETS_DIR)
    if RESULTS_JSON.is_file():
        prior = json.loads(RESULTS_JSON.read_text()).get("runs", [])
    else:
        prior = []
    prior_by_name = {entry["name"]: entry for entry in prior}
    selected_runs = [case for case in RUNS if case["name"] == args.only] if args.only else list(RUNS)
    results: list[dict] = []
    for case in selected_runs:
        print(f"Running {case['name']} ...", flush=True)
        results.append(_run_case(case, references))
        prior_by_name[case["name"]] = results[-1]
        print(f"Finished {case['name']} -> {results[-1]['output_path']} in {results[-1]['elapsed_s']}s", flush=True)
    merged = [prior_by_name[case["name"]] for case in RUNS if case["name"] in prior_by_name]
    RESULTS_JSON.write_text(json.dumps({"runs": merged}, indent=2))
    print(json.dumps({"runs": merged}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
