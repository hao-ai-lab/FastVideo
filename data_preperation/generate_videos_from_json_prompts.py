#!/usr/bin/env python3
"""Generate videos from a directory of JSON prompt files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable

from fastvideo import VideoGenerator


DEFAULT_MODEL_PATH = "Davids048/LTX2-Base-Diffusers"
DEFAULT_NUM_GPUS = 4
DEFAULT_NUM_FRAMES = 121
DEFAULT_HEIGHT = 1088
DEFAULT_WIDTH = 1920
DEFAULT_FPS = 24
DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_PROMPT_KEY = "video_prompt"
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed "
    "out colors, excessive noise, grainy texture, poor lighting, flickering, "
    "motion blur, distorted proportions, unnatural skin tones, deformed "
    "facial features, asymmetrical face, missing facial features, extra "
    "limbs, disfigured hands, wrong hand count, artifacts around text, "
    "inconsistent perspective, camera shake, incorrect depth of field, "
    "background too sharp, background clutter, distracting reflections, "
    "harsh shadows, inconsistent lighting direction, color banding, "
    "cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated "
    "expressions, wrong gaze direction, mismatched lip sync, silent or "
    "muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, "
    "jittery movement, awkward pauses, incorrect timing, unnatural "
    "transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI "
    "artifacts."
)


@dataclass
class GenerationConfig:
    model_path: str = DEFAULT_MODEL_PATH
    num_gpus: int = DEFAULT_NUM_GPUS
    num_frames: int = DEFAULT_NUM_FRAMES
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    fps: int = DEFAULT_FPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    prompt_key: str = DEFAULT_PROMPT_KEY
    seed: int | None = None
    skip_existing: bool = True


def _extract_sort_key(filename: str) -> tuple[int, int, str]:
    matches = re.findall(r"\d+", filename)
    if matches:
        return (0, int(matches[-1]), filename)
    return (1, 0, filename)


def _iter_prompt_files(prompt_dir: str) -> list[str]:
    files = [
        os.path.join(prompt_dir, name)
        for name in os.listdir(prompt_dir)
        if name.endswith(".json")
    ]
    files.sort(key=lambda path: _extract_sort_key(os.path.basename(path)))
    return files


def _load_prompt(path: str, prompt_key: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if prompt_key not in payload:
        raise KeyError(f"Missing key '{prompt_key}' in {path}")
    prompt = payload[prompt_key]
    if not isinstance(prompt, str):
        raise TypeError(f"Prompt value in {path} must be a string")
    return prompt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _select_slice(
    files: list[str],
    start_idx: int,
    end_idx: int | None,
) -> list[str]:
    if start_idx < 0:
        raise ValueError("start_idx must be >= 0")
    if end_idx is not None and end_idx < start_idx:
        raise ValueError("end_idx must be >= start_idx")
    return files[start_idx:end_idx]


def _resolve_output_path(output_dir: str, prompt_path: str) -> str:
    stem = os.path.splitext(os.path.basename(prompt_path))[0]
    return os.path.join(output_dir, f"{stem}.mp4")


def _generate_videos(
    prompt_files: Iterable[str],
    output_dir: str,
    config: GenerationConfig,
) -> None:
    _ensure_dir(output_dir)
    generator = VideoGenerator.from_pretrained(
        config.model_path,
        num_gpus=config.num_gpus,
    )
    try:
        for offset, prompt_path in enumerate(prompt_files):
            output_path = _resolve_output_path(output_dir, prompt_path)
            if config.skip_existing and os.path.exists(output_path):
                print(f"[skip] {output_path} already exists")
                continue

            prompt = _load_prompt(prompt_path, config.prompt_key)
            print(f"[gen] {prompt_path} -> {output_path}")
            generator.generate_video(
                prompt=prompt,
                output_path=output_path,
                save_video=True,
                num_frames=config.num_frames,
                height=config.height,
                width=config.width,
                fps=config.fps,
                negative_prompt=config.negative_prompt,
                guidance_scale=config.guidance_scale,
            )
    finally:
        generator.shutdown()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate videos from a directory of JSON prompt files."
    )
    parser.add_argument(
        "--prompt-dir",
        required=True,
        help="Directory containing JSON prompt files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write generated videos.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index (inclusive) into sorted prompt files.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index (exclusive) into sorted prompt files.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=DEFAULT_NUM_GPUS,
        help="Number of GPUs to use.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Model path or identifier.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of frames to generate per video.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="Output video height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="Output video width.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Output video FPS.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for generation.",
    )
    parser.add_argument(
        "--prompt-key",
        default=DEFAULT_PROMPT_KEY,
        help="JSON key to read prompt from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed (increments by index).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip prompts with existing outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    prompt_dir = os.path.abspath(args.prompt_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(prompt_dir):
        print(f"Prompt dir not found: {prompt_dir}", file=sys.stderr)
        return 1

    files = _iter_prompt_files(prompt_dir)
    if not files:
        print(f"No .json prompt files found in {prompt_dir}", file=sys.stderr)
        return 1

    selected_files = _select_slice(files, args.start_idx, args.end_idx)
    if not selected_files:
        print(
            "No prompt files selected after applying range.",
            file=sys.stderr,
        )
        return 1

    config = GenerationConfig(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        prompt_key=args.prompt_key,
        seed=args.seed,
        skip_existing=not args.no_skip_existing,
    )

    print(
        "Generating videos with config:\n"
        f"  model_path={config.model_path}\n"
        f"  num_gpus={config.num_gpus}\n"
        f"  num_frames={config.num_frames}\n"
        f"  height={config.height}\n"
        f"  width={config.width}\n"
        f"  fps={config.fps}\n"
        f"  guidance_scale={config.guidance_scale}\n"
        f"  prompt_key={config.prompt_key}\n"
        f"  start_idx={args.start_idx}\n"
        f"  end_idx={args.end_idx}\n"
        f"  output_dir={output_dir}\n"
    )

    _generate_videos(selected_files, output_dir, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
