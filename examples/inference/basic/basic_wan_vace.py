# SPDX-License-Identifier: Apache-2.0
"""Official Wan2.1-VACE reference-image generation example."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.api.presets import get_preset

WAN21_EXAMPLES_BASE = "https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/examples"
DEFAULT_MODEL = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
DEFAULT_ASSETS_DIR = Path("outputs/wan_vace/assets")

PROMPT = (
    "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。"
    "她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。"
    "蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。"
    "小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。"
    "周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
)


def _ensure_reference_assets(assets_dir: Path) -> list[str]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for name in ("girl.png", "snake.png"):
        dest = assets_dir / name
        if not dest.is_file():
            urllib.request.urlretrieve(f"{WAN21_EXAMPLES_BASE}/{name}", dest)
        paths.append(str(dest))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--output_path", default="outputs/wan_vace/official_1_3b_480.mp4")
    parser.add_argument("--assets_dir", default=str(DEFAULT_ASSETS_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dit_cpu_offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text_encoder_cpu_offload", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    preset_name = "wan_vace_14b" if "14b" in args.model_path.lower() else "wan_vace_1_3b"
    preset = get_preset(preset_name, "wan")
    height = args.height or preset.defaults["height"]
    width = args.width or preset.defaults["width"]

    references = _ensure_reference_assets(Path(args.assets_dir))

    generator = VideoGenerator.from_pretrained(
        args.model_path,
        num_gpus=1,
        dit_cpu_offload=args.dit_cpu_offload,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=args.text_encoder_cpu_offload,
    )
    try:
        generator.generate_video(
            PROMPT,
            negative_prompt=preset.defaults["negative_prompt"],
            references=references,
            height=height,
            width=width,
            num_frames=preset.defaults["num_frames"],
            fps=preset.defaults["fps"],
            guidance_scale=preset.defaults["guidance_scale"],
            num_inference_steps=preset.defaults["num_inference_steps"],
            conditioning_scale=1.0,
            output_path=args.output_path,
            save_video=True,
            seed=args.seed,
        )
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
