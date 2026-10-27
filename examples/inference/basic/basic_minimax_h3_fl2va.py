# SPDX-License-Identifier: Apache-2.0
"""Generate synchronized video/audio from a first frame with MiniMax H3."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    InputConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    SamplingConfig,
)
from fastvideo.pipelines.basic.minimax_h3.packing import resolve_canvas_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3")
    # Rank-reduced AdaLN checkpoint (-39% params, -23 GiB VRAM): pass
    #   --model-path noctuashap/MiniMax-H3-pruned-r16
    # (or a local dir produced by
    #   scripts/checkpoint_conversion/convert_minimax_h3_adaln_rank.py).
    # adaln_rank is read from the checkpoint config; no other flags needed.
    # Rank-reduced checkpoints are inference-only: training needs the
    # full-rank release.
    parser.add_argument("--image", required=True, help="First-frame image path.")
    parser.add_argument("--last-image", help="Optional last-frame image path.")
    parser.add_argument("--output", default="outputs/minimax_h3_fl2va")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--num-frames", type=int, default=192, help="192 frames is exactly 8 seconds at 24 fps.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first_image = Image.open(args.image).convert("RGB")
    last_image = Image.open(args.last_image).convert("RGB") if args.last_image else None
    height, width = resolve_canvas_size(*first_image.size)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=args.model_path,
            engine=EngineConfig(
                num_gpus=args.num_gpus,
                use_fsdp_inference=args.num_gpus > 1,
                parallelism=ParallelismConfig(tp_size=1, sp_size=args.num_gpus),
                offload=OffloadConfig(
                    dit=False,
                    dit_layerwise=False,
                    text_encoder=True,
                    vae=True,
                    pin_cpu_memory=False,
                ),
            ),
        ))
    try:
        result = generator.generate(
            GenerationRequest(
                prompt=args.prompt,
                negative_prompt="",
                inputs=InputConfig(pil_image=first_image, last_image=last_image),
                sampling=SamplingConfig(
                    height=height,
                    width=width,
                    num_frames=args.num_frames,
                    fps=24,
                    num_inference_steps=args.steps,
                    guidance_scale=1.0,
                    batch_cfg=False,
                    seed=args.seed,
                ),
                output=OutputConfig(
                    output_path=str(output_dir / "minimax_h3_fl2va.mp4"),
                    save_video=True,
                    return_frames=False,
                ),
            ))
        print(f"Output written to: {result.video_path}")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
