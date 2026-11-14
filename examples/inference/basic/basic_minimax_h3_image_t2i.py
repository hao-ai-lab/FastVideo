# SPDX-License-Identifier: Apache-2.0
"""Run MiniMax-H3-Image text-to-image generation through FastVideo.

The checkpoint is guidance-distilled to 6 steps. Sizes below 2048x2048 are
generated on a larger canvas and resized back. Weights load from
``transformer_diffusers/`` and ``vae_diffusers/``.
"""
import argparse
from pathlib import Path

from PIL import Image

from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MiniMax-H3-Image text-to-image generation.")
    parser.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3-Image")
    parser.add_argument("--output", default="image_output/minimax_h3_image.png")
    parser.add_argument(
        "--prompt",
        default="A red fox sitting in a field of tall grass at sunrise, photorealistic.",
    )
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--sp-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tp_size = args.tp_size if args.tp_size is not None else (args.num_gpus if args.num_gpus > 1 else 1)
    sp_size = args.sp_size if args.sp_size is not None else (1 if args.num_gpus > 1 else args.num_gpus)
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=args.model_path,
            engine=EngineConfig(
                num_gpus=args.num_gpus,
                parallelism=ParallelismConfig(tp_size=tp_size, sp_size=sp_size),
            ),
            pipeline=PipelineSelection(workload_type="t2i"),
        ))
    try:
        result = generator.generate(
            GenerationRequest(
                prompt=args.prompt,
                sampling=SamplingConfig(
                    height=args.height,
                    width=args.width,
                    num_frames=1,
                    fps=1,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                ),
                output=OutputConfig(
                    output_path=str(output.parent),
                    save_video=False,
                    return_frames=True,
                ),
            ))
        if isinstance(result, list):
            result = result[0]
        frames = result.frames
        if frames is not None and len(frames):
            Image.fromarray(frames[0]).save(output)
            print(f"Saved image to {output}")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
