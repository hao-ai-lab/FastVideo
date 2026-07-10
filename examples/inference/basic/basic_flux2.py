# SPDX-License-Identifier: Apache-2.0
"""Run full Flux2 text-to-image generation through FastVideo.

User story:
    "I have a local or HF Diffusers-format full Flux2 checkpoint and want a
    minimal text-to-image generation command that uses embedded guidance."
"""
import argparse
import os
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.api import (
    ComponentConfig,
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Flux2 text-to-image generation.")
    parser.add_argument(
        "--model-path",
        default="black-forest-labs/FLUX.2-dev",
        help="HF id or local diffusers-format full Flux2 weights directory.",
    )
    parser.add_argument(
        "--output",
        default="outputs/flux2/flux2.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--prompt",
        default="a photo of a banana on a wooden table, studio lighting",
        help="Text prompt.",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--sp-size", type=int, default=None)
    parser.add_argument(
        "--backend",
        default=None,
        help="Set FASTVIDEO_ATTENTION_BACKEND, for example TORCH_SDPA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = args.backend

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tp_size = args.tp_size if args.tp_size is not None else (
        args.num_gpus if args.num_gpus > 1 else 1
    )
    sp_size = args.sp_size if args.sp_size is not None else (
        1 if args.num_gpus > 1 else args.num_gpus
    )

    generator_config = GeneratorConfig(
        model_path=args.model_path,
        engine=EngineConfig(
            num_gpus=args.num_gpus,
            parallelism=ParallelismConfig(tp_size=tp_size, sp_size=sp_size),
            use_fsdp_inference=False,
            offload=OffloadConfig(
                dit=False,
                vae=True,
                text_encoder=True,
                pin_cpu_memory=False,
            ),
        ),
        pipeline=PipelineSelection(
            workload_type="t2i",
            components=ComponentConfig(override_pipeline_cls_name="Flux2Pipeline"),
        ),
    )

    generator = VideoGenerator.from_config(generator_config)
    try:
        sampling = SamplingConfig(
            height=args.height,
            width=args.width,
            num_frames=1,
            fps=1,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        extensions = {}
        if args.max_sequence_length is not None:
            extensions["max_sequence_length"] = args.max_sequence_length

        request = GenerationRequest(
            prompt=args.prompt,
            sampling=sampling,
            output=OutputConfig(
                output_path=str(output),
                save_video=True,
                return_frames=False,
            ),
            extensions=extensions,
        )
        generator.generate(request)
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
