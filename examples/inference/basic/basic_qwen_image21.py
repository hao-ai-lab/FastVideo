# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1 generation, editing, references and transparent PNG output."""

import argparse

from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    InputConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--revision")
    parser.add_argument("--prompt", default="A red ceramic teapot, isolated on a transparent background.")
    parser.add_argument("--reference", action="append", default=[], help="Ordered reference image; repeat up to ten times")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--reference-resolution", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-kv-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="outputs/qwen_image21/image.png")
    args = parser.parse_args()
    if len(args.reference) > 10:
        parser.error("At most ten ordered reference images are supported")
    workload = "i2i" if args.reference else "t2i"
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=args.model_path,
            revision=args.revision,
            engine=EngineConfig(
                num_gpus=1,
                parallelism=ParallelismConfig(tp_size=1, sp_size=1),
                offload=OffloadConfig(dit=True, dit_layerwise=True, text_encoder=True, vae=True,
                                      lazy_module_load=True),
            ),
            pipeline=PipelineSelection(workload_type=workload),
        ))
    try:
        generator.generate(
            GenerationRequest(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                inputs=InputConfig(references=args.reference or None),
                sampling=SamplingConfig(height=args.height, width=args.width, num_frames=1, fps=1,
                                        num_inference_steps=args.steps, guidance_scale=1.0,
                                        true_cfg_scale=args.true_cfg_scale, seed=args.seed,
                                        reference_resolution=args.reference_resolution, use_kv_cache=args.use_kv_cache),
                output=OutputConfig(output_path=args.output, save_video=True, return_frames=False),
            ))
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
