# SPDX-License-Identifier: Apache-2.0
"""Generate synchronized video and audio from text with LTX-2.5."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.api import (
    CompileConfig,
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)

MODEL_IDS = {
    "distilled": "FastVideo/LTX-2.5-Distilled-Diffusers",
    "dev": "FastVideo/LTX-2.5-Dev-Diffusers",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(MODEL_IDS), default="distilled")
    parser.add_argument(
        "--model-path",
        help="FastVideo-converted model directory or Hub ID; the raw split Lightricks repo is not directly loadable.",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/ltx2_5_t2av.mp4")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--torch-compile", action="store_true", help="Compile the transformer denoising path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    is_distilled = args.variant == "distilled"
    model_path = args.model_path or MODEL_IDS[args.variant]
    height = args.height or (1024 if is_distilled else 512)
    width = args.width or (1536 if is_distilled else 768)
    steps = args.steps or (8 if is_distilled else 30)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=model_path,
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
                compile=CompileConfig(enabled=args.torch_compile),
            ),
            pipeline=PipelineSelection(workload_type="t2v"),
        ))
    try:
        result = generator.generate(
            GenerationRequest(
                prompt=args.prompt,
                sampling=SamplingConfig(
                    seed=args.seed,
                    height=height,
                    width=width,
                    num_frames=args.num_frames,
                    fps=24,
                    num_inference_steps=steps,
                    guidance_scale=1.0 if is_distilled else 3.0,
                ),
                output=OutputConfig(
                    output_path=str(output_path),
                    save_video=True,
                    return_frames=False,
                ),
            ))
        print(f"Synchronized video and audio written to: {result.video_path}")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
