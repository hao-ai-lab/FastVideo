# SPDX-License-Identifier: Apache-2.0
"""Run Z-Image-Turbo text-to-image generation through FastVideo.

User story:
    "I want the official Z-Image-Turbo defaults and a deterministic PNG from
    a local or Hugging Face checkpoint."
"""

import argparse
from pathlib import Path

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


DEFAULT_PROMPT = (
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
    "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
    "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
    "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
)
DEFAULT_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Z-Image-Turbo text-to-image generation.")
    parser.add_argument("--model-path", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", default="outputs/zimage/zimage_turbo.png")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--cfg-normalization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cfg-truncation", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=args.model_path,
            revision=args.revision,
            engine=EngineConfig(
                num_gpus=1,
                parallelism=ParallelismConfig(tp_size=1, sp_size=1),
                use_fsdp_inference=False,
            ),
            # The model registry selects the native zimage_turbo preset.
            pipeline=PipelineSelection(workload_type="t2i"),
        ))
    try:
        generator.generate(
            GenerationRequest(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                sampling=SamplingConfig(
                    height=args.height,
                    width=args.width,
                    num_frames=1,
                    fps=1,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    max_sequence_length=args.max_sequence_length,
                    cfg_normalization=args.cfg_normalization,
                    cfg_truncation=args.cfg_truncation,
                    seed=args.seed,
                ),
                output=OutputConfig(
                    output_path=str(output),
                    save_video=True,
                    return_frames=False,
                ),
            ))
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
