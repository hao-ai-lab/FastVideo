# SPDX-License-Identifier: Apache-2.0
"""Generate a single RGB image from a detailed caption with Pyramid-JiT."""

import argparse

from fastvideo import VideoGenerator
from fastvideo.api import EngineConfig, GeneratorConfig, OffloadConfig, PipelineSelection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-path", default="Linum-AI/pjit-21k")
    parser.add_argument("--qwen-model-path", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/pjit/image.png")
    args = parser.parse_args()

    generator = VideoGenerator.from_pretrained(
        GeneratorConfig(
            model_path=args.model_path,
            engine=EngineConfig(num_gpus=1, offload=OffloadConfig(dit_layerwise=False, lazy_module_load=True)),
            pipeline=PipelineSelection(workload_type="t2i", experimental={"qwen_model_path": args.qwen_model_path}),
        ))
    try:
        generator.generate({
            "prompt": args.prompt,
            "sampling": {
                "height": args.height,
                "width": args.width,
                "num_inference_steps": args.steps,
                "seed": args.seed,
            },
            "output": {"output_path": args.output, "save_video": True},
        })
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
