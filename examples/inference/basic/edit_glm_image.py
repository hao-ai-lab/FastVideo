# SPDX-License-Identifier: Apache-2.0
"""Run GLM-Image image-to-image (edit) generation through FastVideo.

User story:
    "I have the HF `zai-org/GLM-Image` checkpoint and a condition image, and
    want a minimal edit command (text + image -> edited image), saved as a PNG."

GLM-Image is a single unified pipeline: passing a condition image switches it
from text-to-image to the edit path (the condition enters the DiT via a KV-cache
write pass), so the generator config is identical to `basic_glm_image.py` — the
`inputs.pil_image` on the request is what selects the edit mode.
"""
import argparse
from pathlib import Path

from PIL import Image

from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    InputConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GLM-Image image-to-image (edit) generation.")
    parser.add_argument(
        "--model-path",
        default="zai-org/GLM-Image",
        help="HF id or local diffusers-format GLM-Image weights directory.",
    )
    parser.add_argument(
        "--image",
        default="assets/images/couple.jpg",
        help="Condition image to edit.",
    )
    parser.add_argument(
        "--output",
        default="image_output/edited.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--prompt",
        default="Change the background to a snowy mountain landscape at golden hour.",
        help="Edit instruction.",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--sp-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    condition = Image.open(args.image).convert("RGB")
    tp_size = args.tp_size if args.tp_size is not None else (args.num_gpus if args.num_gpus > 1 else 1)
    sp_size = args.sp_size if args.sp_size is not None else (1 if args.num_gpus > 1 else args.num_gpus)

    # GLM-Image needs trust_remote_code for its AR encoder; offload and the
    # pipeline class come from the model's registered defaults — don't override.
    # The pipeline is registered as t2i; passing inputs.pil_image below switches
    # it to the edit path.
    generator_config = GeneratorConfig(
        model_path=args.model_path,
        trust_remote_code=True,
        engine=EngineConfig(
            num_gpus=args.num_gpus,
            parallelism=ParallelismConfig(tp_size=tp_size, sp_size=sp_size),
        ),
        pipeline=PipelineSelection(workload_type="t2i"),
    )

    generator = VideoGenerator.from_config(generator_config)
    try:
        request = GenerationRequest(
            prompt=args.prompt,
            inputs=InputConfig(pil_image=condition),
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
        )
        result = generator.generate(request)
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
