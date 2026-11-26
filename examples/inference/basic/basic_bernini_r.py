# SPDX-License-Identifier: Apache-2.0
"""Single-video editing with a locally prepared Bernini-R 1.3B checkpoint."""
import argparse

from fastvideo import VideoGenerator
from fastvideo.api import (EngineConfig, GenerationRequest, GeneratorConfig, InputConfig, OffloadConfig,
                           OutputConfig, ParallelismConfig, SamplingConfig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Output of bernini_r_to_diffusers.py")
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs_bernini_r")
    args = parser.parse_args()
    generator = VideoGenerator.from_config(GeneratorConfig(
        model_path=args.model,
        engine=EngineConfig(num_gpus=1, use_fsdp_inference=False,
                            parallelism=ParallelismConfig(tp_size=1, sp_size=1),
                            offload=OffloadConfig(dit=True, dit_layerwise=False, text_encoder=True, vae=True)),
    ))
    request = GenerationRequest(
        prompt=args.prompt, negative_prompt="",
        inputs=InputConfig(video_path=args.video),
        sampling=SamplingConfig(height=args.height, width=args.width, num_frames=args.frames,
                                num_inference_steps=args.steps, fps=args.fps, guidance_scale=4.0, seed=args.seed),
        output=OutputConfig(output_path=args.output, save_video=True, return_frames=False),
    )
    try:
        result = generator.generate(request=request)
        print(result.video_path)
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
