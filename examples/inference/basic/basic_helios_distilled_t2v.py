# SPDX-License-Identifier: Apache-2.0
"""Generate one Helios-Distilled T2V chunk through FastVideo's typed API.

Set ``HELIOS_MODEL_PATH`` to a local snapshot to avoid downloading the public
checkpoint again. The 33-frame example is a short integration run; increase
``num_frames`` to 240 for the official eight-chunk default.
"""

from __future__ import annotations

import os

from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    SamplingConfig,
)

MODEL_PATH = os.getenv("HELIOS_MODEL_PATH", "BestWishYsh/Helios-Distilled")
OUTPUT_PATH = os.getenv(
    "HELIOS_OUTPUT_PATH",
    "outputs_video/helios/helios_distilled_t2v.mp4",
)
PROMPT = ("A vibrant tropical fish swims gracefully through a colorful coral reef "
          "in clear turquoise water, cinematic close-up, fluid motion, vivid detail.")
NEGATIVE_PROMPT = ("Bright tones, overexposed, static, blurred details, subtitles, paintings, "
                   "images, overall gray, worst quality, low quality, JPEG artifacts, ugly, "
                   "deformed, disfigured, still picture, messy background.")


def main() -> None:
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=MODEL_PATH,
            engine=EngineConfig(
                num_gpus=1,
                use_fsdp_inference=False,
                offload=OffloadConfig(
                    dit=False,
                    dit_layerwise=True,
                    text_encoder=True,
                    vae=True,
                    pin_cpu_memory=False,
                ),
            ),
        ))
    request = GenerationRequest(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        sampling=SamplingConfig(
            seed=42,
            height=384,
            width=640,
            num_frames=33,
            fps=24,
            num_inference_steps=2,
            guidance_scale=1.0,
            pyramid_num_inference_steps_list=[2, 2, 2],
            history_sizes=[16, 2, 1],
            num_latent_frames_per_chunk=9,
            keep_first_frame=True,
            is_skip_first_chunk=False,
            use_zero_init=True,
            zero_steps=1,
            is_amplify_first_chunk=True,
        ),
        output=OutputConfig(
            output_path=OUTPUT_PATH,
            save_video=True,
            return_frames=False,
        ),
    )

    try:
        generator.generate(request=request)
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
