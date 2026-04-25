# SPDX-License-Identifier: Apache-2.0
"""daVinci-MagiHuman text-to-video inference.

Requirements:
  - HF_TOKEN with access to GAIR/daVinci-MagiHuman (gated) and
    google/t5gemma-9b (gated).
  - GPU: A100 80GB or H100 (15B model, ~30GB weights + VAE + text encoder).

Usage:
  python examples/inference/basic/basic_davinci_magihuman.py
"""

from fastvideo import VideoGenerator
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.api.davinci_magihuman import DaVinciMagiHumanSamplingParam


PROMPTS = [
    "A large metal cylinder is seen pressing down on a pile of Oreo cookies, "
    "flattening them as if they were under a hydraulic press.",
    "A large metal cylinder is seen compressing colorful clay into a compact "
    "shape, demonstrating the power of a hydraulic press.",
    "A large metal cylinder is seen pressing down on a pile of colorful "
    "candies, flattening them as if they were under a hydraulic press. "
    "The candies are crushed and broken into small pieces, creating a mess "
    "on the table.",
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-tiny", action="store_true",
                        help="Use tiny resolution (128x128, 5 frames) to "
                             "isolate SDPA memory issues vs genuine OOB")
    args = parser.parse_args()

    model_path = "/weights/davinci/base"

    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    sampling_param = DaVinciMagiHumanSamplingParam()
    if args.debug_tiny:
        # 128×128, 5 frames → S ≈ 5*64*64 + 256 audio + ~100 text ≈ 20K tokens
        # (vs 129600 at 720p). If the SDPA error disappears here, the issue
        # is memory pressure from the math backend at large S.
        sampling_param.height = 128
        sampling_param.width = 128
        sampling_param.num_frames = 5
        sampling_param.num_inference_steps = 5  # small for debug but avoids scheduler edge case
        print(f"[debug-tiny] height=128, width=128, num_frames=5, steps=2")

    for i, prompt in enumerate(PROMPTS[:1]):  # one prompt in debug mode
        generator.generate_video(
            prompt,
            sampling_param=sampling_param,
            output_path=f"outputs_video/davinci_magihuman_{i:02d}.mp4",
            save_video=True,
        )

    generator.shutdown()


if __name__ == "__main__":
    main()
