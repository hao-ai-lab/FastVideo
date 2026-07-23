# SPDX-License-Identifier: Apache-2.0
"""MMAudio large-44k-v2 video-to-audio example."""

import argparse
import os

from fastvideo import VideoGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-path", default="outputs_audio/mmaudio.wav")
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="music")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = VideoGenerator.from_pretrained(
        os.environ.get(
            "MMAUDIO_MODEL_PATH",
            "converted_weights/mmaudio/large_44k_v2",
        ),
        workload_type="v2a",
        num_gpus=1,
    )
    result = generator.generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        video_path=args.video_path,
        audio_end_in_s=args.duration_seconds,
        output_path=args.output_path,
        save_video=True,
        return_frames=False,
    )
    print(result["video_path"])
    generator.shutdown()


if __name__ == "__main__":
    main()
