"""Generate a five-second Dense LingBot-Video clip with the official defaults."""

import argparse
from pathlib import Path

from fastvideo import VideoGenerator


def parse_args() -> argparse.Namespace:
    """Parse the converted checkpoint and output paths for the sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a converted Dense LingBot-Video checkpoint.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/lingbot-video/dense-t2v"),
        help="Directory for the generated video.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the converted Dense checkpoint and generate the default T2V sample."""
    args = parse_args()
    generator = VideoGenerator.from_pretrained(
        str(args.model_path),
        num_gpus=1,
        use_fsdp_inference=False,
        text_encoder_cpu_offload=True,
        vae_cpu_offload=False,
        pin_cpu_memory=True,
    )
    try:
        generator.generate({
            "prompt": "A red fox runs through fresh snow at sunrise.",
            "output": {
                "output_path": str(args.output_path),
                "save_video": True,
                "return_frames": False,
            },
        })
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
