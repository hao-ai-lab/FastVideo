# SPDX-License-Identifier: Apache-2.0
"""Run daVinci-MagiHuman SR-540p text-to-AV in FastVideo.

The converted repo must contain both ``transformer/`` (base DiT) and
``sr_transformer/`` (540p SR DiT). Build it with:

    python scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py \
        --source GAIR/daVinci-MagiHuman \
        --subfolder base \
        --sr-source GAIR/daVinci-MagiHuman \
        --sr-subfolder 540p_sr \
        --output converted_weights/magi_human_sr_540p
"""
from fastvideo import VideoGenerator


PROMPT = (
    "A warm afternoon scene: a person sits on a park bench reading a book, "
    "surrounded by softly swaying trees."
)


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "FastVideo/MagiHuman-Diffusers/sr_540p",
        num_gpus=1,
    )
    generator.generate_video(
        prompt=PROMPT,
        output_path="outputs_video/magi_human_sr540p/output_magi_human_sr540p.mp4",
        save_video=True,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
