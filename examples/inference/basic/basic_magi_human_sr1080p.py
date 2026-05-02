# SPDX-License-Identifier: Apache-2.0
"""Run daVinci-MagiHuman SR-1080p text-to-AV in FastVideo.

Build the converted repo on large local storage, then symlink it into the
workspace:

    python scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py \
        --source GAIR/daVinci-MagiHuman \
        --subfolder base \
        --sr-source GAIR/daVinci-MagiHuman \
        --sr-subfolder 1080p_sr \
        --output /raid/william5lin_converted_weights/magi_human_sr_1080p \
        --cast-bf16
    ln -s /raid/william5lin_converted_weights/magi_human_sr_1080p \
        converted_weights/magi_human_sr_1080p
"""
from fastvideo import VideoGenerator
from fastvideo.pipelines.basic.magi_human.pipeline_configs import (
    MagiHumanSR1080pConfig,
)


PROMPT = (
    "A warm afternoon scene: a person sits on a park bench reading a book, "
    "surrounded by softly swaying trees."
)


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "converted_weights/magi_human_sr_1080p",
        num_gpus=1,
        override_pipeline_cls_name="MagiHumanSR1080pPipeline",
        pipeline_config=MagiHumanSR1080pConfig(),
    )
    generator.generate_video(
        prompt=PROMPT,
        output_path="outputs_video/magi_human_sr1080p/output_magi_human_sr1080p.mp4",
        save_video=True,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
