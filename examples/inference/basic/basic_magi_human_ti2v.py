# SPDX-License-Identifier: Apache-2.0
"""Minimal daVinci-MagiHuman base text+image-to-AV example."""
from fastvideo import VideoGenerator
from fastvideo.pipelines.basic.magi_human.pipeline_configs import (
    MagiHumanBaseI2VConfig,
)


PROMPT = (
    "A cheerful saxophonist performs a short line with expressive facial "
    "motion, natural head movement, and synchronized audio in a small jazz club."
)
IMAGE_PATH = "assets/images/saxophonist.jpg"


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "FastVideo/MagiHuman-Diffusers/base",
        num_gpus=1,
        workload_type="i2v",
        override_pipeline_cls_name="MagiHumanI2VPipeline",
        pipeline_config=MagiHumanBaseI2VConfig(),
    )
    generator.generate_video(
        prompt=PROMPT,
        image_path=IMAGE_PATH,
        output_path="outputs_video/magi_human_ti2v/output_magi_human_ti2v.mp4",
        save_video=True,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
