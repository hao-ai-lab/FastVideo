# SPDX-License-Identifier: Apache-2.0
"""Run daVinci-MagiHuman SR-1080p text+image-to-AV in FastVideo."""
from fastvideo import VideoGenerator
from fastvideo.pipelines.basic.magi_human.pipeline_configs import (
    MagiHumanSR1080pI2VConfig,
)


PROMPT = (
    "A cheerful saxophonist performs a short line with expressive facial "
    "motion, natural head movement, and synchronized audio in a small jazz club."
)
IMAGE_PATH = "assets/images/saxophonist.jpg"


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "FastVideo/MagiHuman-Diffusers/sr_1080p",
        num_gpus=1,
        workload_type="i2v",
        override_pipeline_cls_name="MagiHumanSR1080pI2VPipeline",
        pipeline_config=MagiHumanSR1080pI2VConfig(),
    )
    generator.generate_video(
        prompt=PROMPT,
        image_path=IMAGE_PATH,
        output_path="outputs_video/magi_human_sr1080p_ti2v/output_magi_human_sr1080p_ti2v.mp4",
        save_video=True,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
