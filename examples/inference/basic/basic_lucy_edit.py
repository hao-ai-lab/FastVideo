from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig, GenerationRequest, GeneratorConfig, InputConfig,
    OffloadConfig, OutputConfig, SamplingConfig,
)

OUTPUT_PATH = "video_samples_lucy_edit"


def main():
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path="decart-ai/Lucy-Edit-Dev",
            engine=EngineConfig(
                num_gpus=1,
                use_fsdp_inference=False,
                offload=OffloadConfig(
                    dit=True,
                    vae=False,
                    text_encoder=True,
                    pin_cpu_memory=True,
                ),
            ),
        ))

    prompt = ("Change the apron and blouse to a classic clown costume: satin "
              "polka-dot jumpsuit in bright primary colors, ruffled white collar, "
              "oversized pom-pom buttons, white gloves, oversized red shoes, red "
              "foam nose; soft window light from left, eye-level medium shot.")
    video_path = "https://d2drjpuinn46lb.cloudfront.net/painter_original_edit.mp4"

    generator.generate(
        GenerationRequest(
            prompt=prompt,
            negative_prompt="",
            inputs=InputConfig(video_path=video_path),
            sampling=SamplingConfig(
                height=480,
                width=832,
                num_frames=81,
                fps=24,
                guidance_scale=5.0,
            ),
            output=OutputConfig(output_path=OUTPUT_PATH, save_video=True),
        ))


if __name__ == "__main__":
    main()
