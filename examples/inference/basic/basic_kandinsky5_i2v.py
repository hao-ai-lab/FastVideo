from fastvideo import VideoGenerator

from fastvideo.api import (EngineConfig, GenerationRequest, GeneratorConfig,
                           InputConfig, OffloadConfig, OutputConfig,
                           SamplingConfig)

OUTPUT_PATH = "video_samples_kandinsky5_i2v"

IMAGE_PATH = "assets/girl.png"


def main():
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path="kandinskylab/Kandinsky-5.0-I2V-Pro-distilled-5s-Diffusers",
            # "kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers"
            # "kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers"
            engine=EngineConfig(
                num_gpus=1,
                use_fsdp_inference=False,
                offload=OffloadConfig(
                    dit=False,
                    vae=False,
                    text_encoder=True,
                    pin_cpu_memory=True,
                    # image_encoder=False,
                ),
            ),
        ))

    prompt = (
        "A woman stands up and walks away"
    )
    _ = generator.generate(
        GenerationRequest(
            prompt=prompt,
            inputs=InputConfig(image_path=IMAGE_PATH),
            sampling=SamplingConfig(height=1024, width=1024, num_frames=121),
            output=OutputConfig(output_path=OUTPUT_PATH, save_video=True),
        ))


if __name__ == "__main__":
    main()
