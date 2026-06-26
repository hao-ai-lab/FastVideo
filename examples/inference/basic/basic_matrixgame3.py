from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig, GenerationRequest, GeneratorConfig, InputConfig, OffloadConfig, OutputConfig, SamplingConfig,
)

MODEL_PATH = "FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers"
IMAGE_URL = "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-3/demo_images/001/image.png"
PROMPT = "A colorful, animated cityscape with a gas station and various buildings."
OUTPUT_PATH = "video_samples_matrixgame3"


def main():
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=MODEL_PATH,
            engine=EngineConfig(
                num_gpus=1,
                use_fsdp_inference=False,
                offload=OffloadConfig(
                    dit=False,
                    vae=False,
                    text_encoder=True,
                    pin_cpu_memory=True,
                ),
            ),
        ))

    generator.generate(
        GenerationRequest(
            prompt=PROMPT,
            inputs=InputConfig(image_path=IMAGE_URL),
            sampling=SamplingConfig(
                height=720,
                width=1280,
                num_frames=57,
                num_inference_steps=3,
                guidance_scale=1.0,
                seed=42,
            ),
            output=OutputConfig(
                output_path=OUTPUT_PATH,
                save_video=True,
            ),
        ))


if __name__ == "__main__":
    main()
