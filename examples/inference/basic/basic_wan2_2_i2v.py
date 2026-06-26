from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig, GenerationRequest, GeneratorConfig, InputConfig,
    OffloadConfig, OutputConfig, SamplingConfig,
)

OUTPUT_PATH = "video_samples_wan2_2_14B_i2v"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            # FastVideo will automatically handle distributed setup
            engine=EngineConfig(
                num_gpus=1,
                use_fsdp_inference=False,  # set to True if GPU is out of memory
                offload=OffloadConfig(
                    dit=True,  # DiT need to be offloaded for MoE
                    vae=False,
                    text_encoder=True,
                    # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
                    pin_cpu_memory=True,
                    # image_encoder=False,
                ),
            ),
        )
    )

    prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    image_path = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"

    video = generator.generate(
        GenerationRequest(
            prompt=prompt,
            inputs=InputConfig(image_path=image_path),
            sampling=SamplingConfig(height=832, width=480, num_frames=81),
            output=OutputConfig(output_path=OUTPUT_PATH, save_video=True),
        )
    )

if __name__ == "__main__":
    main()
