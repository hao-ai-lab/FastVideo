from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig, GenerationRequest, GeneratorConfig, OffloadConfig,
    OutputConfig, SamplingConfig,
)

OUTPUT_PATH = "video_samples_wan2_2_14B_t2v"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            engine=EngineConfig(
                # FastVideo will automatically handle distributed setup
                num_gpus=2,
                use_fsdp_inference=False,  # set to True if GPU is out of memory
                offload=OffloadConfig(
                    dit=True,  # DiT need to be offloaded for MoE
                    vae=False,
                    text_encoder=True,
                    # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
                    pin_cpu_memory=True,
                ),
            ),
        )
    )

    # Generate videos with the same simple API, regardless of GPU count
    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    _ = generator.generate(
        GenerationRequest(
            prompt=prompt,
            sampling=SamplingConfig(height=720, width=1280, num_frames=81),
            output=OutputConfig(output_path=OUTPUT_PATH, save_video=True),
        )
    )

    # Generate another video with a different prompt, without reloading the
    # model!
    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")
    _ = generator.generate(
        GenerationRequest(
            prompt=prompt2,
            sampling=SamplingConfig(height=720, width=1280, num_frames=81),
            output=OutputConfig(output_path=OUTPUT_PATH, save_video=True),
        )
    )


if __name__ == "__main__":
    main()
