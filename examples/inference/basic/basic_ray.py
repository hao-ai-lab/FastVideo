from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig, GenerationRequest, GeneratorConfig, OffloadConfig,
    OutputConfig,
)

OUTPUT_PATH = "video_samples"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            # FastVideo will automatically handle distributed setup
            engine=EngineConfig(
                num_gpus=2,
                use_fsdp_inference=True,
                execution_backend="ray",
                offload=OffloadConfig(
                    dit=False,
                    vae=False,
                    text_encoder=True,
                    pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
                    # image_encoder=False,
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
    video = generator.generate(
        GenerationRequest(prompt=prompt, output=OutputConfig(output_path=OUTPUT_PATH, save_video=True)))

    # Generate another video with a different prompt, without reloading the
    # model!
    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")
    video2 = generator.generate(
        GenerationRequest(prompt=prompt2, output=OutputConfig(output_path=OUTPUT_PATH, save_video=True)))


if __name__ == "__main__":
    main()
