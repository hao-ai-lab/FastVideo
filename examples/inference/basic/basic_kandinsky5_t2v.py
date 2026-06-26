from fastvideo import VideoGenerator

from fastvideo.api import (EngineConfig, GenerationRequest, GeneratorConfig,
                           OffloadConfig, OutputConfig, SamplingConfig)

OUTPUT_PATH = "video_samples_kandinsky5_t2v"

def main():
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path="kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers",
            # "kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers"
            # "kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers"
            # "kandinskylab/Kandinsky-5.0-T2V-Pro-distilled-5s-Diffusers"
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
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    _ = generator.generate(
        GenerationRequest(
            prompt=prompt,
            sampling=SamplingConfig(height=512, width=768, num_frames=121),
            output=OutputConfig(output_path=OUTPUT_PATH, save_video=True),
        ))

    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")
    _ = generator.generate(
        GenerationRequest(
            prompt=prompt2,
            sampling=SamplingConfig(height=512, width=768, num_frames=121),
            output=OutputConfig(output_path=OUTPUT_PATH, save_video=True),
        ))


if __name__ == "__main__":
    main()
