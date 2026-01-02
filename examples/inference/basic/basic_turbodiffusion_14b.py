import os

# Set SLA attention backend BEFORE fastvideo imports
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SLA_ATTN"

from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_turbodiffusion_14B"


def main() -> None:
    # TurboDiffusion 14B: 1-4 step video generation using RCM scheduler + SLA attention
    # FastVideo will automatically use TurboDiffusionPipeline when specified
    generator = VideoGenerator.from_pretrained(
        "loayrashid/TurboWan2.1-T2V-14B-Diffusers",
        # 14B model needs more GPUs
        num_gpus=2,
        # TurboDiffusion uses a custom pipeline with RCM scheduler
        override_pipeline_cls_name="TurboDiffusionPipeline",
    )

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    video = generator.generate_video(
        prompt,
        output_path=OUTPUT_PATH,
        save_video=True,
        num_inference_steps=4,
        seed=42,
        guidance_scale=1.0,
    )

    # Generate another video with a different prompt, without reloading the model!
    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic."
    )
    video2 = generator.generate_video(
        prompt2,
        output_path=OUTPUT_PATH,
        save_video=True,
        num_inference_steps=4,
        seed=42,
        guidance_scale=1.0,
    )


if __name__ == "__main__":
    main()
