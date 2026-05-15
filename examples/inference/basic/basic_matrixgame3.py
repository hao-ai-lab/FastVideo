from fastvideo import VideoGenerator

MODEL_PATH = "/home/hal-kaiqin/FastVideo_mg/models/Matrix-Game-3.0-Base-Distilled-Diffusers"
IMAGE_PATH = "/home/hal-kaiqin/FastVideo_mg/Matrix-Game/Matrix-Game-3/demo_images/001/image.png"
PROMPT = "A colorful, animated cityscape with a gas station and various buildings."
OUTPUT_PATH = "video_samples_matrixgame3"


def main():
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    generator.generate_video(
        prompt=PROMPT,
        image_path=IMAGE_PATH,
        height=720,
        width=1280,
        num_frames=57,
        num_inference_steps=50,
        guidance_scale=5.0,
        seed=42,
        output_path=OUTPUT_PATH,
        save_video=True,
    )


if __name__ == "__main__":
    main()
