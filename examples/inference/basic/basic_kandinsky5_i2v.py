from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_kandinsky5_i2v"

IMAGE_PATH = "assets/girl.png"


def main():
    generator = VideoGenerator.from_pretrained(
        "kandinskylab/Kandinsky-5.0-I2V-Pro-distilled-5s-Diffusers", 
        # "kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers"
        # "kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers"
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True,
        # image_encoder_cpu_offload=False,
    )

    prompt = (
        "A woman stands up and walks away"
    )
    _ = generator.generate_video(
        prompt,
        image_path=IMAGE_PATH,
        output_path=OUTPUT_PATH,
        save_video=True,
        height=1024,
        width=1024,
        num_frames=121,
    )


if __name__ == "__main__":
    main()
