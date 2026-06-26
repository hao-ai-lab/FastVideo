from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_kandinsky5_2B_t2v"

IMAGE_PATH = "assets/girl.png"

def main():
    generator = VideoGenerator.from_pretrained(
        "kandinskylab/Kandinsky-5.0-I2V-Lite-sft-5s-Diffusers",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        # image_encoder_cpu_offload=False,
    )

    prompt = (
        "A woman sits at a wooden table by the window in a cozy café. She reaches out "
        "with her right hand, picks up the white coffee cup from the saucer, and gently "
        "brings it to her lips to take a sip. After drinking, she places the cup back on "
        "the table and looks out the window, enjoying the peaceful atmosphere."
    )
    _ = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True,height=512, width=768, num_frames=121)


if __name__ == "__main__":
    main()