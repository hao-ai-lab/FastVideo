from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_lingbotworld_fast"


def main():
    generator = VideoGenerator.from_pretrained(
        "FastVideo/LingBot-World-Fast-Diffusers",
        num_gpus=1,
        use_fsdp_inference=False,  # set to True if GPU is out of memory
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    prompt = (
        "A serene lakeside scene with a lone tree standing in calm water, "
        "surrounded by distant snow-capped mountains under a bright blue sky "
        "with drifting white clouds — gentle ripples reflect the tree and sky, "
        "creating a tranquil, meditative atmosphere.")
    action_path = "examples/datasets/lingbotworld2"
    image_path = f"{action_path}/image.jpg"

    generator.generate_video(
        prompt,
        image_path=image_path,
        action_path=action_path,
        output_path=OUTPUT_PATH,
        save_video=True,
        num_frames=81,
        height=480,
        width=832,
    )


if __name__ == "__main__":
    main()
