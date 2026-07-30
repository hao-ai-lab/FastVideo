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
        "The video presents a soaring journey through a fantasy jungle. The "
        "wind whips past the rider's blue hands gripping the reins, causing "
        "the leather straps to vibrate. The ancient gothic castle approaches "
        "steadily, its stone details becoming clearer against the backdrop of "
        "floating islands and distant waterfalls.")
    image_path = ("https://raw.githubusercontent.com/Robbyant/lingbot-world/"
                  "main/examples/00/image.jpg")
    action_path = "examples/inference/basic/lingbotworld_examples/00"

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
