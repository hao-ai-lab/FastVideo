from fastvideo import VideoGenerator

prompt = "A beautiful woman in a red dress walking down a street"
prompt2 = "A beautiful woman in a blue dress walking down a street"


def main():
    # This will automatically handle distributed setup if num_gpus > 1
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=2,
        num_inference_steps=2,
        height=480,
        width=832,
        num_frames=77,
        fps=16,
        guidance_scale=3.0,
    )

    # Generate videos with the same simple API, regardless of GPU count
    video = generator.generate_video(prompt)

    # video2 = generator.generate_video(prompt2)


if __name__ == "__main__":
    main()
