from fastvideo import VideoGenerator

prompt = "A beautiful woman in a red dress walking down a street"
prompt2 = "A beautiful woman in a blue dress walking down a street"


def main():
    # This will automatically handle distributed setup if num_gpus > 1
    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastHunyuan-Diffusers",
        num_gpus=4,
        num_inference_steps=2,
        distributed_executor_backend="mp",
    )

    # Generate videos with the same simple API, regardless of GPU count
    video = generator.generate_video(prompt)

    video2 = generator.generate_video(prompt2)


if __name__ == "__main__":
    main()
