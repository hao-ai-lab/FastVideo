from fastvideo import VideoGenerator

def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model. 
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastHunyuan-Diffusers",
        # if num_gpus > 1, FastVideo will automatically handle distributed setup
        num_gpus=4,
    )

    # Generate videos with the same simple API, regardless of GPU count
    prompt = "A beautiful woman in a red dress walking down a street"
    video = generator.generate_video(prompt)

    # Generate another video with a different prompt, without reloading the
    # model!
    prompt2 = "A beautiful woman in a blue dress walking down a street"
    video2 = generator.generate_video(prompt2)


if __name__ == "__main__":
    main()
