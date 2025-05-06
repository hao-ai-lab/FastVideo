from fastvideo import VideoGenerator
# from fastvideo.v1.configs.sample import SamplingParam

def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model. 
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastHunyuan-diffusers",
        # if num_gpus > 1, FastVideo will automatically handle distributed setup
        num_gpus=4,
    )

    # sampling_param = SamplingParam.from_pretrained("/workspace/data/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    # sampling_param.num_frames = 45
    # sampling_param.image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    # Generate videos with the same simple API, regardless of GPU count
    prompt = "A beautiful woman in a red dress walking down a street"
    video = generator.generate_video(prompt)
    # video = generator.generate_video(prompt, sampling_param=sampling_param, output_path="wan_t2v_videos/")

    # Generate another video with a different prompt, without reloading the
    # model!
    prompt2 = "A beautiful woman in a blue dress walking down a street"
    video2 = generator.generate_video(prompt2)


if __name__ == "__main__":
    main()
