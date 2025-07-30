from fastvideo import VideoGenerator

from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        # FastVideo will automatically handle distributed setup
        num_gpus=2,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
    )

    sampling_param = SamplingParam.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    sampling_param.num_frames = 61
    sampling_param.num_inference_steps = 40
    sampling_param.guidance_scale = 5.0
    sampling_param.height = 448
    sampling_param.width = 832
    sampling_param.seed = 1024
    sampling_param.image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    # Generate videos with the same simple API, regardless of GPU count
    prompt = (
        "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
    )
    video = generator.generate_video(prompt, sampling_param=sampling_param, output_path=OUTPUT_PATH, save_video=True)


if __name__ == "__main__":
    main()
