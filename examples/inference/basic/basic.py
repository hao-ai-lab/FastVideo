from fastvideo import VideoGenerator

from fastvideo.configs.sample import SamplingParam

import os

os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"

OUTPUT_PATH = "video_samples"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    model_name = "FastVideo/FastWan2.1-T2V-14B-Diffusers"
    generator = VideoGenerator.from_pretrained(
        model_name,
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
        pin_cpu_memory=True,
        # image_encoder_cpu_offload=False,
    )

    sampling_param = SamplingParam.from_pretrained(model_name)
    sampling_param.image_path = "test.jpg"
    # sampling_param.num_inference_steps = 0
    # Generate videos with the same simple API, regardless of GPU count
    i2v_prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
    i2v_prompt = "A little girl is packing a suitcase and the contents starts flying out of the suitcase everywhere."
    prompt = i2v_prompt
    # prompt = (
    #     "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
    #     "wide with interest. The playful yet serene atmosphere is complemented by soft "
    #     "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    # )
    video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, sampling_param=sampling_param)
    # video = generator.generate_video(prompt, sampling_param=sampling_param, output_path="wan_t2v_videos/")
    return

    # Generate another video with a different prompt, without reloading the
    # model!
    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")
    video2 = generator.generate_video(prompt2, output_path=OUTPUT_PATH, save_video=True)


if __name__ == "__main__":
    main()
