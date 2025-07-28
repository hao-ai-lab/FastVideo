from fastvideo import VideoGenerator
import time
from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        # "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        # image_encoder_cpu_offload=False,
    )

    sampling_param = SamplingParam.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    sampling_param.num_inference_steps = 50

    # with open('Prompts.txt', 'r') as file:
    #     prompts = file.readlines()

    # for prompt in prompts:
    #     prompt = prompt.strip()
    #     video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, sampling_param=sampling_param)
        
    # sampling_param.num_frames = 45
    # sampling_param.image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    # Generate videos with the same simple API, regardless of GPU count
    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    start_time = time.perf_counter()
    video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True)
    end_time = time.perf_counter()
    gen_time = end_time - start_time
    # video = generator.generate_video(prompt, sampling_param=sampling_param, output_path="wan_t2v_videos/")

    # Generate another video with a different prompt, without reloading the
    # model!
    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")
    start_time = time.perf_counter()
    video2 = generator.generate_video(prompt2, output_path=OUTPUT_PATH, save_video=True)
    end_time = time.perf_counter()
    gen_time2 = end_time - start_time
    print(f"Time taken to generate video: {gen_time} seconds")
    print(f"Time taken to generate video: {gen_time2} seconds")


if __name__ == "__main__":
    main()
