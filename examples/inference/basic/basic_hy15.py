from fastvideo import VideoGenerator
import json
# from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples_hy15_step_distilled"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=False, # set to True if GPU is out of memory
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        # image_encoder_cpu_offload=False,
    )

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )

    video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, negative_prompt="")

    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")

    video2 = generator.generate_video(prompt2, output_path=OUTPUT_PATH, save_video=True, negative_prompt="")


if __name__ == "__main__":
    main()
