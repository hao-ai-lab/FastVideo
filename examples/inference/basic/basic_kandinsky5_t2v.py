from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_kandinsky5_t2v"

def main():
    generator = VideoGenerator.from_pretrained(
        "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers", 
        # "kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers"
        # "kandinskylab/Kandinsky-5.0-T2V-Lite-Distilled-5s-Diffusers"
        # "kandinskylab/Kandinsky-5.0-T2V-Pro-Distilled-5s-Diffusers"
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        # image_encoder_cpu_offload=False,
    )

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    _ = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True,height=512, width=768, num_frames=121)

    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")
    _ = generator.generate_video(prompt2, output_path=OUTPUT_PATH, save_video=True, height=512, width=768, num_frames=121)


if __name__ == "__main__":
    main()