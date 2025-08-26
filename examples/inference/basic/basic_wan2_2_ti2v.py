from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_wan2_2_5B_ti2v"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    model_name = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    generator = VideoGenerator.from_pretrained(
        model_name,
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        # image_encoder_cpu_offload=False,
    )

    # I2V is triggered just by passing in an image_path argument
    prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    image_path = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
    video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, image_path=image_path)

    # Generate another video with a different prompt, without reloading the
    # model!

    # T2V mode
    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")
    video2 = generator.generate_video(prompt2, output_path=OUTPUT_PATH, save_video=True)


if __name__ == "__main__":
    main()