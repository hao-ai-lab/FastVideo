from fastvideo import VideoGenerator

OUTPUT_PATH = "skyreels_output"
def main():
    model_path = "/mnt/fast-disks/hao_lab/kaiqin/FastVideo/SkyReels-V2-I2V-1.3B-540P-Diffusers"

    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    image_path = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
    
    video = generator.generate_video(prompt, image_path=image_path, output_path=OUTPUT_PATH, save_video=True, height=832, width=480, num_frames=81)
    
    print("done")

if __name__ == "__main__":
    main()
