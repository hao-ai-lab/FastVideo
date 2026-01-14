from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_hy15_i2v"


def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model. This is the HunyuanVideo 1.5 I2V 480P model.
    generator = VideoGenerator.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
        # FastVideo will automatically handle distributed setup
        num_gpus=8,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        # Set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        pin_cpu_memory=True,
        image_encoder_cpu_offload=True,
    )

    prompt = (
        'A paved pathway leads towards a stone arch bridge spanning a calm body of water.  '
        'Lush green trees and foliage line the path and the far bank of the water. '
        'A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. '
        'The water reflects the surrounding greenery and the sky.  '
        'The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. '
        'The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  '
        'The overall composition emphasizes the peaceful and harmonious nature of the landscape.'
    )

    # Update these paths to match your setup (from run.sh)
    image_path = "/mnt/weka/home/hao.zhang/mhuo/FastVideo/assets/hyworld.png" 

    video = generator.generate_video(
        prompt,
        image_path=image_path,
        output_path=OUTPUT_PATH,
        save_video=True,
        negative_prompt="",
        num_frames=81,
        fps=24,
    )


if __name__ == "__main__":
    main()
