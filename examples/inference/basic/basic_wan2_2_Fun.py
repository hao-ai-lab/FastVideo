from fastvideo import VideoGenerator

# from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples_wan2_2_14B_i2v"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        "IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers",
        # "alibaba-pai/Wan2.2-Fun-A14B-Control",
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=True, # DiT need to be offloaded for MoE
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
        pin_cpu_memory=True,
        # image_encoder_cpu_offload=False,
    )

    prompt                  = "A young woman with beautiful, clear eyes and blonde hair stands in the forest, wearing a white dress and a crown. Her expression is serene, reminiscent of a movie star, with fair and youthful skin. Her brown long hair flows in the wind. The video quality is very high, with a clear view. High quality, masterpiece, best quality, high resolution, ultra-fine, fantastical."
    negative_prompt         = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
    image_path = "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/wan_fun/asset_Wan2_2/v1.0/pose.mp4"
    control_video_path = "https://huggingface.co/datasets/YiYiXu/testing-videos/resolve/main/wan_i2v_input.mp4"

    video = generator.generate_video(prompt, image_path=image_path, control_video=control_video_path, output_path=OUTPUT_PATH, save_video=True, height=832, width=480, num_frames=81)

if __name__ == "__main__":
    main()