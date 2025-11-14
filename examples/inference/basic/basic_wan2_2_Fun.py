from fastvideo import VideoGenerator

# from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples_wan2_1_Fun"
OUTPUT_NAME = "wan2.1_test"
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

    prompt = "一位年轻女性穿着一件粉色的连衣裙，裙子上有白色的装饰和粉色的纽扣。她的头发是紫色的，头上戴着一个红色的大蝴蝶结，显得非常可爱和精致。她还戴着一个红色的领结，整体造型充满了少女感和活力。她的表情温柔，双手轻轻交叉放在身前，姿态优雅。背景是简单的灰色，没有任何多余的装饰，使得人物更加突出。她的妆容清淡自然，突显了她的清新气质。整体画面给人一种甜美、梦幻的感觉，仿佛置身于童话世界中。"
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    # prompt                  = "A young woman with beautiful, clear eyes and blonde hair stands in the forest, wearing a white dress and a crown. Her expression is serene, reminiscent of a movie star, with fair and youthful skin. Her brown long hair flows in the wind. The video quality is very high, with a clear view. High quality, masterpiece, best quality, high resolution, ultra-fine, fantastical."
    # negative_prompt         = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
    image_path = "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/wan_fun/asset_Wan2_2/v1.0/8.png"
    control_video_path = "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/wan_fun/asset_Wan2_2/v1.0/pose.mp4"

    video = generator.generate_video(prompt, negative_prompt=negative_prompt, image_path=image_path, video_path=control_video_path, output_path=OUTPUT_PATH, output_video_name=OUTPUT_NAME, save_video=True)

if __name__ == "__main__":
    main()