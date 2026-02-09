from fastvideo import VideoGenerator
from fastvideo.models.dits.lingbotworld.cam_utils import prepare_camera_embedding

# from fastvideo.configs.sample import SamplingParam
OUTPUT_PATH = "video_samples_lingbotworld"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
         "FastVideo/LingBot-World-Base-Cam-Diffusers",
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=False, # set to True if GPU is out of memory
        dit_cpu_offload=True, # DiT need to be offloaded for MoE
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
        pin_cpu_memory=True,
        # image_encoder_cpu_offload=False,
    )

    num_frames = 81
    prompt = "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls."
    image_path = "https://raw.githubusercontent.com/Robbyant/lingbot-world/main/examples/00/image.jpg"
    action_path = "examples/inference/basic/lingbotworld_examples/00"
    c2ws_plucker_emb, num_frames = prepare_camera_embedding(
        action_path=action_path,
        num_frames=num_frames,
        height=480,
        width=832,
        spatial_scale=8,
    )

    generator.generate_video(
        prompt,
        image_path=image_path,
        output_path=OUTPUT_PATH,
        save_video=True,
        num_frames=num_frames,
        height=480,
        width=832,
        c2ws_plucker_emb=c2ws_plucker_emb,
    )


if __name__ == "__main__":
    main()
