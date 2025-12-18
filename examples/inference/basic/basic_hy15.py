from fastvideo import VideoGenerator
import json
# from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples_hy15"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        # image_encoder_cpu_offload=False,
    )

    with open("data/validation_64.json", "r") as f:
        data = json.load(f)["data"]
        for item in data[1:]:
            prompt = item["caption"]
            video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, negative_prompt="", num_frames=81, fps=16)


if __name__ == "__main__":
    main()
