# NOTE: This is still a work in progress, and the checkpoints are not released yet.

from fastvideo import VideoGenerator, SamplingParam
import json
# from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples_self_forcing_causal_wan2_2_14B_i2v"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        "FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers",
        # FastVideo will automatically handle distributed setup
        num_gpus=4,
        use_fsdp_inference=True,
        dit_cpu_offload=False, # DiT need to be offloaded for MoE
        dit_precision="fp32",
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        dmd_denoising_steps=[1000, 850, 700, 550, 350, 275, 200, 125],
        # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
        pin_cpu_memory=True,
        num_frame_per_block=4,
        # image_encoder_cpu_offload=False,
    )

    sampling_param = SamplingParam.from_pretrained("FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers")
    sampling_param.num_frames = 77
    sampling_param.width = 832
    sampling_param.height = 480
    sampling_param.seed = 1000

    with open("prompts/mixkit_i2v.jsonl", "r") as f:
        prompt_image_pairs = json.load(f)

    for prompt_image_pair in prompt_image_pairs:
        prompt = prompt_image_pair["prompt"]
        image_path = prompt_image_pair["image_path"]
        _ = generator.generate_video(prompt, image_path=image_path, output_path=OUTPUT_PATH, save_video=True, sampling_param=sampling_param)


if __name__ == "__main__":
    main()
