# NOTE: This is still a work in progress, and the checkpoints are not released yet.

from fastvideo import VideoGenerator

# from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples_self_forcing_causal_wan2_2_14B_t2v"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        "rand0nmr/SFWan2.2-T2V-A14B-Diffusers",
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=False, # set to True if GPU is out of memory
        dit_cpu_offload=True, # DiT need to be offloaded for MoE
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        dmd_denoising_steps=[1000, 850, 700, 550, 350, 275, 200, 125],
        # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
        pin_cpu_memory=True,
        init_weights_from_safetensors="/mnt/sharefs/users/hao.zhang/wei/SFwan2.2_distill_self_forcing_release_cfg2/checkpoint-246_weight_only/generator_inference_transformer/",
        init_weights_from_safetensors_2="/mnt/sharefs/users/hao.zhang/wei/SFwan2.2_distill_self_forcing_release_cfg2/checkpoint-246_weight_only/generator_2_inference_transformer/",
        num_frame_per_block=7,
        # image_encoder_cpu_offload=False,
    )

    # sampling_param = SamplingParam.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    # sampling_param.num_frames = 45
    # sampling_param.image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    # Generate videos with the same simple API, regardless of GPU count
    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    _ = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, num_frames=81)


if __name__ == "__main__":
    main()