from fastvideo import VideoGenerator, PipelineConfig, SamplingParam

# from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples_fp16"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    model = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    pipeline_config = PipelineConfig.from_pretrained(model)
    pipeline_config.text_encoder_precisions = ("bf16", )
    generator = VideoGenerator.from_pretrained(
        model,
        # if num_gpus > 1, FastVideo will automatically handle distributed setup
        pipeline_config=pipeline_config,
        use_fsdp_inference=False,      # Disable FSDP for MPS
        dit_cpu_offload=True,          
        text_encoder_cpu_offload=True,    
        pin_cpu_memory=True,           
        disable_autocast=False,        
        num_gpus=1, 
    )

    sampling_param = SamplingParam.from_pretrained(model)
    sampling_param.num_frames = 30
    # sampling_param.image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    # Generate videos with the same simple API, regardless of GPU count
    prompt = "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting."
    video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, sampling_param=sampling_param)
    # video = generator.generate_video(prompt, sampling_param=sampling_param, output_path="wan_t2v_videos/")

    # Generate another video with a different prompt, without reloading the
    # model!


if __name__ == "__main__":
    main()
