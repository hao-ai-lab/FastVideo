# from fastvideo import VideoGenerator

# # from fastvideo.v1.configs.sample import SamplingParam

# OUTPUT_PATH = "video_samples"
# def main():
#     # FastVideo will automatically use the optimal default arguments for the
#     # model.
#     # If a local path is provided, FastVideo will make a best effort
#     # attempt to identify the optimal arguments.
#     generator = VideoGenerator.from_pretrained(
#         "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
#         # FastVideo will automatically handle distributed setup
#         num_gpus=2,
#         use_fsdp_inference=True,
#         use_cpu_offload=False
#     )

#     # sampling_param = SamplingParam.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
#     # sampling_param.num_frames = 45
#     # sampling_param.image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
#     # Generate videos with the same simple API, regardless of GPU count
#     prompt = (
#         "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
#         "wide with interest. The playful yet serene atmosphere is complemented by soft "
#         "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
#     )
#     video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True)
#     # video = generator.generate_video(prompt, sampling_param=sampling_param, output_path="wan_t2v_videos/")

#     # Generate another video with a different prompt, without reloading the
#     # model!
#     prompt2 = (
#         "A majestic lion strides across the golden savanna, its powerful frame "
#         "glistening under the warm afternoon sun. The tall grass ripples gently in "
#         "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
#         "embodying the raw energy of the wild. Low angle, steady tracking shot, "
#         "cinematic.")
#     video2 = generator.generate_video(prompt2, output_path=OUTPUT_PATH, save_video=True)


# if __name__ == "__main__":
#     main()

from fastvideo import VideoGenerator, PipelineConfig
from fastvideo.v1.configs.sample import SamplingParam

def main():
    config = PipelineConfig.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    config.text_encoder_precisions = ["fp16"]
    
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        pipeline_config=config,
        use_fsdp_inference=False,      # Disable FSDP for MPS
        use_cpu_offload=True,          
        text_encoder_offload=True,    
        pin_cpu_memory=True,           
        disable_autocast=False,        
        num_gpus=1,      
    )

    # Create sampling parameters with reduced number of frames
    sampling_param = SamplingParam.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    sampling_param.num_frames = 10  # Reduce from default 81 to 25 frames bc we have to use the SDPA attn backend for mps
    sampling_param.height = 256
    sampling_param.width = 256

    prompt = ("A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
             "wide with interest. The playful yet serene atmosphere is complemented by soft "
             "natural light filtering through the petals. Mid-shot, warm and cheerful tones.")
    
    video = generator.generate_video(prompt, sampling_param=sampling_param, enable_teacache=True)

if __name__ == "__main__":
    main()