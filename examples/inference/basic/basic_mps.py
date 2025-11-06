from fastvideo import VideoGenerator, PipelineConfig
from fastvideo.configs.sample import SamplingParam

def main():
    config = PipelineConfig.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    config.text_encoder_precisions = ["fp16"]
    
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        pipeline_config=config,
        use_fsdp_inference=False,      # Disable FSDP for MPS
        dit_cpu_offload=True,          
        text_encoder_cpu_offload=True,    
        pin_cpu_memory=True,           
        disable_autocast=False,        
        num_gpus=1,      
    )

    # Create sampling parameters with reduced number of frames
    sampling_param = SamplingParam.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    sampling_param.num_frames = 25  # Reduce from default 81 to 25 frames bc we have to use the SDPA attn backend for mps
    sampling_param.height = 256
    sampling_param.width = 256

    prompt = ("A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
             "wide with interest. The playful yet serene atmosphere is complemented by soft "
             "natural light filtering through the petals. Mid-shot, warm and cheerful tones.")
    
    video = generator.generate_video(prompt, sampling_param=sampling_param)

    prompt2 = ("A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")
    
    video2 = generator.generate_video(prompt2, sampling_param=sampling_param)

if __name__ == "__main__":
    main()
