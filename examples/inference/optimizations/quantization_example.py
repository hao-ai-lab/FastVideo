import torch
from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.layers.quantization.fp4_config import FP4Config
# from fastvideo.layers.quantization.fp8_config import FP8Config
from fastvideo.layers.linear import ReplicatedLinear

OUTPUT_PATH = "fp4_video_samples"
# OUTPUT_PATH = "video_samples"


def main():
    print("=== FP4 Quantization Video Generation Example ===")

    if not torch.cuda.is_available():
        print("Warning: CUDA not available. FP4 quantization requires GPU.")
        return
    
    gpu_capability = torch.cuda.get_device_capability()
    if gpu_capability[0] < 9:  # H100 and newer
        print(f"Warning: GPU capability {gpu_capability} may not support FP4. Recommended: 9.0+")
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Capability: {gpu_capability}")
    
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    # model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    pipeline_config = PipelineConfig.from_pretrained(model_id)
    pipeline_config.dit_precision = "bf16" 
    pipeline_config.dit_config.quant_config = FP4Config()
    # pipeline_config.dit_config.quant_config = FP8Config()

    ReplicatedLinear.print_shape_summary()
    
    print("\nLoading model with FP4 quantization...")
    generator = VideoGenerator.from_pretrained(
        model_id,
        pipeline_config=pipeline_config,
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=False,  
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
    )
    
    print("FP4 configuration applied. Generating videos...")
    
    print("\n=== Generating Video with FP4 Quantization ===")
    
    prompt1 = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    
    print(f"Prompt: {prompt1}")
    print("Generating video...")
    
    try:
        video1 = generator.generate_video(
            prompt1,
            output_path=OUTPUT_PATH,
            save_video=True,
        )
        print("✓ First video generated successfully with FP4 quantization!")
        
        # # Generate a second video to show the model can be reused
        # prompt2 = (
        #     "A majestic lion strides across the golden savanna, its powerful frame "
        #     "glistening under the warm afternoon sun. The tall grass ripples gently in "
        #     "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        #     "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        #     "cinematic."
        # )
        
        # print(f"\nGenerating second video...")
        # print(f"Prompt: {prompt2}")
        
        # video2 = generator.generate_video(
        #     prompt2,
        #     output_path=OUTPUT_PATH,
        #     save_video=True,
        # )
        # print("✓ Second video generated successfully with FP4 quantization!")
        
    except Exception as e:
        print(f"Error during video generation: {e}")
        print("Make sure you have the required deep_gemm library installed for FP4 support")
        return
    
    print(f"\n=== FP4 Quantization Summary ===")
    print(f"Videos saved to: {OUTPUT_PATH}")
    print("FP4 quantization benefits:")
    print("- ✓ Reduced memory usage compared to FP16/BF16")
    print("- ✓ Faster inference on supported hardware (H100+)")
    print("- ✓ Maintains model quality with proper scaling")
    print("\nFor optimal FP4 performance:")
    print("1. Use H100 or newer GPUs")
    print("2. Ensure deep_gemm library is installed")
    print("3. Keep models on GPU (avoid CPU offloading for quantized layers)")


if __name__ == "__main__":
    main() 