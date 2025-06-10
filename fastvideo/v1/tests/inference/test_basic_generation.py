import pytest
from fastvideo import VideoGenerator
import os
import glob

@pytest.mark.parametrize("use_fsdp_inference", [True, False])
@pytest.mark.parametrize("use_cpu_offload", [True, False])
@pytest.mark.parametrize("num_gpus", [1, 2])
def test_multiple_generations(use_fsdp_inference: bool, use_cpu_offload: bool, num_gpus: int):
    """Test that multiple video generations don't throw any errors"""
    output_path = "./test_output"
    if num_gpus == 1 and use_fsdp_inference:
        pytest.skip("FSDP inference is not needed with 1 GPU")
    try:
        generator = VideoGenerator.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            num_gpus=num_gpus,
            use_fsdp_inference=use_fsdp_inference,
            use_cpu_offload=use_cpu_offload
        )
        
        prompts = [
            "A cat playing with a ball",
            "A dog running in a park"
        ]
        
        for prompt in prompts:
            video = generator.generate_video(
                prompt,
                output_path=output_path,
                save_video=True,
                height=256,
                width=256,
                num_frames=16,
                guidance_scale=5.0,
                num_inference_steps=8,
                seed=42
            )
    except Exception as e:
        pytest.fail(f"Video generation failed with error: {str(e)}") 
    finally:
        # Clean up generated videos
        if os.path.exists(output_path):
            for file in glob.glob(os.path.join(output_path, "*.mp4")):
                os.remove(file)
            os.rmdir(output_path) 