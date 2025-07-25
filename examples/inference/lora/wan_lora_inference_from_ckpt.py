"""
Inference using a LoRA checkpoint from FastVideo trainer.
"""
from fastvideo import VideoGenerator
from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "./lora_out"
def main():
    # Initialize VideoGenerator with the Wan model
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
        lora_path="checkpoints/wan_t2v_finetune_lora/checkpoint-1250/transformer",
        lora_nickname="crush_smol"
    )
    kwargs = {
        "height": 480,
        "width": 832,
        "num_frames": 77,
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "seed": 42,
    }
    # Generate video with LoRA style
    prompt = "A large metal cylinder is seen pressing down on a pile of colorful candies, flattening them as if they were under a hydraulic press. The candies are crushed and broken into small pieces, creating a mess on the table."

    video = generator.generate_video(
        prompt,
        output_path=OUTPUT_PATH,
        save_video=True,
        **kwargs
    )    

if __name__ == "__main__":
    main()