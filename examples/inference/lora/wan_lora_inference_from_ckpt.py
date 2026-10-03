"""
Inference using a LoRA checkpoint from FastVideo trainer.
"""
from fastvideo import VideoGenerator
from fastvideo.api.sampling_param import SamplingParam

OUTPUT_PATH = "./lora_out"


def main():
    # Initialize VideoGenerator with the Wan model
    generator = VideoGenerator.from_config({
        "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "engine": {
            "num_gpus": 1,
            "offload": {
                "dit": False,
                "vae": True,
                "text_encoder": True,
                "pin_cpu_memory": True,  # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
            },
        },
        "pipeline": {
            "components": {
                "lora_path": "checkpoints/wan_t2v_finetune_lora/checkpoint-160/transformer",
                "lora_nickname": "crush_smol",
            },
        },
    })
    generator.unmerge_lora_weights()
    kwargs = {
        "height": 480,
        "width": 832,
        "num_frames": 77,
        "guidance_scale": 6.0,
        "num_inference_steps": 50,
        "seed": 42,
    }
    # Generate video with LoRA style
    prompt = "A large metal cylinder is seen pressing down on a pile of Oreo cookies, flattening them as if they were under a hydraulic press."

    video = generator.generate({
        "prompt": prompt,
        "sampling": kwargs,
        "output": {"output_path": OUTPUT_PATH, "save_video": True},
    })
    prompt = "A large metal cylinder is seen compressing colorful clay into a compact shape, demonstrating the power of a hydraulic press."
    video = generator.generate({
        "prompt": prompt,
        "sampling": kwargs,
        "output": {"output_path": OUTPUT_PATH, "save_video": True},
    })


if __name__ == "__main__":
    main()
