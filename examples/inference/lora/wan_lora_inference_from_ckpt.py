"""
Inference using a LoRA checkpoint from FastVideo trainer.
"""
from fastvideo import VideoGenerator
from fastvideo.api import (ComponentConfig, EngineConfig, GenerationRequest,
                           GeneratorConfig, OffloadConfig, OutputConfig,
                           PipelineSelection, SamplingConfig)

OUTPUT_PATH = "./lora_out"
def main():
    # Initialize VideoGenerator with the Wan model
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            engine=EngineConfig(
                num_gpus=1,
                offload=OffloadConfig(
                    dit=False,
                    vae=True,
                    text_encoder=True,
                    pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
                ),
            ),
            pipeline=PipelineSelection(
                components=ComponentConfig(
                    lora_path="checkpoints/wan_t2v_finetune_lora/checkpoint-160/transformer",
                ),
                experimental={"lora_nickname": "crush_smol"},
            ),
        ))
    generator.unmerge_lora_weights()
    sampling = SamplingConfig(
        height=480,
        width=832,
        num_frames=77,
        guidance_scale=6.0,
        num_inference_steps=50,
        seed=42,
    )
    output = OutputConfig(
        output_path=OUTPUT_PATH,
        save_video=True,
    )
    # Generate video with LoRA style
    prompt = "A large metal cylinder is seen pressing down on a pile of Oreo cookies, flattening them as if they were under a hydraulic press."

    video = generator.generate(
        GenerationRequest(
            prompt=prompt,
            sampling=sampling,
            output=output,
        ))
    prompt = "A large metal cylinder is seen compressing colorful clay into a compact shape, demonstrating the power of a hydraulic press."
    video = generator.generate(
        GenerationRequest(
            prompt=prompt,
            sampling=sampling,
            output=output,
        ))
if __name__ == "__main__":
    main()
