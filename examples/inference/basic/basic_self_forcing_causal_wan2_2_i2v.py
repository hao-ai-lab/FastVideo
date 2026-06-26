# NOTE: This is still a work in progress, and the checkpoints are not released yet.

from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    InputConfig,
    OffloadConfig,
    OutputConfig,
    PipelineSelection,
    SamplingConfig,
)
import json

OUTPUT_PATH = "video_samples_self_forcing_causal_wan2_2_14B_i2v"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path="FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers",
            # FastVideo will automatically handle distributed setup
            engine=EngineConfig(
                num_gpus=1,
                use_fsdp_inference=False,  # set to True if GPU is out of memory
                offload=OffloadConfig(
                    dit=True,  # DiT need to be offloaded for MoE
                    vae=False,
                    text_encoder=True,
                    # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
                    pin_cpu_memory=True,
                    # image_encoder=False,
                ),
            ),
            pipeline=PipelineSelection(
                experimental={
                    "dit_precision": "fp32",
                    "dmd_denoising_steps": [1000, 850, 700, 550, 350, 275, 200, 125],
                },
            ),
        )
    )

    sampling = SamplingConfig(
        num_frames=81,
        width=832,
        height=480,
        seed=1000,
    )

    with open("assets/prompts/mixkit_i2v.jsonl", "r") as f:
        prompt_image_pairs = json.load(f)

    for prompt_image_pair in prompt_image_pairs:
        prompt = prompt_image_pair["prompt"]
        image_path = prompt_image_pair["image_path"]
        _ = generator.generate(
            GenerationRequest(
                prompt=prompt,
                inputs=InputConfig(image_path=image_path),
                sampling=sampling,
                output=OutputConfig(output_path=OUTPUT_PATH, save_video=True),
            )
        )


if __name__ == "__main__":
    main()
