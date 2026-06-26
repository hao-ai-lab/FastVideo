# NOTE: This is still a work in progress, and the checkpoints are not released yet.

from fastvideo import VideoGenerator
from fastvideo.api import (
    ComponentConfig, EngineConfig, GenerationRequest, GeneratorConfig,
    OffloadConfig, OutputConfig, PipelineSelection, SamplingConfig,
)

OUTPUT_PATH = "video_samples_self_forcing_causal_wan2_2_14B_t2v"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path="rand0nmr/SFWan2.2-T2V-A14B-Diffusers",
            engine=EngineConfig(
                # FastVideo will automatically handle distributed setup
                num_gpus=1,
                use_fsdp_inference=False,  # set to True if GPU is out of memory
                offload=OffloadConfig(
                    dit=True,  # DiT need to be offloaded for MoE
                    vae=False,
                    text_encoder=True,
                    # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
                    pin_cpu_memory=True,
                ),
            ),
            pipeline=PipelineSelection(
                components=ComponentConfig(
                    transformer_weights="/mnt/sharefs/users/hao.zhang/wei/SFwan2.2_distill_self_forcing_release_cfg2/checkpoint-246_weight_only/generator_inference_transformer/",
                    transformer_2_weights="/mnt/sharefs/users/hao.zhang/wei/SFwan2.2_distill_self_forcing_release_cfg2/checkpoint-246_weight_only/generator_2_inference_transformer/",
                ),
                experimental={
                    "dmd_denoising_steps": [1000, 850, 700, 550, 350, 275, 200, 125],
                    "num_frame_per_block": 7,
                },
            ),
            # image_encoder_cpu_offload=False,
        )
    )

    # Generate videos with the same simple API, regardless of GPU count
    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    _ = generator.generate(
        GenerationRequest(
            prompt=prompt,
            sampling=SamplingConfig(num_frames=81),
            output=OutputConfig(output_path=OUTPUT_PATH, save_video=True),
        )
    )


if __name__ == "__main__":
    main()
