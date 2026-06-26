from fastvideo import VideoGenerator, PipelineConfig
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    PipelineSelection,
    SamplingConfig,
)

def main():
    config = PipelineConfig.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    config.text_encoder_precisions = ["fp16"]

    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            engine=EngineConfig(
                num_gpus=1,
                use_fsdp_inference=False,      # Disable FSDP for MPS
                disable_autocast=False,
                offload=OffloadConfig(
                    dit=True,
                    text_encoder=True,
                    pin_cpu_memory=True,
                ),
            ),
            pipeline=PipelineSelection(
                experimental={"pipeline_config": config},
            ),
        )
    )

    # Reduce from default 81 to 25 frames bc we have to use the SDPA attn backend for mps
    sampling = SamplingConfig(
        num_frames=25,
        height=256,
        width=256,
    )

    prompt = ("A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
             "wide with interest. The playful yet serene atmosphere is complemented by soft "
             "natural light filtering through the petals. Mid-shot, warm and cheerful tones.")

    video = generator.generate(GenerationRequest(prompt=prompt, sampling=sampling))

    prompt2 = ("A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")

    video2 = generator.generate(GenerationRequest(prompt=prompt2, sampling=sampling))

if __name__ == "__main__":
    main()
