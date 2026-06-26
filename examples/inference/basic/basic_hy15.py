from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    SamplingConfig,
)
import json

OUTPUT_PATH = "video_samples_hy15"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_config(GeneratorConfig(
        model_path="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        # FastVideo will automatically handle distributed setup
        engine=EngineConfig(
            num_gpus=1,
            use_fsdp_inference=False, # set to True if GPU is out of memory
            offload=OffloadConfig(
                dit=True,
                vae=True,
                text_encoder=True,
                pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
                # image_encoder=False,
            ),
        ),
    ))

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )

    generator.generate(GenerationRequest(
        prompt=prompt,
        negative_prompt="",
        sampling=SamplingConfig(num_frames=81, fps=16),
        output=OutputConfig(output_path=OUTPUT_PATH, save_video=True),
    ))

    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")

    generator.generate(GenerationRequest(
        prompt=prompt2,
        negative_prompt="",
        sampling=SamplingConfig(num_frames=81, fps=16),
        output=OutputConfig(output_path=OUTPUT_PATH, save_video=True),
    ))


if __name__ == "__main__":
    main()
