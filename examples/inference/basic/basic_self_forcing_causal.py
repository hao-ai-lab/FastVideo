import os
import time

from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
)

OUTPUT_PATH = "video_samples_causal"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    model_name = "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
    generator_config = GeneratorConfig(
        model_path=model_name,
        # FastVideo will automatically handle distributed setup
        engine=EngineConfig(
            num_gpus=1,
            use_fsdp_inference=False, # set to True if GPU is out of memory
            offload=OffloadConfig(
                text_encoder=False,
                dit=False,
            ),
        ),
    )
    generator = VideoGenerator.from_config(generator_config)

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    request = GenerationRequest(
        prompt=prompt,
        output=OutputConfig(
            output_path=OUTPUT_PATH,
            save_video=True,
        ),
    )
    video = generator.generate(request)

if __name__ == "__main__":
    main()
