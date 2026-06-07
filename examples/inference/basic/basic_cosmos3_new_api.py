import os
import time

from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    SamplingConfig,
)

# NVIDIA Cosmos3-Nano omni world model — this example exercises the
# text-to-video (T2V) path through FastVideo's native Cosmos3 pipeline.
# Point COSMOS3_MODEL_PATH at a local diffusers checkpoint (e.g.
# ``official_weights/cosmos3``) to skip the Hugging Face download.
OUTPUT_PATH = "video_samples_cosmos3"


def main():
    model_name = os.environ.get("COSMOS3_MODEL_PATH", "nvidia/Cosmos3-Nano")

    generator_config = GeneratorConfig(
        model_path=model_name,
        engine=EngineConfig(
            num_gpus=1,
            use_fsdp_inference=False,
            offload=OffloadConfig(
                text_encoder=True,
                pin_cpu_memory=True,
                dit=False,
                vae=False,
            ),
        ),
    )

    load_start_time = time.perf_counter()
    generator = VideoGenerator.from_config(generator_config)
    load_time = time.perf_counter() - load_start_time

    prompt = (
        "A golden retriever puppy runs across a sunlit meadow toward the camera, "
        "ears flopping and wildflowers swaying in the breeze. Shallow depth of "
        "field, warm afternoon light, smooth cinematic tracking shot."
    )
    request = GenerationRequest(
        prompt=prompt,
        sampling=SamplingConfig(
            # cosmos3_nano native defaults are 704x1280, 189 frames, 35 steps;
            # overridable via env for quick smoke runs.
            num_frames=int(os.environ.get("COSMOS3_NUM_FRAMES", "189")),
            height=int(os.environ.get("COSMOS3_HEIGHT", "704")),
            width=int(os.environ.get("COSMOS3_WIDTH", "1280")),
            num_inference_steps=int(os.environ.get("COSMOS3_STEPS", "35")),
            guidance_scale=6.0,
            fps=24,
            seed=1024,
        ),
        output=OutputConfig(
            output_path=OUTPUT_PATH,
            save_video=True,
            return_frames=False,
        ),
    )

    start_time = time.perf_counter()
    result = generator.generate(request)
    gen_time = time.perf_counter() - start_time

    print(f"Time taken to load model: {load_time} seconds")
    print(f"Time taken to generate video: {gen_time} seconds")
    print(f"Output written to: {result.video_path}")


if __name__ == "__main__":
    main()
