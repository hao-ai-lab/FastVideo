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

# NVIDIA Cosmos3-Nano omni world model — text-to-image (T2I) path through
# FastVideo's native Cosmos3 pipeline. T2I is the single-frame case
# (num_frames=1); the canonical Cosmos3 T2I resolution is 960x960 (the model's
# "720" bucket, UniPC flow_shift=10.0). Point COSMOS3_MODEL_PATH at a local
# diffusers checkpoint (e.g. ``official_weights/cosmos3``) to skip the download.
OUTPUT_PATH = "video_samples_cosmos3_t2i"


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
        "A photograph of a red panda sitting on a mossy log in a misty bamboo "
        "forest, soft golden morning light filtering through the leaves, shallow "
        "depth of field, crisp fur detail, serene atmosphere."
    )
    request = GenerationRequest(
        prompt=prompt,
        sampling=SamplingConfig(
            # T2I is single-frame; canonical Cosmos3 T2I is 960x960. Overridable
            # via env for quick smoke runs.
            num_frames=1,
            height=int(os.environ.get("COSMOS3_HEIGHT", "960")),
            width=int(os.environ.get("COSMOS3_WIDTH", "960")),
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
    print(f"Time taken to generate image: {gen_time} seconds")
    print(f"Output written to: {result.video_path}")


if __name__ == "__main__":
    main()
