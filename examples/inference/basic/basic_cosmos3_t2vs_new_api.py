import os
import time

# t2vs (text -> video + sound). The Cosmos3 denoise stage generates a joint
# [vision | sound] latent and AVAE-decodes the sound to a waveform muxed into the
# mp4. The joint-sound path is gated on COSMOS3_T2VS (set here for the example).
os.environ.setdefault("COSMOS3_T2VS", "1")

from fastvideo import VideoGenerator  # noqa: E402
from fastvideo.api import (  # noqa: E402
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    SamplingConfig,
)

OUTPUT_PATH = "video_samples_cosmos3_t2vs"


def main():
    model_name = os.environ.get("COSMOS3_MODEL_PATH", "nvidia/Cosmos3-Nano")

    generator_config = GeneratorConfig(
        model_path=model_name,
        engine=EngineConfig(
            num_gpus=1,
            use_fsdp_inference=False,
            offload=OffloadConfig(text_encoder=True, pin_cpu_memory=True, dit=False, vae=False),
        ),
    )

    load_start = time.perf_counter()
    generator = VideoGenerator.from_config(generator_config)
    load_time = time.perf_counter() - load_start

    prompt = (
        "Ocean waves crash against a rocky shore at sunset, white foam spraying "
        "into the air as seagulls wheel overhead. Golden light, cinematic wide "
        "shot, the rhythmic roar of the surf."
    )
    request = GenerationRequest(
        prompt=prompt,
        sampling=SamplingConfig(
            num_frames=int(os.environ.get("COSMOS3_NUM_FRAMES", "189")),
            height=int(os.environ.get("COSMOS3_HEIGHT", "704")),
            width=int(os.environ.get("COSMOS3_WIDTH", "1280")),
            num_inference_steps=int(os.environ.get("COSMOS3_STEPS", "35")),
            guidance_scale=6.0,
            fps=24,
            seed=1024,
        ),
        output=OutputConfig(output_path=OUTPUT_PATH, save_video=True, return_frames=False),
    )

    start = time.perf_counter()
    result = generator.generate(request)
    gen_time = time.perf_counter() - start

    print(f"Time taken to load model: {load_time} seconds")
    print(f"Time taken to generate video+sound: {gen_time} seconds")
    print(f"Output written to: {result.video_path}")


if __name__ == "__main__":
    main()
