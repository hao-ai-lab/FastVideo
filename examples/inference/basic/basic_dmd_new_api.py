import os
import time

from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    PipelineSelection,
)

OUTPUT_PATH = "video_samples_dmd2_typed"


def main():
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"

    model_name = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
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
        # PR 2 still routes a few advanced inference knobs through the
        # compatibility bridge until they get first-class typed fields.
        pipeline=PipelineSelection(
            experimental={
                "VSA_sparsity": 0.8,
            },
        ),
    )

    load_start_time = time.perf_counter()
    generator = VideoGenerator.from_config(generator_config)
    load_end_time = time.perf_counter()
    load_time = load_end_time - load_start_time

    prompt = (
        "A neon-lit alley in futuristic Tokyo during a heavy rainstorm at night. "
        "The puddles reflect glowing signs in kanji, advertising ramen, karaoke, "
        "and VR arcades. A woman in a translucent raincoat walks briskly with an "
        "LED umbrella. Steam rises from a street food cart, and a cat darts "
        "across the screen. Raindrops are visible on the camera lens, creating "
        "a cinematic bokeh effect."
    )
    request = GenerationRequest(
        prompt=prompt,
        output=OutputConfig(
            output_path=OUTPUT_PATH,
            save_video=True,
            return_frames=False,
        ),
    )

    start_time = time.perf_counter()
    result = generator.generate(request)
    end_time = time.perf_counter()
    gen_time = end_time - start_time

    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently "
        "in the breeze, enhancing the lion's commanding presence. The tone is "
        "vibrant, embodying the raw energy of the wild. Low angle, steady "
        "tracking shot, cinematic."
    )
    request2 = GenerationRequest(
        prompt=prompt2,
        output=OutputConfig(
            output_path=OUTPUT_PATH,
            save_video=True,
            return_frames=False,
        ),
    )

    start_time = time.perf_counter()
    result2 = generator.generate(request2)
    end_time = time.perf_counter()
    gen_time2 = end_time - start_time

    print(f"Time taken to load model: {load_time} seconds")
    print(f"Time taken to generate video: {gen_time} seconds")
    print(f"First output written to: {result.video_path}")
    print(f"Time taken to generate video2: {gen_time2} seconds")
    print(f"Second output written to: {result2.video_path}")


if __name__ == "__main__":
    main()
