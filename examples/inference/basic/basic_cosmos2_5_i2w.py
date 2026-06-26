# SPDX-License-Identifier: Apache-2.0
from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig, GenerationRequest, GeneratorConfig, InputConfig, OffloadConfig, OutputConfig,
)


def main():
    # Point this to your local diffusers model dir (or replace with a HF model ID).
    model_path = "KyleShao/Cosmos-Predict2.5-2B-Diffusers"

    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=model_path,
            engine=EngineConfig(
                num_gpus=1,
                use_fsdp_inference=False,  # set True if GPU is out of memory
                offload=OffloadConfig(
                    dit=False,
                    vae=False,
                    text_encoder=True,
                    pin_cpu_memory=True,
                ),
            ),
        )
    )

    # image2world example from official repo
    image_path = "assets/images/bus_terminal.jpg"

    prompt = (
        "A nighttime city bus terminal gradually shifts from stillness to subtle movement. "
        "At first, multiple double-decker buses are parked under the glow of overhead lights, "
        "with a central bus labeled '87D' facing forward and stationary. "
        "As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area "
        "and casting reflections onto adjacent vehicles. "
        "The motion creates space in the lineup, signaling activity within the otherwise quiet station. "
        "It then comes to a smooth stop, resuming its position in line. "
        "Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."
    )

    generator.generate(
        GenerationRequest(
            prompt=prompt,
            inputs=InputConfig(image_path=str(image_path)),
            output=OutputConfig(
                output_path="outputs_video/cosmos2_5_i2w.mp4",
                save_video=True,
            ),
            extensions={"num_cond_frames": 1},
        )
    )

    generator.shutdown()


if __name__ == "__main__":
    main()
