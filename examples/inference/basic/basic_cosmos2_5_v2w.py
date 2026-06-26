# SPDX-License-Identifier: Apache-2.0
from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig, GenerationRequest, GeneratorConfig, InputConfig,
    OffloadConfig, OutputConfig,
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
        ))

    # video2world example from official repo
    video_path = "assets/videos/robot_pouring.mp4"

    prompt = (
        "A robotic arm, primarily white with black joints and cables, is shown in a clean, modern indoor setting with a white tabletop. "
        "The arm, equipped with a gripper holding a small, light green pitcher, is positioned above a clear glass containing a reddish-brown liquid and a spoon. "
        "The robotic arm is in the process of pouring a transparent liquid into the glass. "
        "To the left of the pitcher, there is an opened jar with a similar reddish-brown substance visible through its transparent body. "
        "In the background, a vase with white flowers and a brown couch are partially visible, adding to the contemporary ambiance. "
        "The lighting is bright, casting soft shadows on the table. "
        "The robotic arm's movements are smooth and controlled, demonstrating precision in its task. "
        "As the video progresses, the robotic arm completes the pour, leaving the glass half-filled with the reddish-brown liquid. "
        "The jar remains untouched throughout the sequence, and the spoon inside the glass remains stationary. "
        "The other robotic arm on the right side also stays stationary throughout the video. "
        "The final frame captures the robotic arm with the pitcher finishing the pour, with the glass now filled to a higher level, while the pitcher is slightly tilted but still held securely by the gripper."
    )

    generator.generate(
        GenerationRequest(
            prompt=prompt,
            inputs=InputConfig(video_path=str(video_path)),
            output=OutputConfig(
                output_path="outputs_video/cosmos2_5_v2w.mp4",
                save_video=True,
            ),
            extensions={"num_cond_frames": 1},
        ))

    generator.shutdown()


if __name__ == "__main__":
    main()
