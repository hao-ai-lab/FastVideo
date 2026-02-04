from fastvideo import VideoGenerator
from fastvideo.configs.sample import SamplingParam


def main():
    # Point this to your local diffusers model dir (or replace with a HF model ID).
    model_path = "KyleShao/Cosmos-Predict2.5-2B-Diffusers"

    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        use_fsdp_inference=False,  # set True if GPU is out of memory
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    # Load default sampling parameters (negative_prompt, resolution, steps, etc.)
    sampling_param = SamplingParam.from_pretrained(model_path)

    prompt = (
        "A high-definition video captures the precision of robotic welding in an industrial setting. "
        "The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. "
        "The welding process is in full swing, with bright sparks and intense light illuminating the scene, "
        "creating a vivid display of blue and white hues. "
        "A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity. "
        "The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery, "
        "indicating a busy and functional industrial workspace. "
        "As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. "
        "The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. "
        "The metal surface beneath the torch shows ongoing signs of heating and melting. "
        "The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, "
        "underscoring the ongoing nature of the welding operation."
    )

    generator.generate_video(
        prompt,
        sampling_param=sampling_param,
        output_path="outputs_video/cosmos2_5_t2w.mp4",
        save_video=True,
    )

    generator.shutdown()


if __name__ == "__main__":
    main()



