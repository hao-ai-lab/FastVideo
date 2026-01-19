# SPDX-License-Identifier: Apache-2.0
from fastvideo import VideoGenerator


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

    # image2world example from official repo
    image_path = "cosmos-predict2.5/assets/base/bus_terminal.jpg"

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

    negative_prompt = (
        "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
        "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
        "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, "
        "low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, "
        "unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
        "Overall, the video is of poor quality."
    )

    generator.generate_video(
        prompt,
        negative_prompt=negative_prompt,
        image_path=str(image_path),
        height=704,
        width=1280,
        num_frames=93,
        num_inference_steps=35,
        guidance_scale=7.0,
        fps=24,
        seed=0,
        output_path="outputs_video/cosmos2_5_i2w.mp4",
        save_video=True,
    )

    generator.shutdown()


if __name__ == "__main__":
    main()


