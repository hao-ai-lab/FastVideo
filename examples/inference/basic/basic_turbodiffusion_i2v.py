import os

# Set SLA attention backend BEFORE fastvideo imports
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SLA_ATTN"

from fastvideo import VideoGenerator

# Use local model path
MODEL_PATH = "loayrashid/TurboWan2.2-I2V-A14B-Diffusers"
OUTPUT_PATH = "video_samples_turbodiffusion_i2v"


def main() -> None:
    # TurboDiffusion I2V: 1-4 step image-to-video generation
    # Uses RCM scheduler with sigma_max=200 for I2V
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=2,
        override_pipeline_cls_name="TurboDiffusionI2VPipeline",
    )

    # Example prompt and image for I2V
    prompt = (
        "POV selfie video, ultra-messy and extremely fast. A white cat in sunglasses stands on a surfboard with a neutral look when the board suddenly whips sideways, throwing cat and camera into the water; the frame dives sharply downward, swallowed by violent bursts of bubbles, spinning turbulence, and smeared water streaks as the camera sinks. Shadows thicken, pressure ripples distort the edges, and loose bubbles rush upward past the lens, showing the camera is still sinking. Then the cat kicks upward with explosive speed, dragging the view through churning bubbles and rapidly brightening water as sunlight floods back in; the camera races upward, water streaming off the lens, and finally breaks the surface in a sudden blast of light and spray, snapping back into a crooked, frantic selfie as the cat resurfaces."
    )

    # Use an example image path
    image_path = "/mnt/fast-disks/hao_lab/loay/TurboDiffusion/assets/i2v_inputs/i2v_input_0.jpg"

    video = generator.generate_video(
        prompt,
        image_path=image_path,
        output_path=OUTPUT_PATH,
        save_video=True,
        num_inference_steps=4,
        seed=42,
        guidance_scale=1.0,
    )


if __name__ == "__main__":
    main()
