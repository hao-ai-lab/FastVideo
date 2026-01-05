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
    prompt = ("Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.")

    # Use an example image path
    image_path = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"


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
