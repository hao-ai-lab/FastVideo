import os

from fastvideo import VideoGenerator

# PAN2 is trained on the structured captions its prompt rewriter produces (serialized as one-line JSON). Plain
# prompts like this one work, with weaker results.
PROMPT = ("A red fox trots through fresh snow in a pine forest at dawn, its breath visible in the cold air. "
          "The camera tracks it from the side at ground level as low golden light filters through the trees.")

OUTPUT_PATH = "video_samples_pan2"
MODEL_PATH = os.environ.get("PAN2_MODEL_PATH", "IFM/PAN2")

# Set to a first frame (e.g. "assets/images/girl.png") for image-to-video; the same pipeline runs both modes.
IMAGE_PATH = None


def main():
    # The 29B DiT runs in bf16; with more GPUs, num_gpus enables Ulysses sequence parallelism (the 32 attention
    # heads must be divisible by it).
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=int(os.environ.get("PAN2_NUM_GPUS", "1")),
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    # Defaults come from the pan2 preset: 704x1248, 313 frames (13 s at 24 fps), 50 steps, guidance 3.0.
    generator.generate_video(
        PROMPT,
        image_path=IMAGE_PATH,
        output_path=OUTPUT_PATH,
        save_video=True,
    )


if __name__ == "__main__":
    main()
