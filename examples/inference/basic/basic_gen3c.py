from fastvideo import VideoGenerator


def main():
    # Point this to your local converted model dir.
    # To convert official weights:
    #   1. huggingface-cli download nvidia/GEN3C-Cosmos-7B --local-dir official_weights/GEN3C-Cosmos-7B
    #   2. python scripts/checkpoint_conversion/convert_gen3c_to_fastvideo.py \
    #        --source ./official_weights/GEN3C-Cosmos-7B/model.pt \
    #        --output ./converted_weights/GEN3C-Cosmos-7B
    model_path = "converted_weights/GEN3C-Cosmos-7B"

    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    prompt = (
        "A cinematic aerial shot slowly circling around an ancient stone castle "
        "perched on a cliff overlooking a misty valley at sunrise, with warm "
        "golden light illuminating the weathered walls and surrounding forest."
    )

    video = generator.generate_video(
        prompt,
        negative_prompt="",
        height=720,
        width=1280,
        num_frames=121,
        num_inference_steps=50,
        guidance_scale=6.0,
        fps=24,
        output_path="outputs_video/gen3c.mp4",
        save_video=True,
    )

    generator.shutdown()


if __name__ == "__main__":
    main()
