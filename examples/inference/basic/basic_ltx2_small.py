import os
from fastvideo import VideoGenerator

PROMPT = (
    "A curious raccoon peers through a vibrant field of yellow sunflowers."
)

NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, artifacts, jpeg compression"
)


def main() -> None:
    diffusers_path = os.getenv("LTX2_DIFFUSERS_PATH",
                               "converted/ltx2_diffusers")

    generator = VideoGenerator.from_pretrained(
        diffusers_path,
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        ltx2_vae_tiling=True,
        ltx2_vae_spatial_tile_size_in_pixels=512,
        ltx2_vae_spatial_tile_overlap_in_pixels=64,
        ltx2_vae_temporal_tile_size_in_frames=64,
        ltx2_vae_temporal_tile_overlap_in_frames=24,
    )

    output_path = "outputs_video/ltx2_basic"
    # Use smaller resolution and fewer frames to fit in memory
    generator.generate_video(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        output_path=output_path,
        save_video=True,
        height=512,       # Reduced from 1024
        width=768,        # Reduced from 1536
        num_frames=41,    # Reduced from 121
    )
    generator.shutdown()


if __name__ == "__main__":
    main()

