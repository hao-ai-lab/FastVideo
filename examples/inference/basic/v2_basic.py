"""v2 port of basic.py — Wan2.1-T2V-1.3B through the v2 VideoGenerator.

Same convenience API as upstream (from_pretrained + generate_video); only delta is importing
VideoGenerator from v2. v2 bring-up: single-GPU, resident, SDPA; modest res/frames for a quick run.
"""
from v2 import VideoGenerator

OUTPUT_PATH = "v2_video_samples"


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
    )
    common = dict(output_path=OUTPUT_PATH, save_video=True,
                  num_frames=25, height=480, width=832, num_inference_steps=30, guidance_scale=5.0)

    prompt = ("A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with "
              "interest. The playful yet serene atmosphere is complemented by soft natural light "
              "filtering through the petals. Mid-shot, warm and cheerful tones.")
    video = generator.generate_video(prompt, output_video_name="wan21_raccoon", **common)

    prompt2 = ("A majestic lion strides across the golden savanna, its powerful frame glistening under "
               "the warm afternoon sun. Low angle, steady tracking shot, cinematic.")
    video2 = generator.generate_video(prompt2, output_video_name="wan21_lion", **common)
    print(f"Outputs: {video.video_path} , {video2.video_path}")


if __name__ == "__main__":
    main()
