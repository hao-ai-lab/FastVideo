from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_lucy_edit"


def main():
    generator = VideoGenerator.from_pretrained(
        "decart-ai/Lucy-Edit-Dev",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    prompt = ("Change the apron and blouse to a classic clown costume: satin "
              "polka-dot jumpsuit in bright primary colors, ruffled white collar, "
              "oversized pom-pom buttons, white gloves, oversized red shoes, red "
              "foam nose; soft window light from left, eye-level medium shot.")
    video_path = "https://d2drjpuinn46lb.cloudfront.net/painter_original_edit.mp4"

    generator.generate_video(
        prompt,
        negative_prompt="",
        video_path=video_path,
        output_path=OUTPUT_PATH,
        save_video=True,
        height=480,
        width=832,
        num_frames=81,
        fps=24,
        guidance_scale=5.0,
    )


if __name__ == "__main__":
    main()
