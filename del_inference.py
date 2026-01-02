from fastvideo import VideoGenerator
import os
# os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SLA_ATTN"
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SAGE_SLA_ATTN"

def main() -> None:
    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=2,  # Adjust based on your hardware
    )

    # Define a prompt for your video
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."

    # Generate the video
    generator.generate_video(
        prompt,
        return_frames=
        True,  # Also return frames from this call (defaults to False)
        output_path="my_videos/",  # Controls where videos are saved
        save_video=True)


if __name__ == '__main__':
    main()
