import os
import argparse
from fastvideo import VideoGenerator, SamplingParam, PipelineConfig

def main(args):
    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "/home/bcds/model/Wan-AI/Wan2.1-T2V-14B-Diffusers",
        num_gpus=1,  # Adjust based on your hardware
        STA_mode="STA_searching"
    )

    # Define a prompt for your video
    prompt = args.prompt
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    prompts = [
        "A man is dancing.",
        "A man is doing yoga.",
        "A man is running.",
        "A man is walking.",
        "A woman is doing yoga",
        "A woman is running.",
        "A woman is walking.",
        "people are doing yoga.",
        "people are running.",
        "people are walking.",
    ]

    params = SamplingParam(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        fps=args.fps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        return_frames=True,  # Also return frames from this call (defaults to False)
        output_path=args.output_path,  # Controls where videos are saved
        save_video=True,
        negative_prompt=negative_prompt
    )

    # Generate the video
    for prompt in prompts:
        video = generator.generate_video(
            prompt,
            sampling_param=params,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A man is dancing.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=69)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output_path", type=str, default="my_videos/")
    args = parser.parse_args()
    main(args)