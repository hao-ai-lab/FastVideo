"""
GEN3C: 3D-aware camera-controlled video generation.

This example generates a video from a single input image with camera control.
The pipeline uses MoGe depth estimation, 3D point cloud forward warping,
and the GEN3C diffusion model.

Requirements:
  1. Install MoGe: pip install moge-pytorch
  2. Download and convert weights:
     huggingface-cli download nvidia/GEN3C-Cosmos-7B --local-dir official_weights/GEN3C-Cosmos-7B
     python scripts/checkpoint_conversion/convert_gen3c_to_fastvideo.py \
       --source ./official_weights/GEN3C-Cosmos-7B/model.pt \
       --output ./converted_weights/GEN3C-Cosmos-7B \
       --components-source nvidia/Cosmos-Predict2-2B-Video2World
  3. Provide an input image for 3D-conditioned generation.
"""

import argparse

from fastvideo import VideoGenerator


def main():
    parser = argparse.ArgumentParser(description="GEN3C video generation")
    parser.add_argument("--model_path",
                        type=str,
                        default="converted_weights/GEN3C-Cosmos-7B")
    parser.add_argument("--image_path",
                        type=str,
                        default=None,
                        help="Input image for 3D cache conditioning")
    parser.add_argument("--prompt",
                        type=str,
                        default="A slow camera pan over a sunlit landscape.")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
            "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
            "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
            "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special "
            "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
            "flickering. Overall, the video is of poor quality."
        ),
    )
    parser.add_argument("--trajectory",
                        type=str,
                        default="left",
                        choices=[
                            "left", "right", "up", "down", "zoom_in",
                            "zoom_out", "clockwise", "counterclockwise", "none"
                        ])
    parser.add_argument("--movement_distance", type=float, default=0.3)
    parser.add_argument("--camera_rotation",
                        type=str,
                        default="center_facing",
                        choices=[
                            "center_facing", "no_rotation",
                            "trajectory_aligned"
                        ])
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--output_path",
                        type=str,
                        default="outputs_video/gen3c.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generator = VideoGenerator.from_pretrained(
        args.model_path,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    video = generator.generate_video(
        args.prompt,
        negative_prompt=args.negative_prompt,
        image_path=args.image_path,
        trajectory_type=args.trajectory,
        movement_distance=args.movement_distance,
        camera_rotation=args.camera_rotation,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        fps=24,
        seed=args.seed,
        output_path=args.output_path,
        save_video=True,
    )

    generator.shutdown()


if __name__ == "__main__":
    main()
