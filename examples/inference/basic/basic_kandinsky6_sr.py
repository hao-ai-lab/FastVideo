# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 video super-resolution: upscale a low-resolution video x2, x2.25 or x4.

The request needs no prompt and no height / width / num_frames: the geometry and frame rate follow the input video
(the first 121 frames, 5 s at 24 fps, are processed and the source audio is kept).  Use
``kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers`` for the 2-step distilled model.
See docs/inference/kandinsky6_sr.md.
"""
import argparse

from fastvideo import VideoGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers")
    parser.add_argument("--video-path", required=True, help="Low-resolution source video")
    parser.add_argument("--output-path", default="outputs_video/kandinsky6_sr",
                        help="Output directory (the file is named after the input) or .mp4 path")
    parser.add_argument("--scale", type=float, default=2.25, choices=[2.0, 4.0, 2.25], help="Total upscale factor")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Denoising steps per tile (default: the checkpoint's preset, 4 or 2)")
    parser.add_argument("--tiles-batch-size", type=int, default=1, help="Tiles denoised per DiT call")
    parser.add_argument("--target-resolution", default=None, help="Optional final size: hd, fullhd, 2k or WxH")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generator = VideoGenerator.from_pretrained(args.model_path, num_gpus=1)
    sampling = {"seed": args.seed}
    if args.num_steps is not None:
        sampling["num_inference_steps"] = args.num_steps
    result = generator.generate({
        "inputs": {
            "video_path": args.video_path
        },
        "sampling": sampling,
        "output": {
            "output_path": args.output_path,
            "save_video": True,
            "return_frames": False,
        },
        "extensions": {
            "sr_resolution_scale": args.scale,
            "sr_tiles_batch_size": args.tiles_batch_size,
            "sr_target_resolution": args.target_resolution,
        },
    })
    print(f"Wrote {result.video_path}")
    generator.shutdown()


if __name__ == "__main__":
    main()
