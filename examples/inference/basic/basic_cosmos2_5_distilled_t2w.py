"""Run the released Cosmos Predict2.5 2B distilled Text2World checkpoint."""

import argparse

from fastvideo import VideoGenerator
from fastvideo.api.sampling_param import SamplingParam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Converted FastVideo model directory")
    parser.add_argument("--output", default="outputs_video/cosmos2_5_distilled_t2w.mp4")
    parser.add_argument("--steps", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--frames", type=int, default=77)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--return-frames",
        action="store_true",
        help="Return decoded frames without writing an MP4 (DreamVerse contract smoke)",
    )
    args = parser.parse_args()

    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )
    sampling = SamplingParam(
        num_inference_steps=args.steps,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        seed=args.seed,
        guidance_scale=1.0,
    )
    prompt = (
        "A robotic arm performs precision welding in an industrial workshop. "
        "Bright blue-white sparks scatter over the metal while smoke rises, "
        "cinematic lighting, steady camera, realistic motion."
    )
    result = generator.generate_video(
        prompt,
        sampling_param=sampling,
        output_path=args.output,
        save_video=not args.return_frames,
        return_frames=args.return_frames,
    )
    if args.return_frames:
        frames = result.get("frames") if isinstance(result, dict) else None
        if not isinstance(frames, list) or not frames:
            raise RuntimeError("DreamVerse contract failed: generation did not return a nonempty frames list")
        first_shape = getattr(frames[0], "shape", None)
        print(f"COSMOS25_DREAMVERSE_FRAMES: PASS count={len(frames)} first_shape={first_shape}")
    generator.shutdown()


if __name__ == "__main__":
    main()
