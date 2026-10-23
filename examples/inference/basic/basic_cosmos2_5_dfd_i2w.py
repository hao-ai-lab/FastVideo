"""Run the public four-step Cosmos Predict2.5 DFD Video2World student."""

import argparse

from fastvideo import VideoGenerator
from fastvideo.api.sampling_param import SamplingParam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Converted FastVideo DFD model directory")
    parser.add_argument("--image", required=True, help="Conditioning image")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs_video/cosmos2_5_dfd_i2w.mp4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--return-frames",
        action="store_true",
        help="Return decoded frames without writing an MP4",
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
        num_inference_steps=4,
        num_frames=81,
        height=704,
        width=1280,
        fps=24,
        seed=args.seed,
        guidance_scale=1.0,
    )
    result = generator.generate_video(
        args.prompt,
        sampling_param=sampling,
        image_path=args.image,
        num_cond_frames=1,
        output_path=args.output,
        save_video=not args.return_frames,
        return_frames=args.return_frames,
    )
    if args.return_frames:
        frames = result.get("frames") if isinstance(result, dict) else None
        if not isinstance(frames, list) or len(frames) != 81:
            raise RuntimeError("DFD frame-return contract failed: expected 81 decoded frames")
        print(f"COSMOS25_DFD_FRAMES: PASS count={len(frames)}")
    generator.shutdown()


if __name__ == "__main__":
    main()
