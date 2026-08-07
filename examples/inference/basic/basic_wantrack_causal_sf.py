# SPDX-License-Identifier: Apache-2.0
"""Run causal WanTrack Self-Forcing I2V through FastVideo.
"""
import argparse
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    InputConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)
from fastvideo.models.dits.trackwan.utils import load_tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run causal WanTrack Self-Forcing I2V.")
    parser.add_argument(
        "--model-path",
        default="path/to/ckpt",
        help="Local diffusers-format Track-v0 weights directory (or HF id).",
    )
    parser.add_argument(
        "--image",
        default="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG",
        help="Condition image path or URL.",
    )
    parser.add_argument(
        "--tracks",
        default=None,
        help=("Track package: .pt/.npz dict with track_points+track_visibility "
              "(optional track_ids), or a directory containing those files."),
    )
    parser.add_argument(
        "--track-points",
        default=None,
        help="Optional standalone track_points tensor file (.pt/.npy/.npz).",
    )
    parser.add_argument(
        "--track-visibility",
        default=None,
        help="Optional standalone track_visibility tensor file (.pt/.npy/.npz).",
    )
    parser.add_argument(
        "--track-ids",
        default=None,
        help="Optional standalone track_ids tensor file (.pt/.npy/.npz).",
    )
    parser.add_argument(
        "--output",
        default="video_samples_wantrack_causal_sf/output.mp4",
        help="Output mp4 path.",
    )
    parser.add_argument(
        "--prompt",
        default=("Summer beach vacation style, a white cat wearing sunglasses "
                 "sits on a surfboard. The fluffy-furred feline gazes directly "
                 "at the camera with a relaxed expression."),
        help="Text prompt.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--num-tracks",
        type=int,
        default=8,
        help="Demo track count when no track files are provided.",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--sp-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tp_size = args.tp_size if args.tp_size is not None else (args.num_gpus if args.num_gpus > 1 else 1)
    sp_size = args.sp_size if args.sp_size is not None else (1 if args.num_gpus > 1 else args.num_gpus)

    tracks = load_tracks(
        tracks_path=args.tracks,
        track_points_path=args.track_points,
        track_visibility_path=args.track_visibility,
        track_ids_path=args.track_ids,
        num_frames=args.num_frames,
        num_tracks=args.num_tracks,
        seed=args.seed,
    )

    generator_config = GeneratorConfig(
        model_path=args.model_path,
        engine=EngineConfig(
            num_gpus=args.num_gpus,
            use_fsdp_inference=False,
            parallelism=ParallelismConfig(tp_size=tp_size, sp_size=sp_size),
            offload=OffloadConfig(
                dit=False,
                vae=False,
                text_encoder=True,
                image_encoder=True,
                pin_cpu_memory=True,
            ),
        ),
        pipeline=PipelineSelection(workload_type="i2v"),
    )

    generator = VideoGenerator.from_config(generator_config)
    try:
        request = GenerationRequest(
            prompt=args.prompt,
            inputs=InputConfig(
                image_path=args.image,
                track_points=tracks["track_points"],
                track_visibility=tracks["track_visibility"],
                track_ids=tracks["track_ids"],
            ),
            sampling=SamplingConfig(
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=args.fps,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            ),
            output=OutputConfig(
                output_path=str(output.parent),
                output_video_name=output.stem,
                save_video=True,
                return_frames=False,
            ),
        )
        result = generator.generate(request)
        if isinstance(result, list):
            result = result[0]
        print(f"Saved video to {result.video_path}")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
