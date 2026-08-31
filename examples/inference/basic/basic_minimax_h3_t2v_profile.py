# SPDX-License-Identifier: Apache-2.0
"""Profile compiled MiniMax H3 text-to-video-and-audio inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fastvideo import VideoGenerator
from fastvideo.api import (
    CompileConfig,
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    SamplingConfig,
)
from fastvideo.profiler import nvtx_range


def parse_args() -> argparse.Namespace:
    """Parse the fixed H3 profiling contract and its warmup count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--model-revision")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/minimax_h3_t2v_profile")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    # This value counts transformer evaluations; the scheduler also receives
    # one terminal sigma-grid point that does not invoke the transformer.
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, choices=(1, 4), default=1)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of identical generations before the single profiled generation.",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be nonnegative")
    return args


def main() -> None:
    """Warm the compiled H3 graphs, then emit one measured NVTX generation."""
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # FSDP-wrapped modules bypass pipeline compilation, so both supported
    # topologies replicate weights and use sequence parallelism for DiT work.
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=args.model_path,
            revision=args.model_revision,
            engine=EngineConfig(
                num_gpus=args.num_gpus,
                use_fsdp_inference=False,
                parallelism=ParallelismConfig(tp_size=1, sp_size=args.num_gpus),
                # Resident components keep warmup and measured execution on
                # the same devices without transfer work in the captured run.
                offload=OffloadConfig(
                    dit=False,
                    dit_layerwise=False,
                    text_encoder=False,
                    image_encoder=False,
                    vae=False,
                    pin_cpu_memory=False,
                ),
                # Compile the tiled video VAE through the component-owned
                # decoder and fixed-shape tile-helper boundaries from PR #1734.
                compile=CompileConfig(
                    enabled=True,
                    backend="inductor",
                    mode="default",
                    dynamic=False,
                    text_encoder_enabled=True,
                    vae_enabled=True,
                    audio_vae_enabled=True,
                    vae_kwargs={
                        "backend": "inductor",
                        "mode": "reduce-overhead",
                        "dynamic": False,
                    },
                ),
            ),
        ))
    try:
        request = GenerationRequest(
            prompt=args.prompt,
            negative_prompt="",
            sampling=SamplingConfig(
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=24,
                # MiniMax H3 consumes one fewer transformer step than sigma
                # grid points, so add the terminal point to execute five steps.
                num_inference_steps=args.steps + 1,
                guidance_scale=1.0,
                batch_cfg=False,
                seed=args.seed,
            ),
            output=OutputConfig(
                output_path=str(output_dir / "minimax_h3_t2v_profile.mp4"),
                save_video=True,
                return_frames=False,
            ),
        )

        # Every warmup reuses the measured request shape and seed so Inductor
        # builds the exact graphs that Nsight Systems observes afterward.
        for warmup_index in range(args.warmup_runs):
            warmup_result = generator.generate(request)
            print(
                f"Warmup {warmup_index + 1}/{args.warmup_runs} complete: "
                f"{warmup_result.video_path}"
            )

        # The CUDA profiler API bounds Nsight Systems collection while the
        # NVTX range supplies a named parent interval for the measured request.
        torch.cuda.profiler.start()
        try:
            with nvtx_range("minimax_h3.profiled_generation"):
                measured_result = generator.generate(request)
        finally:
            torch.cuda.profiler.stop()

        print(f"Output written to: {measured_result.video_path}")
        if measured_result.generation_time is not None:
            print(f"Generation time: {measured_result.generation_time:.2f}s")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
