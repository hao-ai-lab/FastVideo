# SPDX-License-Identifier: Apache-2.0
"""Run a one-step interleaved generation trace through FastVideo.

This is intentionally small: it uses the single-prompt planner and an
accept-all critic from the app-level Interleave helper package.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.api.schema import (
    EngineConfig,
    GeneratorConfig,
    OffloadConfig,
    PipelineSelection,
)
from fastvideo.workflows.interleave_thinker import (
    AcceptAllCritic,
    FastVideoImageGeneratorBackend,
    InterleaveOrchestrator,
    SinglePromptPlanner,
    save_trace,
)


DEFAULT_PROMPT = "a brushed steel espresso machine on a marble counter, morning window light"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-step FastVideo interleave trace.")
    parser.add_argument(
        "--model-path",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="HF id or local diffusers-format image model directory.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Interleaved generation instruction.")
    parser.add_argument("--output-dir", default="outputs/interleave_single_prompt", help="Output directory.")
    parser.add_argument("--trace-path", default=None, help="Trace JSON path. Defaults under output-dir.")
    parser.add_argument("--seed", type=int, default=0, help="Generation seed.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--steps", type=int, default=4, help="Number of denoising steps.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument(
        "--backend",
        default=None,
        help="Set FASTVIDEO_ATTENTION_BACKEND, for example TORCH_SDPA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = args.backend

    output_dir = Path(args.output_dir)
    trace_path = Path(args.trace_path) if args.trace_path else output_dir / "trace.json"

    generator_config = GeneratorConfig(
        model_path=args.model_path,
        engine=EngineConfig(
            num_gpus=args.num_gpus,
            use_fsdp_inference=False,
            offload=OffloadConfig(
                dit=False,
                vae=True,
                text_encoder=True,
                pin_cpu_memory=False,
            ),
        ),
        pipeline=PipelineSelection(workload_type="t2i"),
    )

    generator = VideoGenerator.from_config(generator_config)
    try:
        backend = FastVideoImageGeneratorBackend(
            generator,
            output_dir=str(output_dir),
        )
        orchestrator = InterleaveOrchestrator(
            planner=SinglePromptPlanner(),
            generator=backend,
            critic=AcceptAllCritic(),
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
            seed=args.seed,
        )
        trace = orchestrator.run(args.prompt)
        save_trace(trace, trace_path)
        if trace.final_image is None or trace.final_image.file_path is None:
            raise RuntimeError("Interleave run completed without a final image path")
        print(f"Image: {trace.final_image.file_path}")
        print(f"Trace: {trace_path}")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
