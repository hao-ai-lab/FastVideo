# SPDX-License-Identifier: Apache-2.0
"""Generate video and audio from text with MiniMax H3."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3")
    # Rank-reduced AdaLN checkpoint (-39% params, -23 GiB VRAM): pass
    #   --model-path noctuashap/MiniMax-H3-pruned-r16
    # (or a local dir produced by
    #   scripts/checkpoint_conversion/convert_minimax_h3_adaln_rank.py).
    # adaln_rank is read from the checkpoint config; no other flags needed.
    # Rank-reduced checkpoints are inference-only: training needs the
    # full-rank release.
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/minimax_h3_t2v")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--torch-compile", action="store_true", help="torch.compile the DiT transformer path")
    parser.add_argument("--compile-mode",
                        default=None,
                        help='torch.compile mode, e.g. "reduce-overhead" for CUDA graphs')
    parser.add_argument("--repeats",
                        type=int,
                        default=1,
                        help="generate N times; with --torch-compile the first run pays "
                        "compilation, so steady-state is the last repeat")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=args.model_path,
            engine=EngineConfig(
                num_gpus=args.num_gpus,
                use_fsdp_inference=args.num_gpus > 1,
                parallelism=ParallelismConfig(tp_size=1, sp_size=args.num_gpus),
                offload=OffloadConfig(
                    dit=False,
                    dit_layerwise=False,
                    text_encoder=True,
                    vae=True,
                    pin_cpu_memory=False,
                ),
                compile=CompileConfig(
                    enabled=args.torch_compile,
                    mode=args.compile_mode,
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
                num_inference_steps=args.steps,
                guidance_scale=1.0,
                batch_cfg=False,
                seed=args.seed,
            ),
            output=OutputConfig(
                output_path=str(output_dir / "minimax_h3_t2v.mp4"),
                save_video=True,
                return_frames=False,
            ),
        )
        result = generator.generate(request)
        print(f"Output written to: {result.video_path}")
        if result.generation_time is not None:
            # machine-readable: benchmark harnesses parse this line to separate
            # generation from model-load time (last occurrence = steady state)
            print(f"Generation time: {result.generation_time:.2f}s")
        for _ in range(args.repeats - 1):
            result = generator.generate(request)
            if result.generation_time is not None:
                print(f"Generation time: {result.generation_time:.2f}s")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
