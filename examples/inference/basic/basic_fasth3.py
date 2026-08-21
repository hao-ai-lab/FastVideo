# SPDX-License-Identifier: Apache-2.0
"""Few-step video+audio generation with the DMD2-distilled MiniMax H3 preview.

FastVideo/FastH3-Preview-v0.1 is a 4-step distillation of
MiniMaxAI/MiniMax-H3 (data-free DMD2): it walks a 4-step grid on the release
sampler's shift-12 schedule instead of the base model's 50 steps, generating
synchronized video and audio in one pipeline call.

The student was trained with block-sparse video attention (VSA, 64-token
tiles) and its checkpoint carries the trained sparse-gate parameters
(``attn.to_gate_compress``), so this script always runs the VSA-H3 attention
backend. At the default ``--vsa-sparsity 0.0`` the attention math is exactly
dense (every tile is selected); raise the sparsity for additional speedup.
"""

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
    PipelineSelection,
    SamplingConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="FastVideo/FastH3-Preview-v0.1")
    # The HF repo is private while the MiniMax H3 Community License review
    # completes; until it flips public, pass --model-path with a local
    # snapshot of the release instead (e.g. the team export at
    # /mnt/lustre/vlm-wlsaidhi/fastvideo/exports/FastH3-Preview-v0.1).
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/fasth3")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    # 4 is the grid the student was distilled for; other step counts are
    # off-distribution.
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--vsa-sparsity",
                        type=float,
                        default=0.0,
                        help="Run-level VSA sparsity in [0, 1). 0.0 (default) selects every tile, which is "
                        "exactly dense attention; the student was trained at 0.9")
    # 64 is the trained contract: the student was TRAINED with 64-token
    # (4,4,4) tiles, and its to_gate_compress gates were learned against
    # pooling at that granularity — keep 64 unless you are ablating.
    parser.add_argument("--vsa-tile-size",
                        type=int,
                        choices=(64, 256),
                        default=64,
                        help="VSA-H3 tile size in tokens; 64 (default) is what the student was trained "
                        "with and runs the native Triton block-sparse path, 256 is the FA4-CuTe-capable "
                        "geometry for ablations")
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

    # Boot-time run configuration, folded into FastVideoArgs (the same route
    # examples/inference/basic/basic_minimax_h3_t2v.py uses for sparsity):
    # - attention_backend: the checkpoint carries trained to_gate_compress
    #   gates, which only exist under the VSA-H3 backend — a dense-backend
    #   load would reject them as unexpected weights. Layers that do not
    #   support VSA-H3 (e.g. the token refiner) fall back to flash attention.
    # - VSA_tile_size: forwarded even at sparsity 0.0 because the gate-compress
    #   branch pools per tile, and the gates were trained at 64 tokens/tile.
    experimental: dict[str, object] = {
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "VSA_tile_size": args.vsa_tile_size,
    }
    if args.vsa_sparsity > 0.0:
        experimental["VSA_sparsity"] = args.vsa_sparsity

    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=args.model_path,
            pipeline=PipelineSelection(experimental=experimental),
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
                # the base model is guidance-distilled; the student inherits it
                guidance_scale=1.0,
                batch_cfg=False,
                seed=args.seed,
            ),
            output=OutputConfig(
                output_path=str(output_dir / "fasth3.mp4"),
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
