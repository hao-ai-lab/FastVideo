# SPDX-License-Identifier: Apache-2.0
"""Generate synchronized video/audio from ordered references with MiniMax H3."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.api import (
    ComponentConfig,
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
from fastvideo.pipelines.basic.minimax_h3 import MiniMaxH3Reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--reference-video", required=True)
    parser.add_argument("--reference-audio", help="Optional additional audio reference.")
    parser.add_argument("--output", default="outputs/minimax_h3_ref2va")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    references = [MiniMaxH3Reference(source=args.reference_video, media_type="video")]
    if args.reference_audio:
        references.append(MiniMaxH3Reference(source=args.reference_audio, media_type="audio"))
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
            ),
            pipeline=PipelineSelection(
                workload_type="i2v",
                components=ComponentConfig(override_pipeline_cls_name="MiniMaxH3Ref2VAModularPipeline"),
            ),
        ))
    try:
        result = generator.generate(
            GenerationRequest(
                prompt=args.prompt,
                negative_prompt="",
                inputs=InputConfig(references=references),
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
                    output_path=str(output_dir / "minimax_h3_ref2va.mp4"),
                    save_video=True,
                    return_frames=False,
                ),
            ))
        print(f"Output written to: {result.video_path}")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
