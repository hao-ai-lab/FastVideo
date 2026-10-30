# SPDX-License-Identifier: Apache-2.0
"""Regenerate a 768P MiniMax-H3 draft video at 2560x1440 with Open-H3-Regenerate-2K.

The draft is VAE-encoded (the first 121 frames at 24 fps, 1344x768), its latent is upscaled, and three refine steps
of the H3 transformer with the released LoRA run at 2560x1440. The output video is silent. See
docs/inference/open_h3_regenerate_2k.md.

    python examples/inference/basic/basic_open_h3_regenerate_2k.py --video draft.mp4 --prompt "..." --output out/
"""

from __future__ import annotations

import argparse
import time
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
    PipelineSelection,
    QuantizationConfig,
    SamplingConfig,
)

DEFAULT_MODEL = "Efficient-Large-Model/Open-H3-Regenerate-2K"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default=DEFAULT_MODEL, help="release repo id or a local copy of it")
    parser.add_argument("--video", default=None, help="768P draft video (URL-free local path)")
    parser.add_argument("--prompt", required=True, help="text the refine steps are conditioned on")
    parser.add_argument("--output", type=Path, default=Path("outputs/open_h3_regenerate_2k"))
    parser.add_argument("--steps", type=int, choices=(2, 3), default=3, help="refine steps (3 is the released setting)")
    parser.add_argument("--seed", type=int, default=1766580932)
    parser.add_argument("--attention-backend", default="TORCH_SDPA")
    parser.add_argument("--draft-size", choices=("768p", "576p"), default="768p",
                        help="size the draft is encoded at; 576p for drafts from other video models")
    parser.add_argument("--refine-transformer", default=None,
                        help="refine on another MiniMax-H3 DiT (e.g. a FastH3 export) instead of the release's own")
    parser.add_argument("--refine-transformer-quant", default=None,
                        help="NVFP4 layer profile for a packed refine DiT, e.g. h3_dit_vsa")
    parser.add_argument("--draft-latent", default=None,
                        help="native draft latent .pt to refine instead of encoding --video")
    parser.add_argument("--runs", type=int, default=1, help="repeat the request to measure warm timings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.video is None) == (args.draft_latent is None):
        raise SystemExit("pass exactly one of --video or --draft-latent")
    args.output.mkdir(parents=True, exist_ok=True)
    engine = EngineConfig(num_gpus=1,
                          offload=OffloadConfig(dit=False, dit_layerwise=False, text_encoder=True, vae=False))
    if args.refine_transformer_quant:
        engine.quantization = QuantizationConfig(transformer_quant="NVFP4", layer_profile=args.refine_transformer_quant)
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=args.model_path,
            engine=engine,
            pipeline=PipelineSelection(
                components=ComponentConfig(),
                experimental={
                    "attention_backend": args.attention_backend,
                    "draft_size": args.draft_size,
                    "refine_transformer_dir": args.refine_transformer,
                    "draft_latent_path": args.draft_latent,
                },
            ),
        ))
    try:
        for run in range(args.runs):
            started = time.perf_counter()
            result = generator.generate(
                GenerationRequest(
                    prompt=args.prompt,
                    inputs=InputConfig(video_path=args.video),
                    sampling=SamplingConfig(height=1440,
                                            width=2560,
                                            num_frames=124,
                                            fps=24,
                                            num_inference_steps=args.steps,
                                            guidance_scale=1.0,
                                            seed=args.seed),
                    output=OutputConfig(output_path=str(args.output / f"regen2k_{run}.mp4"),
                                        save_video=True,
                                        return_frames=False),
                ))
            print(f"run {run}: {time.perf_counter() - started:.1f}s -> {result.video_path}")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
