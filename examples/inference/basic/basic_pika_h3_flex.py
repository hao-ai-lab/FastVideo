# SPDX-License-Identifier: Apache-2.0
"""Generate synchronized video/audio with Pika H3 Flex.

Five task shapes share one model release:

``t2va``  text only
``i2va``  first image
``l2va``  last image
``fl2va`` first and last images
``ref2va`` ordered image/video/audio references

The six-forward ladder is part of the adapter contract; this script deliberately
does not expose a step-count flag.  ``--lossless-accel`` turns on the existing
training-free FastVideo switches (bit-identical kernels, reference memo,
parallel VAE tiles) before the generator starts.  Kernel/VSA work is separate.

Examples:
    python examples/inference/basic/basic_pika_h3_flex.py --task t2va \
        --prompt "a paper boat drifting down a rainy street"
    python examples/inference/basic/basic_pika_h3_flex.py --task fl2va \
        --image first.png --last-image last.png --prompt "the flower opens"
    python examples/inference/basic/basic_pika_h3_flex.py --task ref2va \
        --reference image:person.png --reference video:dance.mp4 \
        --reference audio:voice.wav --prompt "the person dances to the music"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from fastvideo import VideoGenerator
from fastvideo.api import (
    CompileConfig,
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

# Training-free switches FastVideo already exposes. Workers inherit the
# environment, so they must be set before VideoGenerator starts.
LOSSLESS_ACCEL_ENV = {
    "FASTVIDEO_MINIMAX_H3_EXACT_KERNELS": "all",
    "FASTVIDEO_H3_REF2VA_MEMO_ENTRIES": "16",
    "FASTVIDEO_H3_VAE_TILE_PARALLEL": "1",
    "FASTVIDEO_ULYSSES_A2A": "auto",
}

TASKS = ("t2va", "i2va", "l2va", "fl2va", "ref2va")


def _reference(value: str) -> MiniMaxH3Reference:
    try:
        media_type, source = value.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("A reference must be image:PATH, video:PATH, or audio:PATH.") from error
    if media_type not in ("image", "video", "audio"):
        raise argparse.ArgumentTypeError(f"Unknown reference type {media_type!r}.")
    return MiniMaxH3Reference(source=source, media_type=media_type)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="Pika-Labs/Pika-H3-Flex")
    parser.add_argument("--task", choices=TASKS, default="t2va")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", help="first frame (i2va/fl2va)")
    parser.add_argument("--last-image", help="last frame (l2va/fl2va)")
    parser.add_argument("--reference",
                        action="append",
                        type=_reference,
                        default=[],
                        help="ordered reference (repeatable; ref2va)")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="torch.compile the DiT")
    parser.add_argument("--lossless-accel", action="store_true", help="enable bit-identical speed-ups")
    parser.add_argument("--low-vram", action="store_true", help="stream the DiT layerwise")
    parser.add_argument("--output", default="outputs/pika_h3_flex")
    args = parser.parse_args()

    if args.task in ("i2va", "fl2va") and not args.image:
        parser.error(f"--task {args.task} needs --image")
    if args.task in ("l2va", "fl2va") and not args.last_image:
        parser.error(f"--task {args.task} needs --last-image")
    if args.task == "ref2va" and not args.reference:
        parser.error("--task ref2va needs at least one --reference")
    if args.task != "ref2va" and args.reference:
        parser.error("--reference is only valid with --task ref2va")
    return args


def _inputs(args: argparse.Namespace) -> InputConfig:
    if args.task == "ref2va":
        return InputConfig(references=args.reference)
    return InputConfig(image_path=args.image, last_image=args.last_image)


def _pipeline(args: argparse.Namespace) -> PipelineSelection:
    components = ComponentConfig()
    if args.task == "ref2va":
        components.override_pipeline_cls_name = "PikaH3FlexRef2VAModularPipeline"
    return PipelineSelection(
        workload_type="t2v" if args.task == "t2va" else "i2v",
        preset=f"pika_h3_flex_{args.task}",
        components=components,
        vae_tiling=True,
    )


def main() -> None:
    args = parse_args()
    if args.lossless_accel:
        for key, value in LOSSLESS_ACCEL_ENV.items():
            os.environ.setdefault(key, value)

    output_path = Path(args.output) / f"pika_h3_flex_{args.task}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    offload = OffloadConfig(
        dit=False,
        dit_layerwise=args.low_vram,
        text_encoder=True,
        vae=True,
        pin_cpu_memory=not args.low_vram,
        lazy_module_load=True,
    )
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=args.model_path,
            engine=EngineConfig(
                num_gpus=args.num_gpus,
                use_fsdp_inference=args.num_gpus > 1,
                parallelism=ParallelismConfig(tp_size=1, sp_size=args.num_gpus),
                offload=offload,
                compile=CompileConfig(enabled=args.compile),
            ),
            pipeline=_pipeline(args),
        ))
    try:
        result = generator.generate(
            GenerationRequest(
                prompt=args.prompt,
                negative_prompt="",
                inputs=_inputs(args),
                sampling=SamplingConfig(
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    fps=24,
                    num_inference_steps=6,
                    guidance_scale=1.0,
                    batch_cfg=False,
                    seed=args.seed,
                ),
                output=OutputConfig(
                    output_path=str(output_path),
                    save_video=True,
                    return_frames=False,
                ),
            ))
        print(f"Output written to: {result.video_path}")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
