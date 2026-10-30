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
parallel VAE tiles) before the generator starts.  ``--nvfp4-decoder`` and
``--nvfp4-text-encoder`` opt into the existing Blackwell FP4 weight paths;
they change weights, not the six-forward sampler.  Kernel/VSA work is separate.

``--resolution`` (480p/768p), ``--aspect`` (one of Pika's eleven buckets), and
``--duration`` (5-15 s) pick the bucket and the ``17n + 5`` frame count;
``--height``/``--width``/``--num-frames`` override them with explicit values.
Launch with ``FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN FASTVIDEO_FA4=1`` on a
FlashAttention-4 host (Hopper/Blackwell); without FA4 attention falls back to
FA3/FA2 or, if neither is installed, to SDPA.

``--offload resident`` keeps every component on the GPU, which is how the
GB200 timings in docs/inference/pika_h3_flex.md were measured; the default
``lazy`` mode loads components on demand for smaller devices and is slower.

Examples:
    FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN FASTVIDEO_FA4=1 \
    python examples/inference/basic/basic_pika_h3_flex.py --task t2va --offload resident \
        --prompt "a paper boat drifting down a rainy street"
    python examples/inference/basic/basic_pika_h3_flex.py --task t2va \
        --resolution 480p --aspect 9:16 --duration 10 --prompt "a lighthouse at dusk"
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
from fastvideo.models.vision_utils import load_image
from fastvideo.pipelines.basic.minimax_h3 import MiniMaxH3Reference
from fastvideo.pipelines.basic.pika_h3_flex.buckets import (
    PIKA_ASPECTS,
    PIKA_FPS,
    PIKA_MAX_DURATION,
    PIKA_MIN_DURATION,
    canvas_for_aspect,
    resolve_frames,
)

# Training-free switches FastVideo already exposes. Workers inherit the
# environment, so they must be set before VideoGenerator starts.
LOSSLESS_ACCEL_ENV = {
    "FASTVIDEO_MINIMAX_H3_EXACT_KERNELS": "all",
    "FASTVIDEO_H3_REF2VA_MEMO_ENTRIES": "16",
    "FASTVIDEO_H3_VAE_TILE_PARALLEL": "1",
    "FASTVIDEO_ULYSSES_A2A": "auto",
}

TASKS = ("t2va", "i2va", "l2va", "fl2va", "ref2va")
RESOLUTIONS = {"480p": "p480", "768p": "p768"}


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
    parser.add_argument("--resolution", choices=tuple(RESOLUTIONS), default="768p", help="Pika resolution tier")
    parser.add_argument("--aspect", choices=PIKA_ASPECTS, default="16:9", help="bucket aspect ratio (W:H)")
    parser.add_argument("--duration",
                        type=float,
                        default=5.0,
                        help=f"seconds ({PIKA_MIN_DURATION:g}-{PIKA_MAX_DURATION:g}); rounded up to the 17n+5 grid")
    parser.add_argument("--height", type=int, help="explicit canvas height (with --width; overrides the bucket)")
    parser.add_argument("--width", type=int, help="explicit canvas width (with --height; overrides the bucket)")
    parser.add_argument("--num-frames", type=int, help="explicit frame count (overrides --duration)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="torch.compile the DiT")
    parser.add_argument("--lossless-accel", action="store_true", help="enable bit-identical speed-ups")
    parser.add_argument("--offload",
                        choices=("lazy", "resident"),
                        default="lazy",
                        help="lazy: load components on demand and offload the encoders/VAE (fits smaller devices); "
                        "resident: keep everything on the GPU (fastest, about 105-130 GiB on GB200)")
    parser.add_argument("--low-vram", action="store_true", help="stream the DiT layerwise (lazy offload only)")
    parser.add_argument("--nvfp4-decoder",
                        help="Blackwell-only NVFP4 H3 decoder (quality-36L or fast-8L decoder-nvfp4.safetensors)")
    parser.add_argument("--nvfp4-text-encoder",
                        help="Blackwell-only serialized NVFP4 Qwen3-VL text_encoder directory")
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
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be passed together")
    if args.low_vram and args.offload == "resident":
        parser.error("--low-vram streams the DiT and cannot be combined with --offload resident")
    return args


def _offload(args: argparse.Namespace) -> OffloadConfig:
    if args.offload == "resident":
        return OffloadConfig(dit=False,
                             dit_layerwise=False,
                             text_encoder=False,
                             vae=False,
                             pin_cpu_memory=False,
                             lazy_module_load=False)
    return OffloadConfig(
        dit=False,
        dit_layerwise=args.low_vram,
        text_encoder=True,
        vae=True,
        pin_cpu_memory=not args.low_vram,
        lazy_module_load=True,
    )


def _canvas(args: argparse.Namespace) -> tuple[int, int]:
    if args.height is not None:
        return args.height, args.width
    return canvas_for_aspect(args.aspect, RESOLUTIONS[args.resolution])


def _num_frames(args: argparse.Namespace) -> int:
    if args.num_frames is not None:
        return args.num_frames
    return resolve_frames(round(args.duration * PIKA_FPS))


def _inputs(args: argparse.Namespace) -> InputConfig:
    if args.task == "ref2va":
        return InputConfig(references=args.reference)
    # Keep only the keyframes the task uses, so switching --task never turns
    # an I2VA run into FL2VA because a --last-image was also passed.
    first = args.image if args.task in ("i2va", "fl2va") else None
    last = args.last_image if args.task in ("l2va", "fl2va") else None
    # The request adapter copies image fields verbatim; the stage loads
    # ``image_path`` itself but requires ``last_image`` to already be a PIL image.
    return InputConfig(image_path=first, last_image=load_image(last) if last else None)


def _pipeline(args: argparse.Namespace) -> PipelineSelection:
    components = ComponentConfig()
    if args.task == "ref2va":
        components.override_pipeline_cls_name = "PikaH3FlexRef2VAModularPipeline"
    if args.nvfp4_decoder:
        components.h3_vae_nvfp4_decoder = args.nvfp4_decoder
    if args.nvfp4_text_encoder:
        components.text_encoder_weights = args.nvfp4_text_encoder
    return PipelineSelection(
        workload_type="t2v" if args.task == "t2va" else "i2v",
        components=components,
        vae_tiling=True,
    )


def main() -> None:
    args = parse_args()
    if args.lossless_accel:
        for key, value in LOSSLESS_ACCEL_ENV.items():
            os.environ.setdefault(key, value)

    height, width = _canvas(args)
    num_frames = _num_frames(args)
    print(f"Pika H3 Flex {args.task}: {width}x{height}, {num_frames} frames ({num_frames / PIKA_FPS:.2f} s)")
    output_path = Path(args.output) / f"pika_h3_flex_{args.task}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    offload = _offload(args)
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
                    height=height,
                    width=width,
                    num_frames=num_frames,
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
