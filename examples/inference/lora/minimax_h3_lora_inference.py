# SPDX-License-Identifier: Apache-2.0
"""Run a FastH3 distillation adapter on top of the base MiniMax-H3 checkpoint.

The FastH3 checkpoints are full fine-tunes of MiniMax-H3 distilled to four steps under
video sparse attention. Published as adapters they are three things at once, and all
three have to land for the result to match the checkpoint:

* low-rank factors for the attention, feed-forward, and AdaLN projections
* exact ``.diff`` deltas for the norms and biases an SVD cannot usefully factor
* ``.set_weight`` values for ``attn.to_gate_compress``, the VSA compression gate that
  does not exist in the base model at all

The last one is why ``--vsa`` is not optional here. Under any other attention backend
the gate module is never constructed, so half the distillation has nowhere to go; the
loader will say so rather than quietly produce a worse video.

Because a parameter the base lacks has to be supplied while weights are still unsharded,
the adapter is passed at construction rather than swapped in afterwards.

    python examples/inference/lora/minimax_h3_lora_inference.py \\
        --lora-path /models/fasth3-loras-publish/FastH3-4-step-v1.1/rank-64 \\
        --prompts-file prompts.jsonl --output outputs/v1.1-rank64

Pass no ``--lora-path`` to render the unmodified base model as a control.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

# MiniMax-H3 generates 5-15 s at 24 fps, on a latent grid that only admits frame counts
# of the form 17n + 5. 124 is the 5-second point the FastH3 profile is measured at.
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
FPS = 24
MIN_DURATION, MAX_DURATION = 5.0, 15.0


def align_num_frames(num_frames: int) -> int:
    """Round up to the next 17n + 5 the latent grid accepts."""
    if num_frames <= LATENTS_PER_CHUNK:
        return LATENTS_PER_CHUNK
    chunks = -(-(num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK)
    return LATENTS_PER_CHUNK + chunks * FRAMES_PER_CHUNK


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3", help="the BASE checkpoint the adapter targets")
    parser.add_argument("--lora-path", default=None, help="adapter file or directory; omit to render the base model")
    parser.add_argument("--lora-nickname", default="fasth3")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompts-file", default=None, help="JSONL with a 'prompt' field per line")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N prompts")
    parser.add_argument("--num-shards", type=int, default=1, help="split the prompt list across processes")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--output", default="outputs/minimax_h3_lora")
    parser.add_argument("--skip-existing", action="store_true", help="leave already-rendered clips alone")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    # Counts sigma-GRID POINTS: N points run N-1 DiT forwards. The distilled ladder is
    # t=1000,750,500,250 -> 0, which is five points and exactly four forwards.
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--vsa", action=argparse.BooleanOptionalAction, default=True,
                        help="video sparse attention; required for any adapter carrying to_gate_compress")
    parser.add_argument("--vsa-sparsity", type=float, default=0.9)
    parser.add_argument("--vsa-tile-size", type=int, choices=(64, 256), default=64)
    parser.add_argument("--vsa-kernel", choices=("triton", "sm100a"), default="sm100a")
    parser.add_argument("--fa4", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not args.prompt and not args.prompts_file:
        parser.error("pass --prompt or --prompts-file")
    if args.lora_path and not args.vsa:
        parser.error("--no-vsa with an adapter: to_gate_compress would have no module to load into. "
                     "Drop --no-vsa, or use an adapter with no .set_weight keys.")
    aligned = align_num_frames(args.num_frames)
    if not MIN_DURATION <= aligned / FPS <= MAX_DURATION:
        parser.error(f"MiniMax-H3 generates {MIN_DURATION}-{MAX_DURATION}s at {FPS} fps; "
                     f"aligned num_frames={aligned} is {aligned / FPS:.1f}s")
    args.num_frames = aligned
    return args


def configure_environment(args: argparse.Namespace) -> None:
    """Set the boot-time backend selection explicitly, including what is off.

    An inherited FASTVIDEO_* from an earlier experiment would otherwise silently change
    which attention path the run actually took, which is the one thing this comparison
    cannot afford to be vague about.
    """
    env = {
        "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3" if args.vsa else "FLASH_ATTN",
        "FASTVIDEO_VSA_SM100A": "1" if (args.vsa and args.vsa_kernel == "sm100a") else "0",
        "FASTVIDEO_VSA_CUTEDSL": "0",
        "FASTVIDEO_FA4": "1" if args.fa4 else "0",
        "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",
        "FASTVIDEO_INFERENCE_TORCH_COMPILE": "0",
        "FASTVIDEO_STAGE_LOGGING": "1",
    }
    for name, value in env.items():
        os.environ[name] = value


def load_prompts(args: argparse.Namespace) -> list[dict]:
    if args.prompt:
        records = [{"id": "000", "prompt": args.prompt}]
    else:
        records = []
        with open(args.prompts_file) as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                records.append({
                    "id": str(item.get("id", item.get("sample_id", f"{index:03d}"))),
                    "prompt": item["prompt"],
                })
    if args.limit is not None:
        records = records[:args.limit]
    return [r for i, r in enumerate(records) if i % args.num_shards == args.shard]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_environment(args)

    # Imported after the environment is set: backend selection is read at import time.
    from fastvideo import VideoGenerator
    from fastvideo.api import (ComponentConfig, EngineConfig, GenerationRequest, GeneratorConfig, OffloadConfig,
                               OutputConfig, ParallelismConfig, PipelineSelection, SamplingConfig)

    experimental: dict[str, object] = {
        "vae_parallel_decode": True,
        "vae_parallel_decode_strategy": "gather",
    }
    if args.vsa:
        experimental.update({
            "attention_backend": "VIDEO_SPARSE_ATTN_H3",
            "VSA_sparsity": args.vsa_sparsity,
            "VSA_tile_size": args.vsa_tile_size,
        })

    config = GeneratorConfig(
        model_path=args.model_path,
        pipeline=PipelineSelection(
            components=ComponentConfig(lora_path=args.lora_path),
            experimental=experimental,
        ),
        engine=EngineConfig(
            num_gpus=args.num_gpus,
            use_fsdp_inference=False,
            parallelism=ParallelismConfig(tp_size=1, sp_size=args.num_gpus),
            offload=OffloadConfig(dit=False, dit_layerwise=False, text_encoder=True, vae=True, pin_cpu_memory=True),
        ),
    )

    records = load_prompts(args)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"adapter: {args.lora_path or '(none, base model)'}")
    print(f"prompts: {len(records)} (shard {args.shard}/{args.num_shards})")

    generator = VideoGenerator.from_config(config)
    for index, record in enumerate(records):
        stem = f"{index:03d}_{record['id']}"
        if args.skip_existing and (out_dir / f"{stem}.mp4").exists():
            print(f"[{index}] skip {stem}")
            continue
        generator.generate(
            GenerationRequest(
                prompt=record["prompt"],
                negative_prompt="",
                sampling=SamplingConfig(
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    fps=FPS,
                    num_inference_steps=args.steps,
                    # MiniMax-H3 is guidance-distilled; FastH3 inherits that contract.
                    guidance_scale=1.0,
                    batch_cfg=False,
                    seed=args.seed,
                ),
                output=OutputConfig(output_path=str(out_dir / f"{stem}.mp4"), save_video=True, return_frames=False),
            ))
        print(f"[{index}] wrote {stem}")


if __name__ == "__main__":
    main()
