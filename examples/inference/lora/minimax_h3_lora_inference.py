# SPDX-License-Identifier: Apache-2.0
"""Run a FastH3 distillation adapter on top of the base MiniMax-H3 checkpoint.

The FastH3 checkpoints are full fine-tunes of MiniMax-H3 distilled to four steps under
video sparse attention. Published as adapters they are three things at once, and all
three have to land for the result to match the checkpoint:

* low-rank factors for the attention, feed-forward, and AdaLN projections
* exact ``.diff`` deltas for the norms and biases an SVD cannot usefully factor
* ``.set_weight`` values for ``attn.to_gate_compress``, the VSA compression gate that
  does not exist in the base model at all

An adapter carrying that last one needs ``--vsa``: under any other attention backend the
gate module is never constructed, so part of the distillation has nowhere to go. The
requirement is read off the adapter rather than assumed, because community adapters
built against the ComfyUI layout carry no gate and run fine either way -- run one of
those with ``--no-vsa`` (see ``scripts/checkpoint_conversion/convert_minimax_h3_comfy_lora.py``
for getting them into a layout this loads).

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
import math
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
    parser.add_argument("--lora-strength", type=float, default=1.0)
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
    parser.add_argument("--vsa", action=argparse.BooleanOptionalAction, default=None,
                        help="video sparse attention; inferred from the adapter when omitted")
    parser.add_argument("--vsa-sparsity", type=float, default=0.9)
    parser.add_argument("--vsa-tile-size", type=int, choices=(64, 256), default=64)
    parser.add_argument("--vsa-kernel", choices=("triton", "sm100a"), default="sm100a")
    parser.add_argument("--fa4", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not args.prompt and not args.prompts_file:
        parser.error("pass --prompt or --prompts-file")
    # Whether VSA is required is a property of the adapter, not of having one at all --
    # community adapters built against the ComfyUI layout carry no gate. Checked in
    # main(), once the path has been resolved.
    aligned = align_num_frames(args.num_frames)
    if not MIN_DURATION <= aligned / FPS <= MAX_DURATION:
        parser.error(f"MiniMax-H3 generates {MIN_DURATION}-{MAX_DURATION}s at {FPS} fps; "
                     f"aligned num_frames={aligned} is {aligned / FPS:.1f}s")
    args.num_frames = aligned
    if not math.isfinite(args.lora_strength):
        parser.error("--lora-strength must be finite")
    return args


def configure_environment(args: argparse.Namespace) -> None:
    """Set the boot-time backend selection explicitly, including what is off.

    An inherited FASTVIDEO_* from an earlier experiment would otherwise silently change
    which attention path the run actually took, which is the one thing this comparison
    cannot afford to be vague about.
    """
    env: dict[str, str | None] = {
        "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3" if args.vsa else "FLASH_ATTN",
        "FASTVIDEO_VSA_SM100A": "1" if (args.vsa and args.vsa_kernel == "sm100a") else "0",
        "FASTVIDEO_VSA_CUTEDSL": "0",
        "FASTVIDEO_H3_VSA_PROBE": None,
        "FASTVIDEO_DISABLE_ATTENTION_COMPILE": "0",
        "FASTVIDEO_FA4": "1" if args.fa4 else "0",
        "FASTVIDEO_NVFP4_FA4": "0",
        "FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN": "0",
        "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",
        "FASTVIDEO_INFERENCE_TORCH_COMPILE": "1",
        "FASTVIDEO_VAE_PARALLEL_DECODE": "1",
        "FASTVIDEO_VAE_PARALLEL_ENCODE": "0",
        "FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY": "gather",
        "FASTVIDEO_ULYSSES_A2A": "off",
        "FASTVIDEO_STAGE_LOGGING": "1",
    }
    for name, value in env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
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
    # Backend selection is finalized from the adapter before model construction.
    from fastvideo.models.loader.lora_patch import DenseLoRAPatch
    from fastvideo import VideoGenerator
    from fastvideo.api import (CompileConfig, ComponentConfig, EngineConfig, GenerationRequest, GeneratorConfig,
                               OffloadConfig, OutputConfig, ParallelismConfig, PipelineSelection, SamplingConfig)

    # An adapter carrying to_gate_compress needs the VSA backend, because that is the
    # only configuration in which the module exists. One that does not carry it runs
    # fine either way, so the requirement is read off the file rather than assumed from
    # the presence of an adapter at all.
    patch = (DenseLoRAPatch.from_adapter(args.lora_path, strength=args.lora_strength)
             if args.lora_path else None)
    needs_vsa = bool(patch and any("gate_compress" in name for name in patch.replacement_parameters))
    if args.vsa is None:
        args.vsa = needs_vsa
    if needs_vsa and not args.vsa:
        raise SystemExit(f"{args.lora_path} carries to_gate_compress, which exists only under the VSA "
                         "attention backend. Drop --no-vsa.")
    if args.vsa and args.lora_path and not needs_vsa:
        print(f"note: {args.lora_path} carries no VSA gate; running under VSA leaves the "
              "compression branch at its zero-initialized value.")
    configure_environment(args)

    experimental: dict[str, object] = {
        "inference_torch_compile": True,
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
            components=ComponentConfig(lora_path=args.lora_path, lora_strength=args.lora_strength),
            experimental=experimental,
        ),
        engine=EngineConfig(
            num_gpus=args.num_gpus,
            use_fsdp_inference=False,
            parallelism=ParallelismConfig(tp_size=1, sp_size=args.num_gpus),
            offload=OffloadConfig(dit=False, dit_layerwise=False, text_encoder=True, vae=True, pin_cpu_memory=True),
            compile=CompileConfig(enabled=False, vae_enabled=True),
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
