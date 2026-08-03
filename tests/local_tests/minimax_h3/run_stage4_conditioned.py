# SPDX-License-Identifier: Apache-2.0
"""Run released MiniMax H3 FL2VA or Ref2VA weights on multiple GPUs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image
import torch
import torch.distributed as dist

from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.distributed.parallel_state import cleanup_dist_env_and_memory
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.pipelines.basic.minimax_h3 import MiniMaxH3Reference
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import (
    MiniMaxH3ModularPipeline,
    MiniMaxH3Ref2VAModularPipeline,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fl2va", "ref2va"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", help="Plain text or an official reproducible request shell script.")
    parser.add_argument("--image", help="FL2VA first-frame path.")
    parser.add_argument("--last-image", help="FL2VA optional last-frame path.")
    parser.add_argument("--reference-video", help="Ref2VA video reference path.")
    parser.add_argument("--reference-audio", help="Ref2VA optional extra audio reference path.")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--sp-size", type=int, default=4)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--output-type", choices=("pil", "latent"), default="pil")
    return parser.parse_args()


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if not args.prompt_file:
        raise ValueError("Pass --prompt or --prompt-file.")
    text = Path(args.prompt_file).read_text(encoding="utf-8")
    marker = "<<'JSON'"
    if marker in text:
        request_text = text.split(marker, 1)[1].split("\nJSON", 1)[0].strip()
        return str(json.loads(request_text)["prompt"])
    return text.strip()


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _prepare_joint_output(batch: ForwardBatch) -> tuple[list[Any], torch.Tensor, int]:
    """Move decoded media to CPU before tearing down the distributed runtime."""
    if batch.output is None:
        raise RuntimeError("MiniMax-H3 pipeline returned no decoded video tensor.")
    audio = batch.extra.get("audio")
    sample_rate = batch.extra.get("audio_sample_rate")
    if not isinstance(audio, torch.Tensor) or not isinstance(sample_rate, int):
        raise RuntimeError("MiniMax-H3 pipeline returned no decoded stereo-audio tensor.")
    samples = batch.output.detach().float().cpu()
    frames = [
        (frame.clamp(0, 1) * 255).round().to(torch.uint8).contiguous().numpy()
        for frame in samples[0].permute(1, 2, 3, 0)
    ]
    return frames, audio.detach().float().cpu(), sample_rate


def _build_batch(args: argparse.Namespace, prompt: str) -> ForwardBatch:
    common = {
        "data_type": "video",
        "prompt": prompt,
        "negative_prompt": "",
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": 24,
        "num_inference_steps": args.num_inference_steps,
        "num_videos_per_prompt": 1,
        "seed": args.seed,
        "generator": torch.Generator("cpu").manual_seed(args.seed),
        "guidance_scale": 1.0,
        "batch_cfg": False,
        "save_video": False,
        "return_frames": False,
    }
    if args.mode == "fl2va":
        if not args.image:
            raise ValueError("FL2VA requires --image.")
        common["pil_image"] = Image.open(args.image).convert("RGB")
        if args.last_image:
            common["last_image"] = Image.open(args.last_image).convert("RGB")
        if args.num_frames is None:
            common["num_frames"] = 192
    else:
        if not args.reference_video:
            raise ValueError("Ref2VA requires --reference-video.")
        references = [MiniMaxH3Reference(source=args.reference_video, media_type="video")]
        if args.reference_audio:
            references.append(MiniMaxH3Reference(source=args.reference_audio, media_type="audio"))
        common["references"] = references
        if args.height is None and args.width is None:
            common["height"], common["width"] = 768, 1344
        if args.num_frames is None:
            common["num_frames"] = 124
    return ForwardBatch(**common)


def main() -> None:
    args = parse_args()
    prompt = _load_prompt(args)
    pipeline_cls = MiniMaxH3ModularPipeline if args.mode == "fl2va" else MiniMaxH3Ref2VAModularPipeline
    pipeline = pipeline_cls.from_pretrained(
        args.model_path,
        pipeline_config=MiniMaxH3PipelineConfig(),
        num_gpus=args.num_gpus,
        tp_size=1,
        sp_size=args.sp_size,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=args.num_gpus,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        text_encoder_cpu_offload=True,
        vae_cpu_offload=True,
        pin_cpu_memory=False,
        attention_backend=args.attention_backend,
        output_type=args.output_type,
    )

    joint_output: tuple[list[Any], torch.Tensor, int] | None = None
    latent_output: dict[str, torch.Tensor] | None = None
    is_main_process = _rank() == 0
    try:
        start = time.perf_counter()
        output_batch = pipeline.forward(_build_batch(args, prompt), pipeline.fastvideo_args)
        elapsed = time.perf_counter() - start
        output_path = Path(args.output)
        if is_main_process:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            assert output_batch.output is not None
            if args.output_type == "latent":
                audio_latents = output_batch.extra.get("audio")
                if not isinstance(audio_latents, torch.Tensor):
                    raise RuntimeError("MiniMax-H3 pipeline returned no audio latents.")
                latent_output = {
                    "video": output_batch.output.detach().float().cpu(),
                    "audio": audio_latents.detach().float().cpu(),
                }
            else:
                joint_output = _prepare_joint_output(output_batch)
            metadata = {
                "mode": args.mode,
                "model_path": args.model_path,
                "height": int(output_batch.output.shape[-2]),
                "width": int(output_batch.output.shape[-1]),
                "num_frames": int(output_batch.output.shape[-3]),
                "num_inference_steps": args.num_inference_steps,
                "seed": args.seed,
                "num_gpus": args.num_gpus,
                "sp_size": args.sp_size,
                "attention_backend": args.attention_backend,
                "output_type": args.output_type,
                "pipeline_seconds": elapsed,
                "image": args.image,
                "last_image": args.last_image,
                "reference_video": args.reference_video,
                "reference_audio": args.reference_audio,
                "prompt": prompt,
            }
            output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    finally:
        pipeline.close()
        cleanup_dist_env_and_memory()

    # PyAV occasionally blocks in container close when its encoder threads are
    # still sharing a live NCCL process. Encode only after every rank has
    # released the distributed runtime; the media already resides on CPU.
    if is_main_process:
        if args.output_type == "latent":
            assert latent_output is not None
            torch.save(latent_output, output_path)
        else:
            assert joint_output is not None
            frames, audio, sample_rate = joint_output
            if not VideoGenerator._save_video_with_audio_single_pass(
                output_path=str(output_path),
                frames=frames,
                fps=24,
                audio=audio,
                sample_rate=sample_rate,
            ):
                raise RuntimeError(f"Failed to mux MiniMax-H3 output at {output_path}.")
        print(f"STAGE4_OUTPUT={output_path}")
        print(f"STAGE4_PIPELINE_SECONDS={elapsed:.3f}")


if __name__ == "__main__":
    main()
