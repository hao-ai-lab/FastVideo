# SPDX-License-Identifier: Apache-2.0
"""Run the private MiniMax-H3 T2VA pipeline with released weights on Shifu.

This is a Stage-4 acceptance runner, not a public example. Launch it with
``torchrun`` so the DiT can use FSDP and sequence parallelism while the shared
Qwen3-VL and VAE components follow the pipeline's CPU-offload lifecycle.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.distributed.parallel_state import cleanup_dist_env_and_memory
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3Pipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


DEFAULT_PROMPT = """integrated_multimodal_description: [Shot 1] Cinematic low-angle tracking shot of a bright red
vintage sports car accelerating along a rain-soaked city street at night. Neon signs reflect across the wet asphalt
and beads of water stream over the bodywork. At 00:02.500 the driver snaps the steering wheel and the car performs a
controlled drift around a sharp corner, throwing a fan of water toward the camera. The camera pans with the car and
ends on the glowing red tail lights as it rockets into a tunnel.
overall_soundscape: The engine rises from a deep mechanical growl into a loud high-rev roar. The drift is synchronized
with a sharp tire squeal and a broad splash of water, followed by a brief Doppler shift and reverberant tunnel echo.
non_diegetic_music: Tense electronic pulse with a strong bass hit exactly as the drift begins."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--sp-size", type=int, default=4)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    return parser.parse_args()


def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def save_joint_output(batch: ForwardBatch, output_path: Path, fps: int) -> None:
    if batch.output is None:
        raise RuntimeError("MiniMax-H3 pipeline returned no decoded video tensor.")
    audio = batch.extra.get("audio")
    sample_rate = batch.extra.get("audio_sample_rate")
    if not isinstance(audio, torch.Tensor) or not isinstance(sample_rate, int):
        raise RuntimeError("MiniMax-H3 pipeline returned no decoded stereo-audio tensor.")

    samples = batch.output.detach().float().cpu()
    audio = audio.detach().float().cpu()
    frames = [
        (frame.clamp(0, 1) * 255).round().to(torch.uint8).contiguous().numpy()
        for frame in samples[0].permute(1, 2, 3, 0)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not VideoGenerator._save_video_with_audio_single_pass(
        output_path=str(output_path),
        frames=frames,
        fps=fps,
        audio=audio,
        sample_rate=sample_rate,
    ):
        raise RuntimeError(f"Failed to mux MiniMax-H3 output at {output_path}.")


def main() -> None:
    args = parse_args()
    pipeline_config = MiniMaxH3PipelineConfig()
    pipeline = MiniMaxH3Pipeline.from_pretrained(
        args.model_path,
        pipeline_config=pipeline_config,
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
        output_type="pil",
    )

    batch = ForwardBatch(
        data_type="video",
        prompt=args.prompt,
        negative_prompt="",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=24,
        num_inference_steps=args.num_inference_steps,
        num_videos_per_prompt=1,
        seed=args.seed,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        guidance_scale=1.0,
        batch_cfg=False,
        save_video=False,
        return_frames=False,
    )

    start = time.perf_counter()
    output_batch = pipeline.forward(batch, pipeline.fastvideo_args)
    elapsed = time.perf_counter() - start
    output_path = Path(args.output)
    if rank() == 0:
        save_joint_output(output_batch, output_path, fps=24)
        metadata = {
            "model_path": args.model_path,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "num_gpus": args.num_gpus,
            "sp_size": args.sp_size,
            "attention_backend": args.attention_backend,
            "pipeline_seconds": elapsed,
            "prompt": args.prompt,
        }
        output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"STAGE4_OUTPUT={output_path}")
        print(f"STAGE4_PIPELINE_SECONDS={elapsed:.3f}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    pipeline.close()
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
