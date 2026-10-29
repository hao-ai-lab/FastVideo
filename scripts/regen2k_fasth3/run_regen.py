# SPDX-License-Identifier: Apache-2.0
"""Drive the Open-H3-Regenerate-2K refine on a swappable DiT, capturing timings and artefacts.

Exploratory experiment harness for the FastH3 refine study, not part of the library. Every run writes
``times.json`` (stage and per-forward seconds), ``latents.pt`` (the refined video rows) and an MP4.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

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
from fastvideo.pipelines.lazy_module import is_lazy_module

MEASURED_STAGES = ("latent_preparation_stage", "denoising_stage", "video_decoding_stage", "audio_decoding_stage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="/mnt/nfs/vlm-aryan/regen2k/weights")
    parser.add_argument("--transformer", default=None, help="DiT override (FastH3 export); omit for base H3")
    parser.add_argument("--quant", default=None, help="NVFP4 layer profile, e.g. h3_dit_vsa; omit for dense bf16")
    parser.add_argument("--video", default=None, help="768P draft video; omit when --draft-latent is used")
    parser.add_argument("--draft-latent", default=None, help=".pt holding the native draft latent [C, T, h, w]")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--backend", default="FLASH_ATTN")
    parser.add_argument("--sparsity", type=float, default=None, help="VSA sparsity; omit to keep the checkpoint's")
    parser.add_argument("--tile-size", type=int, default=None, help="VSA tile size; omit to keep 64")
    parser.add_argument("--seed", type=int, default=1766580932)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def pipeline_experimental(args: argparse.Namespace) -> dict[str, object]:
    experimental: dict[str, object] = {
        "attention_backend": args.backend,
        "draft_size": "768p",
        "refine_transformer_dir": args.transformer,
        "draft_latent_path": args.draft_latent,
    }
    if args.backend != "FLASH_ATTN":
        if args.sparsity is not None:
            experimental["VSA_sparsity"] = args.sparsity
        if args.tile_size is not None:
            experimental["VSA_tile_size"] = args.tile_size
    return experimental


def instrument(pipeline, timings: dict[str, float], forwards: list[float], captured: dict[str, torch.Tensor]) -> None:
    """Wrap the measured stages and the DiT forward as the pipeline adds them (stages are built on first run)."""
    forward_started: list[float] = []

    def hook_transformer(transformer):
        if is_lazy_module(transformer):
            transformer = transformer.materialize()
        transformer.register_forward_pre_hook(lambda module, args: forward_started.append(time.perf_counter()))
        transformer.register_forward_hook(lambda module, args, output: forwards.append(time.perf_counter() -
                                                                                     forward_started[-1]))

    def add_stage(stage_name, stage, original_add=pipeline.add_stage):
        if stage_name == "denoising_stage":
            hook_transformer(stage.transformer)
        if stage_name not in MEASURED_STAGES:
            return original_add(stage_name, stage)
        original = stage.forward

        def wrapper(batch, fastvideo_args, original=original, name=stage_name):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = original(batch, fastvideo_args)
            torch.cuda.synchronize()
            timings[name] = time.perf_counter() - started
            if name == "denoising_stage":
                captured["refined_rows"] = batch.latents.detach().float().cpu()
            return result

        stage.forward = wrapper
        return original_add(stage_name, stage)

    pipeline.add_stage = add_stage


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = EngineConfig(num_gpus=1,
                          offload=OffloadConfig(dit=False, dit_layerwise=False, text_encoder=True, vae=False))
    if args.quant:
        engine.quantization = QuantizationConfig(transformer_quant="NVFP4", layer_profile=args.quant)
    generator = VideoGenerator.from_config(
        GeneratorConfig(model_path=args.weights,
                        engine=engine,
                        pipeline=PipelineSelection(components=ComponentConfig(),
                                                   experimental=pipeline_experimental(args))))
    worker = generator.executor.driver_worker
    pipeline = getattr(worker, "worker", worker).pipeline
    timings: dict[str, float] = {}
    forwards: list[float] = []
    captured: dict[str, torch.Tensor] = {}
    instrument(pipeline, timings, forwards, captured)

    runs = []
    for run in range(args.runs):
        forwards.clear()
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
                output=OutputConfig(output_path=str(out / f"run_{run}.mp4"), save_video=True, return_frames=False),
            ))
        total = time.perf_counter() - started
        runs.append({
            "total_s": total,
            "per_forward_s": list(forwards),
            "stages_s": dict(timings),
        })
        print("RUN", run, json.dumps(runs[-1]), result.video_path, flush=True)
        if run == 0:
            torch.save(captured["refined_rows"], out / "latents.pt")
    (out / "times.json").write_text(json.dumps({"args": vars(args), "runs": runs}, indent=1))
    generator.shutdown()
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
