# SPDX-License-Identifier: Apache-2.0
"""Generate a 768P draft with a FastH3 t2v checkpoint and save its native latent.

The refine pipeline can start from that latent directly (``draft_latent_path``), which skips the draft VAE decode and
re-encode. Writes ``draft_latent.pt`` ([C, T, h, w]) and ``draft.mp4`` next to each other.
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
    OffloadConfig,
    OutputConfig,
    PipelineSelection,
    QuantizationConfig,
    SamplingConfig,
)
from fastvideo.pipelines.basic.minimax_h3.packing import unpatchify_video_tokens
from fastvideo.pipelines.basic.minimax_h3.stages import minimax_h3_decoding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--backend", default="VIDEO_SPARSE_ATTN_H3")
    parser.add_argument("--quant", default="h3_dit_vsa")
    parser.add_argument("--sparsity", default="0.8")
    parser.add_argument("--tile-size", default="64")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = EngineConfig(num_gpus=1,
                          offload=OffloadConfig(dit=False, dit_layerwise=False, text_encoder=True, vae=False))
    if args.quant:
        engine.quantization = QuantizationConfig(transformer_quant="NVFP4", layer_profile=args.quant)
    experimental = {
        "attention_backend": args.backend,
    }
    if args.backend != "FLASH_ATTN":
        experimental["VSA_sparsity"] = float(args.sparsity)
        experimental["VSA_tile_size"] = int(args.tile_size)
    generator = VideoGenerator.from_config(
        GeneratorConfig(model_path=args.model_path,
                        engine=engine,
                        pipeline=PipelineSelection(components=ComponentConfig(), experimental=experimental)))
    pipeline = getattr(generator.executor.driver_worker, "worker", generator.executor.driver_worker).pipeline

    captured: dict[str, torch.Tensor] = {}
    real_unpatchify = minimax_h3_decoding.unpatchify_video_tokens

    def capturing_unpatchify(rows, num_latent_frames, latent_height, latent_width, channels, patch_size):
        latents = real_unpatchify(rows, num_latent_frames, latent_height, latent_width, channels, patch_size)
        captured.setdefault("draft_latent", latents[0].detach().float().cpu())
        return latents

    minimax_h3_decoding.unpatchify_video_tokens = capturing_unpatchify
    started = time.perf_counter()
    result = generator.generate(
        GenerationRequest(
            prompt=args.prompt,
            sampling=SamplingConfig(height=768,
                                    width=1344,
                                    num_frames=124,
                                    fps=24,
                                    num_inference_steps=args.steps,
                                    guidance_scale=1.0,
                                    seed=args.seed),
            output=OutputConfig(output_path=str(out / "draft.mp4"), save_video=True, return_frames=False),
        ))
    total = time.perf_counter() - started
    minimax_h3_decoding.unpatchify_video_tokens = real_unpatchify
    latent = captured["draft_latent"]
    torch.save(latent, out / "draft_latent.pt")
    (out / "draft_times.json").write_text(json.dumps({"total_s": total, "args": vars(args),
                                                     "latent_shape": list(latent.shape)}, indent=1))
    print("DRAFT", total, result.video_path, tuple(latent.shape), flush=True)
    generator.shutdown()


if __name__ == "__main__":
    main()
