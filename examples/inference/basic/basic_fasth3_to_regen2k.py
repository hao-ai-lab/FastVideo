# SPDX-License-Identifier: Apache-2.0
"""Draft with a FastH3 checkpoint at 768P, then regenerate the draft at 2560x1440 with Open-H3-Regenerate-2K.

The draft's native latent goes straight to the regen's upscaler: the draft is never decoded to pixels and re-encoded.
The regen stack is unchanged (base MiniMax-H3, the released merged LoRA and Sharp branches, three Euler steps).

    python examples/inference/basic/basic_fasth3_to_regen2k.py \
        --draft-model FastVideo/FastVideo-FastH3-8-Step-V2 --regen-model Efficient-Large-Model/Open-H3-Regenerate-2K \
        --prompt "A red fox trots through fresh snow" --output outputs/fasth3_regen2k

A packed NVFP4 draft checkpoint needs ``--draft-quant h3_dit_vsa``.
"""

from __future__ import annotations

import argparse
import json
import os
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
from fastvideo.utils import maybe_download_model

DRAFT_SIZE = (768, 1344)
NUM_FRAMES = 124


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft-model", required=True, help="FastH3 checkpoint id or local path")
    parser.add_argument("--draft-quant", default=None, help="NVFP4 layer profile of a packed draft, e.g. h3_dit_vsa")
    parser.add_argument("--regen-model", default="Efficient-Large-Model/Open-H3-Regenerate-2K")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--draft-steps", type=int, default=9, help="sigma-grid points of the draft (9 = 8 forwards)")
    parser.add_argument("--draft-seed", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1766580932, help="regen refine noise seed")
    parser.add_argument("--output", type=Path, default=Path("outputs/fasth3_regen2k"))
    return parser.parse_args()


def normalize_h3_latents(latents: torch.Tensor, regen_model: str) -> torch.Tensor:
    """Re-apply the H3 VAE's latent statistics: a draft run returns denormalized latents, the regen takes normalized."""
    vae_config = os.path.join(maybe_download_model(regen_model, allow_patterns=["base_model/vae/config.json"]),
                              "base_model", "vae", "config.json")
    with open(vae_config, encoding="utf-8") as handle:
        config = json.load(handle)
    mean = torch.tensor(config["latents_mean"], dtype=torch.float32).view(-1, 1, 1, 1)
    std = torch.tensor(config["latents_std"], dtype=torch.float32).view(-1, 1, 1, 1)
    return (latents.float().cpu().reshape(-1, *latents.shape[-3:]) - mean) / std


def draft_latent(args: argparse.Namespace) -> torch.Tensor:
    engine = EngineConfig(num_gpus=1, offload=OffloadConfig(dit=False, dit_layerwise=False, vae=False))
    if args.draft_quant:
        engine.quantization = QuantizationConfig(transformer_quant="NVFP4", layer_profile=args.draft_quant)
    generator = VideoGenerator.from_config(
        GeneratorConfig(model_path=args.draft_model,
                        engine=engine,
                        pipeline=PipelineSelection(components=ComponentConfig(),
                                                   experimental={
                                                       "attention_backend": "VIDEO_SPARSE_ATTN_H3",
                                                       "VSA_sparsity": 0.8,
                                                       "VSA_tile_size": 64,
                                                       "output_type": "latent",
                                                   })))
    try:
        result = generator.generate(
            GenerationRequest(prompt=args.prompt,
                              sampling=SamplingConfig(height=DRAFT_SIZE[0],
                                                      width=DRAFT_SIZE[1],
                                                      num_frames=NUM_FRAMES,
                                                      fps=24,
                                                      num_inference_steps=args.draft_steps,
                                                      guidance_scale=1.0,
                                                      seed=args.draft_seed),
                              output=OutputConfig(save_video=False, return_frames=True)))
    finally:
        generator.shutdown()
    return normalize_h3_latents(torch.as_tensor(result.frames), args.regen_model)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    latent_path = args.output / "draft_latent.pt"
    torch.save(draft_latent(args), latent_path)

    generator = VideoGenerator.from_config(
        GeneratorConfig(model_path=args.regen_model,
                        engine=EngineConfig(num_gpus=1,
                                            offload=OffloadConfig(dit=False, dit_layerwise=False, text_encoder=True,
                                                                  vae=False)),
                        pipeline=PipelineSelection(components=ComponentConfig(),
                                                   experimental={
                                                       "attention_backend": "TORCH_SDPA",
                                                       "draft_size": "768p",
                                                   })))
    try:
        result = generator.generate(
            GenerationRequest(prompt=args.prompt,
                              inputs=InputConfig(video_path=str(latent_path)),
                              sampling=SamplingConfig(height=1440,
                                                      width=2560,
                                                      num_frames=NUM_FRAMES,
                                                      fps=24,
                                                      num_inference_steps=3,
                                                      guidance_scale=1.0,
                                                      seed=args.seed),
                              output=OutputConfig(output_path=str(args.output / "regen2k.mp4"),
                                                  save_video=True,
                                                  return_frames=False)))
        print(f"wrote {result.video_path}")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
