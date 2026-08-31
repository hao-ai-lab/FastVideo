# SPDX-License-Identifier: Apache-2.0
"""Profile one FastH3 Preview generation with NVTX after warmup."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from fastvideo import VideoGenerator
from fastvideo.api import (
    CompileConfig,
    ComponentConfig,
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    QuantizationConfig,
    SamplingConfig,
)
from fastvideo.profiler import nvtx_range

VARIANT_BACKENDS = {
    "dense-datafree": "FLASH_ATTN",
    "vsa-datafree": "VIDEO_SPARSE_ATTN_H3",
    "vsa-synthetic-step1300": "VIDEO_SPARSE_ATTN_H3",
    "vsa-synthetic-step1900": "VIDEO_SPARSE_ATTN_H3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANT_BACKENDS)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, default=345)
    parser.add_argument("--warmup-runs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Warm the selected FastH3 recipe, then profile one identical generation."""
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    attention_backend = VARIANT_BACKENDS[args.variant]
    experimental = {
        "attention_backend": attention_backend,
        "inference_torch_compile": attention_backend == "FLASH_ATTN",
        "vae_parallel_decode": True,
        "vae_parallel_decode_strategy": "gather",
    }
    if attention_backend == "VIDEO_SPARSE_ATTN_H3":
        experimental.update({
            "VSA_sparsity": 0.9,
            "VSA_tile_size": 64,
        })

    adapter_path = hf_hub_download(
        repo_id="FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA",
        filename=f"{args.variant}/adapter_model.safetensors",
    )
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path="MiniMaxAI/MiniMax-H3",
            pipeline=PipelineSelection(
                components=ComponentConfig(lora_path=adapter_path, lora_strength=1.0),
                experimental=experimental,
            ),
            engine=EngineConfig(
                num_gpus=4,
                parallelism=ParallelismConfig(tp_size=1, sp_size=4),
                offload=OffloadConfig(dit=False, dit_layerwise=False),
                compile=CompileConfig(vae_enabled=True),
                quantization=QuantizationConfig(transformer_quant="MXFP8"),
            ),
        )
    )

    request = GenerationRequest(
        prompt=args.prompt,
        negative_prompt="",
        sampling=SamplingConfig(
            height=768,
            width=1344,
            num_frames=args.num_frames,
            fps=24,
            num_inference_steps=5,
            guidance_scale=1.0,
            batch_cfg=False,
            seed=1000,
        ),
        output=OutputConfig(
            output_path=str(output_dir / "fasth3_profile.mp4"),
            save_video=True,
            return_frames=False,
        ),
    )

    try:
        for warmup_index in range(args.warmup_runs):
            warmup_result = generator.generate(request)
            print(f"Warmup {warmup_index + 1}/{args.warmup_runs}: {warmup_result.video_path}")

        torch.cuda.profiler.start()
        try:
            with nvtx_range("fasth3.profiled_generation"):
                measured_result = generator.generate(request)
        finally:
            torch.cuda.profiler.stop()

        print(f"Profiled output: {measured_result.video_path}")
        if measured_result.generation_time is not None:
            print(f"Generation time: {measured_result.generation_time:.2f}s")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
