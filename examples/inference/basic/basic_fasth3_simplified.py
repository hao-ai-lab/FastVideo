# SPDX-License-Identifier: Apache-2.0
"""Generate 5-second FastH3 videos with a supported Preview adapter."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

FASTVIDEO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FASTVIDEO_ROOT))

from fastvideo import VideoGenerator  # noqa: E402
from fastvideo.api import (  # noqa: E402
    CompileConfig,
    ComponentConfig,
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)

# FastVideo exposes these three backend switches only through environment variables.
os.environ.update({
    "FASTVIDEO_FA4": "1",
    "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",
    "FASTVIDEO_VSA_SM100A": "0",
})
os.environ.pop("FASTVIDEO_INFERENCE_TORCH_COMPILE", None)

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
    return parser.parse_args()


def main() -> None:
    """Run one compile warmup and three measured generations with one fixed recipe."""
    args = parse_args()

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
    output_dir = FASTVIDEO_ROOT / "outputs/fasth3_lora_preview" / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)

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
            ),
        )
    )

    generation_count = 4
    warmup_count = 1
    measured_count = generation_count - warmup_count
    measured_seconds: list[float] = []

    print(f"Variant: {args.variant} ({attention_backend})")
    print(f"Output directory: {output_dir}")
    try:
        for generation_index in range(generation_count):
            measured = generation_index >= warmup_count
            if measured:
                measured_index = generation_index - warmup_count + 1
                label = f"measured {measured_index}/{measured_count}"
                output_path = output_dir / f"fasth3_all_run_{measured_index:02d}.mp4"
            else:
                warmup_index = generation_index + 1
                label = f"warmup {warmup_index}/{warmup_count}"
                output_path = output_dir / f"_fasth3_warmup_{warmup_index:02d}.mp4"

            request = GenerationRequest(
                prompt=args.prompt,
                negative_prompt="",
                sampling=SamplingConfig(
                    height=768,
                    width=1344,
                    num_frames=345, # <-- Change video length: 5sec: 124; 10sec: 243; 15sec: 345.
                    fps=24,
                    num_inference_steps=5,
                    guidance_scale=1.0,
                    batch_cfg=False,
                    seed=1000,
                ),
                output=OutputConfig(
                    output_path=str(output_path),
                    save_video=True,
                    return_frames=False,
                ),
            )
            started = time.perf_counter()
            result = generator.generate(request)
            elapsed = time.perf_counter() - started
            if measured:
                measured_seconds.append(elapsed)
            suffix = "" if measured else " (excluded from median)"
            print(f"[{label}] {result.video_path or output_path}: {elapsed:.3f}s{suffix}")
    finally:
        generator.shutdown()

    print(f"Median E2E wall time: {statistics.median(measured_seconds):.3f}s")


if __name__ == "__main__":
    main()
