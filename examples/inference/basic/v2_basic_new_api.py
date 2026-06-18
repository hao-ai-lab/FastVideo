"""v2 typed-API inference example — mirrors ``basic_dmd_new_api.py`` but drives the **v2
(recipe, runtime) substrate + real torch backend** for the three models brought up on GPU
(Wan2.1, SF-causal Wan, LTX-2).

The ONLY delta from the upstream example is importing ``VideoGenerator`` from ``v2`` instead of
``fastvideo`` — the typed config classes are the SAME ``fastvideo.api`` dataclasses.

Run (on a GPU box, with the v2 venv active):
    python examples/inference/basic/v2_basic_new_api.py

Notes vs upstream: the v2 bring-up runs single-GPU, resident, on the TORCH_SDPA backend (no
fastvideo-kernel / VSA), so resolutions/steps are modest here for a quick runnable demo. LTX-2 loads
an 18.88B DiT (slow first load).
"""
import os
import time

from v2 import VideoGenerator
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    SamplingConfig,
)

OUTPUT_PATH = "v2_video_samples"

MODELS = [
    {
        "family": "wan21",
        "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "prompt": "a red panda surfing on ocean waves at sunset, cinematic, highly detailed",
        "sampling": SamplingConfig(num_frames=25, height=480, width=832,
                                   num_inference_steps=30, guidance_scale=5.0, seed=1, fps=16),
    },
    {
        "family": "wan_causal",
        "model_path": "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers",
        "prompt": "a cat walking through a sunlit garden, cinematic",
        "sampling": SamplingConfig(num_frames=25, height=480, width=832,
                                   num_inference_steps=4, guidance_scale=5.0, seed=1, fps=16),
    },
    {
        "family": "ltx2",
        "model_path": "FastVideo/LTX2-Distilled-Diffusers",
        "prompt": "surfers riding ocean waves at sunset, cinematic, highly detailed",
        "sampling": SamplingConfig(num_frames=9, height=512, width=768,
                                   num_inference_steps=8, guidance_scale=1.0, seed=1, fps=16),
    },
]


def run_one(m: dict) -> None:
    generator_config = GeneratorConfig(
        model_path=m["model_path"],
        engine=EngineConfig(
            num_gpus=1,
            use_fsdp_inference=False,
            offload=OffloadConfig(text_encoder=False, dit=False, vae=False, pin_cpu_memory=False),
        ),
    )
    load_start = time.perf_counter()
    generator = VideoGenerator.from_config(generator_config)
    load_time = time.perf_counter() - load_start

    request = GenerationRequest(
        prompt=m["prompt"],
        sampling=m["sampling"],
        output=OutputConfig(output_path=OUTPUT_PATH, output_video_name=f"v2_{m['family']}",
                            save_video=True, return_frames=False),
    )
    gen_start = time.perf_counter()
    result = generator.generate(request)
    gen_time = time.perf_counter() - gen_start

    print(f"[{m['family']:10s}] load={load_time:6.1f}s  gen={gen_time:6.1f}s  -> {result.video_path}")


def main() -> None:
    for m in MODELS:
        run_one(m)


if __name__ == "__main__":
    main()
