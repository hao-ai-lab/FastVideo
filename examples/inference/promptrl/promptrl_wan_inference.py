# SPDX-License-Identifier: Apache-2.0
"""PromptRL inference: refine a prompt, then generate with the Wan LoRA.

Explicit two-stage flow (VideoGenerator behavior is unchanged — the
refiner is applied by the caller, not inside the generator):

1. ``PromptRefiner.from_bundle(bundle).refine(prompt)`` loads the
   PromptRL refiner adapter and returns the refined prompt (falling
   back to the original when the completion misses <answer>...</answer>).
2. The exported Wan LoRA loads through the existing generator
   configuration (``ComponentConfig.lora_path``).
3. ``VideoGenerator.generate(...)`` runs the typed request API.

Usage::

    python examples/inference/promptrl/promptrl_wan_inference.py \
        --bundle outputs/wan2.1_promptrl_joint/bundle \
        --prompt "a cat riding a skateboard" \
        --output outputs/promptrl/skateboard.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)
from fastvideo.api.schema import ComponentConfig
from fastvideo.train.methods.rl.promptrl import PromptRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PromptRL Wan inference")
    parser.add_argument("--bundle", required=True, help="PromptRL bundle directory")
    parser.add_argument("--model-path", default=None,
                        help="Wan base model; defaults to the bundle manifest value")
    parser.add_argument("--prompt", default="a cat riding a skateboard")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", default="outputs/promptrl/promptrl_wan.mp4")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=77)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refiner-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 1. Refine the prompt with the bundled PromptRL refiner.
    refiner = PromptRefiner.from_bundle(args.bundle, device="cuda")
    model_path = args.model_path or refiner.manifest.base_generator_model
    refinement = refiner.refine(args.prompt, seed=args.refiner_seed)
    print(f"[promptrl] original : {refinement.original_prompt}")
    print(f"[promptrl] refined  : {refinement.refined_prompt} "
          f"(format_valid={refinement.format_valid})")

    # 2. Load the exported Wan LoRA through the existing generator config.
    generator_lora = os.path.join(args.bundle, "generator", "promptrl_generator_lora.safetensors")
    lora_path = generator_lora if os.path.isfile(generator_lora) else None
    if lora_path is None:
        print("[promptrl] no generator LoRA in bundle; running the base Wan model")
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=model_path,
            engine=EngineConfig(
                num_gpus=1,
                parallelism=ParallelismConfig(tp_size=1, sp_size=1),
                use_fsdp_inference=False,
            ),
            pipeline=PipelineSelection(
                workload_type="t2v",
                components=ComponentConfig(lora_path=lora_path),
            ),
        ))
    try:
        # 3. Typed request API with the refined prompt.
        generator.generate(
            GenerationRequest(
                prompt=refinement.refined_prompt,
                negative_prompt=args.negative_prompt,
                sampling=SamplingConfig(
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                ),
                output=OutputConfig(
                    output_path=str(output),
                    save_video=True,
                    return_frames=False,
                ),
            ))
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
