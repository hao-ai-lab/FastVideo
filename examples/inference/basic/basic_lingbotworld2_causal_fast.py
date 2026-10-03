# SPDX-License-Identifier: Apache-2.0
"""Run LingBot World 2 14B causal-fast I2V generation with FastVideo."""

import os
from pathlib import Path

from fastvideo import VideoGenerator

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "examples" / "dataset" / "lingbotworld2"
OUTPUT_PATH = REPO_ROOT / "outputs" / "lingbotworld2_causal_fast.mp4"


def main() -> None:
    """Load the native FastVideo LingBot World 2 causal-fast pipeline and generate one video."""
    generator = VideoGenerator.from_config({
        "model_path": os.environ["LINGBOTWORLD2_MODEL_PATH"],
        "engine": {
            "num_gpus": 8,
            "parallelism": {"sp_size": 8, "hsdp_shard_dim": 8},
            "use_fsdp_inference": True,
            "offload": {
                "dit_layerwise": False,
                "dit": False,
                "vae": False,
                "text_encoder": False,
                "pin_cpu_memory": True,
            },
        },
        "pipeline": {"components": {"override_pipeline_cls_name": "LingBotWorld2CausalFastPipeline"}},
    })

    try:
        generator.generate({
            "prompt":
            "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds; gentle ripples reflect the tree and sky, creating a tranquil, meditative atmosphere.",
            "negative_prompt": "",
            "inputs": {
                "image_path": str(DATASET_DIR / "image.jpg"),
                "action_path": str(DATASET_DIR),
            },
            "sampling": {
                "height": 480,
                "width": 832,
                "num_frames": 65,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
                "fps": 16,
                "seed": 42,
            },
            "output": {"output_path": str(OUTPUT_PATH), "save_video": True},
        })
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
