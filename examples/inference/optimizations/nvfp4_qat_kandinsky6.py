# SPDX-License-Identifier: Apache-2.0
"""Generate video and audio from a Kandinsky 6 pi-Flow checkpoint with packed NVFP4 weights."""

import argparse
import os
from pathlib import Path


def resolve_model_path(model_path: str) -> str:
    from fastvideo.layers.quantization.nvfp4_config import H3_NVFP4_DIT_EXPORT_FILENAME
    from fastvideo.pipelines.basic.kandinsky6.kandinsky6_ti2va_pipeline import (
        NVFP4_TRANSFORMER_DIR,
        Kandinsky6TI2VAPipeline,
    )
    from fastvideo.utils import _split_hf_repo_subfolder, maybe_download_model

    local = Path(model_path).expanduser()
    if local.is_dir():
        root = local.parent if local.name == "nvfp4" else local
    else:
        repo_id, subfolder = _split_hf_repo_subfolder(model_path.rstrip("/"))
        if subfolder not in (None, "nvfp4"):
            raise ValueError("Pass the Kandinsky 6 Hub model ID or its nvfp4 subfolder.")
        root = Path(maybe_download_model(
            repo_id,
            allow_patterns=[
                "model_index.json",
                *(f"{directory}/**" for directory in Kandinsky6TI2VAPipeline.nvfp4_component_dirs()),
            ],
        ))

    packed_file = root / NVFP4_TRANSFORMER_DIR / H3_NVFP4_DIT_EXPORT_FILENAME
    if not packed_file.is_file():
        raise FileNotFoundError(f"Packed transformer file not found: {packed_file}")
    return str(root.absolute())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local NVFP4 export or Hugging Face model ID")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-path", default="outputs_video/kandinsky6_nvfp4")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=575431)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    os.environ.pop("FASTVIDEO_NVFP4_ACT_AMAX", None)
    os.environ.pop("FASTVIDEO_NVFP4_DYNAMIC_ACT", None)
    os.environ["FASTVIDEO_LAYERWISE_OFFLOAD_BUFFERS"] = "1"

    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_config({
        "model_path": resolve_model_path(args.model_path),
        "engine": {
            "quantization": {"transformer_quant": "NVFP4", "layer_profile": "kandinsky6"},
            "offload": {"dit": False, "dit_layerwise": True, "text_encoder": True, "vae": True},
            "compile": {"enabled": True},
        },
        "pipeline": {
            "experimental": {"attention_backend": "ATTN_QAT_INFER"},
        },
    })
    try:
        result = generator.generate({
            "prompt": args.prompt,
            "sampling": {
                "height": 512, "width": 768, "num_frames": 121, "fps": 24,
                "num_inference_steps": args.steps, "guidance_scale": 1.0, "seed": args.seed,
            },
            "output": {"output_path": args.output_path, "save_video": True, "return_frames": False},
        })
        if not result.video_path or not Path(result.video_path).is_file():
            raise RuntimeError("FastVideo did not save the generated MP4.")
        print(f"Saved {result.video_path}")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
