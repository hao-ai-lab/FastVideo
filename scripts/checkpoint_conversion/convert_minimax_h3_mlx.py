# SPDX-License-Identifier: Apache-2.0
"""Convert the FastH3 Preview DiT into pre-quantized MLX checkpoints.

Each output dir is a self-contained, shippable artifact in the
`mlx_h3_dit.safetensors` format: weights already cast/quantized, optional
AdaLN tables dropped at load time by the runtime. Conversion streams source
weights so the full BF16 DiT is not resident at once.

Usage (on the target Mac):

    python scripts/checkpoint_conversion/convert_minimax_h3_mlx.py \\
        --model-root ~/models/FastH3-Preview-v0.2/transformer \\
        --out ~/models/FastH3-MLX \\
        --formats "int8 int6 int4"

Dense conversion drops the 50 ``attn.to_gate_compress`` matrices (~3.6 GiB BF16).
Add ``--include-vsa`` to keep them, quantize them with the selected affine
INT8/INT6/INT4 grid, and record ``vsa.capable`` in the manifest. Write VSA
checkpoints to a new directory — do not overwrite an existing dense export.

    python scripts/checkpoint_conversion/convert_minimax_h3_mlx.py \\
        --model-root ~/models/FastH3-Preview-v0.2/transformer \\
        --out ~/models/FastH3-MLX-vsa \\
        --formats int6 --include-vsa

Output layout: `~/models/FastH3-MLX/<format>/mlx_h3_dit.safetensors` + manifest.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from fastvideo.logger import init_logger
from fastvideo.mlx_runtime.fastwan import MLXQuantizationSpec, ensure_quantization_supported
from fastvideo.mlx_runtime.minimax_h3 import (
    H3_MANIFEST_FILENAME,
    H3_WEIGHTS_FILENAME,
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_VIDEO_SHIFT,
    mlx_h3_checkpoint_vsa_capable,
    mlx_h3_dit_from_diffusers_safetensors,
    minimax_h3_sigmas,
    save_mlx_h3_checkpoint,
)

logger = init_logger(__name__)

SUPPORTED_FORMATS = ("int8", "int6", "int4")
DEFAULT_FORMATS = " ".join(SUPPORTED_FORMATS)


def _adaln_cache_timesteps() -> np.ndarray:
    video = 1.0 - minimax_h3_sigmas(MINIMAX_H3_VIDEO_SHIFT, 4)[:-1]
    audio = 1.0 - minimax_h3_sigmas(MINIMAX_H3_AUDIO_SHIFT, 4)[:-1]
    return np.unique(np.concatenate([video, audio, [1.0]])).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, help="transformer/ dir of the diffusers snapshot")
    parser.add_argument("--out", required=True, help="base dir for the per-format MLX checkpoints")
    parser.add_argument("--formats", default=DEFAULT_FORMATS)
    parser.add_argument(
        "--include-vsa",
        action="store_true",
        help=("retain and quantize transformer_blocks.*.attn.to_gate_compress.weight "
              "(required for MLX VSA inference; omitted by dense conversion)"),
    )
    return parser.parse_args()


def main() -> None:
    import mlx.core as mx

    args = parse_args()
    formats = args.formats.split()
    unsupported = sorted(set(formats) - set(SUPPORTED_FORMATS))
    if unsupported:
        raise ValueError(f"Unsupported H3 MLX formats: {unsupported}. Choose from {SUPPORTED_FORMATS}.")
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)
    cache_timesteps = _adaln_cache_timesteps()

    for fmt in formats:
        spec = MLXQuantizationSpec.from_name(fmt)
        if spec is None:
            raise ValueError(f"Expected a quantized H3 format, got {fmt}.")
        ensure_quantization_supported(spec)
        dtype = "bf16"
        out_dir = out_base / fmt
        already = (out_dir / H3_MANIFEST_FILENAME).exists() and (out_dir / H3_WEIGHTS_FILENAME).exists()
        if already:
            capable = mlx_h3_checkpoint_vsa_capable(out_dir)
            if bool(capable) == bool(args.include_vsa):
                print(f"[skip] {fmt} already converted at {out_dir} (vsa.capable={capable})", flush=True)
                continue
            logger.warning(
                "Skipping %s: existing checkpoint has vsa.capable=%s, requested include_vsa=%s. "
                "Use a new output directory to convert this format in the requested mode.",
                out_dir, capable, args.include_vsa,
            )
            continue
        print(f"[convert] {fmt} (dtype={dtype}, spec={spec}, include_vsa={args.include_vsa})", flush=True)
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        t0 = time.perf_counter()
        dit = mlx_h3_dit_from_diffusers_safetensors(
            args.model_root,
            quantization=spec,
            dtype=dtype,
            adaln_cache_timesteps=cache_timesteps,
            include_vsa=args.include_vsa,
        )
        mx.eval()
        save_mlx_h3_checkpoint(dit, out_dir)
        peak_gib = float(getattr(mx, "get_peak_memory", lambda: 0)()) / 2**30
        print(f"[done] {fmt} in {time.perf_counter() - t0:.1f}s | peak {peak_gib:.1f} GiB -> {out_dir}", flush=True)
        del dit
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()


if __name__ == "__main__":
    main()
