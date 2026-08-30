# SPDX-License-Identifier: Apache-2.0
"""End-to-end MiniMax-H3 (FastH3) generation with the Apple Silicon MLX runtime.

Accepts a text prompt and produces an MP4 with H.264 video at 24 fps and
stereo AAC audio at 32 kHz. One heavyweight model phase is resident at a time.

    python examples/inference/basic/mlx_fasth3.py \
      --model-root ~/models/FastH3-Preview-v0.2 \
      --mlx-checkpoint ~/models/FastH3-MLX/int8 \
      --prompt '(S1) A red panda says <d>[English] Fast H3 is amazing.</d>' \
      --height 480 --width 832 --num-frames 124 --seed 2026 \
      --output-path ~/fasth3_outputs/int8.mp4

Conditioning uses the streamed Qwen3-VL text encoder on first use and caches
the resulting embeddings under --prompt-cache-dir for instant reuse.

``--fast`` is temporal fast mode. It keeps full-duration audio while
denoising fewer video frames, then uses MLX RIFE 4.25 to reconstruct the
requested frame count. A 1280x720 request runs on H3's 1280x736 grid and is
center-cropped after decode.

This entrypoint currently supports text-to-video-with-audio only. It does not
yet wire FL2VA, Ref2VA, spatial fast mode, or two-pass refinement.

VSA is off by default; existing dense MLX checkpoints remain supported.
H3 uses fused MLX RMSNorm, which can change BF16 rounding compared with the
older explicit normalization path. Convert with ``--include-vsa`` and pass
``--vsa`` to enable the sparse path. Attention activations stay BF16; INT6/INT8/INT4 apply only to
linear weights, including the optional gate projection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _dense_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError("dense layer indices must be non-negative")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-root", type=Path, default=Path.home() / "models/FastH3-Preview-v0.2",
                        help="H3 snapshot root (vae/, audio_vae/, text_encoder/, tokenizer/)")
    parser.add_argument("--mlx-checkpoint", type=Path, required=True,
                        help="pre-quantized MLX DiT directory (int8/int6/int4 mlx_h3_dit format)")
    parser.add_argument(
        "--prompt",
        required=True,
        help="H3 text prompt; use (S1) and <d>[Language] words</d> for explicit dialogue",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4, help="denoise steps (trained ladder = 4)")
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="denoise fewer video frames, then use MLX RIFE to restore the target frame count; audio stays full length",
    )
    parser.add_argument("--fast-factor", type=int, default=2,
                        help="temporal reduction target for --fast (default: 2)")
    parser.add_argument("--fast-sharpen", type=float, default=0.6,
                        help="unsharp strength after RIFE interpolation (0 disables)")
    parser.add_argument("--rife-weights-dir", type=Path, default=None,
                        help="optional local mlx-community/RIFE-4.25 snapshot")
    parser.add_argument("--vae-dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--prompt-cache-dir", type=Path, default=None,
                        help="directory for reusable prompt embedding caches")
    parser.add_argument(
        "--tiled-video-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="decode with the reference 256px overlapping VAE tiles (disable only for diagnostics)",
    )
    parser.add_argument(
        "--vsa",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "enable MiniMax H3 VSA; requires a VSA-capable MLX checkpoint "
            "from --include-vsa"
        ),
    )
    parser.add_argument(
        "--vsa-sparsity",
        type=float,
        default=0.9,
        help="VSA sparsity in [0, 1); 0.9 is the trained FastH3 policy",
    )
    parser.add_argument(
        "--vsa-tile-size",
        type=int,
        default=64,
        choices=(64, 256),
        help="VSA tile size in tokens",
    )
    parser.add_argument(
        "--vsa-prefix-mode",
        choices=("exempt", "compete"),
        default="exempt",
        help=(
            "prefix-key policy: always keep (exempt) or FLOP-matched top-k "
            "(compete)"
        ),
    )
    parser.add_argument(
        "--vsa-dense-first-n-steps",
        type=int,
        default=0,
        help="run the first N denoise steps dense",
    )
    parser.add_argument(
        "--vsa-dense-layers",
        type=_dense_layers,
        default=(),
        help="comma-separated layer indices forced dense",
    )
    parser.add_argument(
        "--vsa-impl",
        choices=("auto", "reference", "simd"),
        default="auto",
        help=(
            "sparse attention implementation; auto uses chunked gather+SDPA "
            "(simd is opt-in)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from fastvideo.mlx_runtime.minimax_h3_pipeline import MiniMaxH3MLXPipeline

    pipeline = MiniMaxH3MLXPipeline(
        model_root=args.model_root,
        mlx_dit_checkpoint=args.mlx_checkpoint,
        vae_dtype=args.vae_dtype,
        prompt_cache_dir=args.prompt_cache_dir,
    )
    result = pipeline.generate(
        args.prompt,
        output_path=args.output_path,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        num_steps=args.steps,
        tiled_video_decode=args.tiled_video_decode,
        fast=args.fast,
        fast_factor=args.fast_factor,
        fast_sharpen=args.fast_sharpen,
        rife_weights_dir=args.rife_weights_dir,
        vsa=args.vsa,
        vsa_sparsity=args.vsa_sparsity,
        vsa_tile_size=args.vsa_tile_size,
        vsa_prefix_mode=args.vsa_prefix_mode,
        vsa_dense_first_n_steps=args.vsa_dense_first_n_steps,
        vsa_dense_layers=args.vsa_dense_layers,
        vsa_impl=args.vsa_impl,
    )
    print(json.dumps({
        "video_path": result.video_path,
        "timings_s": {k: round(v, 2) for k, v in result.timings.items()},
        "peak_memory_gib": {k: round(v, 2) for k, v in result.peak_memory_gib.items()},
        "vsa": result.vsa,
        "audio_samples": int(result.waveform.shape[-1]),
    }, indent=2))


if __name__ == "__main__":
    main()
