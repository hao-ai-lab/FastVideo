# SPDX-License-Identifier: Apache-2.0
"""End-to-end Wan2.2-TI2V-5B generation on Apple Silicon (MLX DiT + MLX TAEHV).

Pipeline: torch/MPS UMT5 encode (shared with 1.3B) → MLXWan22DiT 3-step DMD
(warped schedule, flow_shift=5) → MLX TAEHV decode (taew2_2.pth). Fully MLX
on the heavy DiT + decode path.

    PYTHONPATH=$PWD python examples/inference/basic/mlx_wan22_generate.py \
      --prompt "A red fox trotting through a snowy pine forest at golden hour" \
      --output-path video_samples/demo_5b/fox_5b_mlx.mp4

Decoder backends: ``taehv`` (default, MLX, ~seconds), ``taehv-torch`` (parity),
``wan-vae`` (full AutoencoderKLWan on MPS, slow).
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np


def _default_paths() -> tuple[Path, Path, Path]:
    fw21 = Path(
        glob.glob(str(Path.home() / ".cache/huggingface/hub/models--FastVideo--FastWan2.1-T2V-1.3B-Diffusers/snapshots/*"))
        [0])
    wan22 = Path(
        glob.glob(
            str(Path.home() /
                ".cache/huggingface/hub/models--FastVideo--FastWan2.2-TI2V-5B-FullAttn-Diffusers/snapshots/*"))[0])
    dit_root = Path.home() / "models" / "fastwan22_5b" / "transformer"
    return fw21, wan22, dit_root


def main() -> None:
    fw21_default, wan22_default, dit_default = _default_paths()
    parser = argparse.ArgumentParser(description="MLX Wan2.2-5B T2V (encode → DiT DMD → TAEHV/VAE decode)")
    parser.add_argument("--prompt", default="A red fox trotting through a snowy pine forest at golden hour, cinematic")
    parser.add_argument("--output-path", type=Path, default=Path("video_samples/demo_5b/fox_5b_mlx.mp4"))
    parser.add_argument("--text-encoder-root", type=Path, default=fw21_default, help="Root with text_encoder/ + tokenizer/")
    parser.add_argument("--dit-checkpoint", type=Path, default=dit_default / "diffusion_pytorch_model.safetensors")
    parser.add_argument("--dit-config", type=Path, default=dit_default / "config.json")
    parser.add_argument("--vae-root", type=Path, default=wan22_default / "vae")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81, help="Pixel frames (latent T = (F-1)//4+1)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--renoise-seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--dmd-denoising-steps", default="1000,757,522")
    parser.add_argument("--no-warp", action="store_true", help="Disable schedule warping (debug only).")
    parser.add_argument("--decode-backend", choices=("taehv", "taehv-torch", "wan-vae"), default="taehv")
    parser.add_argument("--save-latents", type=Path, default=None)
    parser.add_argument("--compile", action="store_true", help="Attempt mx.compile on DiT (experimental).")
    args = parser.parse_args()

    import mlx.core as mx
    import torch

    from examples.inference.basic.mlx_wan_prompt_to_video import encode_prompt, make_rotary_embeddings
    from fastvideo.mlx_runtime.wan22 import mlx_wan22_dit_from_diffusers_safetensors
    from fastvideo.mlx_runtime.wan22_sample import sample_wan22_dmd
    from fastvideo.mlx_runtime.wan_vae import decode_latents_to_video

    config = json.loads(args.dit_config.read_text())
    # Wan2.2 spatial /16, temporal /4
    lat_h, lat_w = args.height // 16, args.width // 16
    lat_t = (args.num_frames - 1) // 4 + 1
    in_ch = int(config["in_channels"])
    print(f"[5B] latent {in_ch}x{lat_t}x{lat_h}x{lat_w}", flush=True)

    t0 = time.perf_counter()
    embeds = encode_prompt(
        model_root=args.text_encoder_root,
        prompt=args.prompt,
        max_sequence_length=512,
        device_arg="auto",
        dtype_arg="fp16",
    )
    ehs = mx.array(embeds.numpy()).astype(mx.float16)
    print(f"[5B] prompt encoded {tuple(ehs.shape)} in {time.perf_counter() - t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    dit = mlx_wan22_dit_from_diffusers_safetensors(args.dit_checkpoint, args.dit_config, dtype="fp16")
    if args.compile:
        # Experimental: compile a pure function wrapper if shapes are static.
        print("[5B] note: mx.compile on full DiT is experimental; skipping if unsupported", flush=True)
    print(f"[5B] DiT loaded in {time.perf_counter() - t1:.1f}s", flush=True)

    freqs = make_rotary_embeddings(config, latent_frames=lat_t, latent_height=lat_h, latent_width=lat_w)
    gen = torch.Generator().manual_seed(args.seed)
    noise = mx.array(
        torch.randn(1, in_ch, lat_t, lat_h, lat_w, generator=gen, dtype=torch.float32).numpy()).astype(mx.float16)

    steps = [int(s) for s in args.dmd_denoising_steps.split(",") if s.strip()]
    t2 = time.perf_counter()
    mx.reset_peak_memory()
    latents = sample_wan22_dmd(
        dit,
        ehs,
        noise,
        freqs,
        dmd_denoising_steps=steps,
        flow_shift=args.flow_shift,
        warp_denoising_step=not args.no_warp,
        seed=args.renoise_seed,
    )
    denoise_s = time.perf_counter() - t2
    peak = mx.get_peak_memory() / (1024**3)
    print(f"[5B] denoise {len(steps)} steps in {denoise_s:.1f}s, peak {peak:.2f} GiB", flush=True)

    latents_np = np.array(latents.astype(mx.float32))
    if args.save_latents is not None:
        args.save_latents.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_latents, latents=latents_np, prompt=args.prompt, seed=args.seed)
        print(f"[5B] wrote latents {args.save_latents}", flush=True)

    metrics = decode_latents_to_video(
        latents_np,
        args.output_path,
        fps=args.fps,
        backend=args.decode_backend,
        vae_dir=args.vae_root if args.decode_backend == "wan-vae" else None,
        z_dim=in_ch,
    )
    print(f"[5B] decoded via {metrics['backend']} in {metrics['decode_s']:.1f}s → {args.output_path}", flush=True)
    print(
        json.dumps({
            "denoise_s": round(denoise_s, 2),
            "decode_s": round(metrics["decode_s"], 2),
            "peak_gib": round(peak, 3),
            "warp": not args.no_warp,
            "decode_backend": args.decode_backend,
        },
                   indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
