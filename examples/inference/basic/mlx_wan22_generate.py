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
    def first_or_empty(pattern: str) -> Path:
        matches = glob.glob(pattern)
        return Path(matches[0]) if matches else Path()

    fw21 = first_or_empty(
        str(Path.home() / ".cache/huggingface/hub/models--FastVideo--FastWan2.1-T2V-1.3B-Diffusers/snapshots/*"))
    wan22 = first_or_empty(
        str(Path.home() /
            ".cache/huggingface/hub/models--FastVideo--FastWan2.2-TI2V-5B-FullAttn-Diffusers/snapshots/*"))
    dit_root = Path.home() / "models" / "fastwan22_5b" / "transformer"
    return fw21, wan22, dit_root


def main() -> None:
    fw21_default, wan22_default, dit_default = _default_paths()
    parser = argparse.ArgumentParser(description="MLX Wan2.2-5B T2V (encode → DiT DMD → TAEHV/VAE decode)")
    parser.add_argument("--prompt", default="A red fox trotting through a snowy pine forest at golden hour, cinematic")
    parser.add_argument("--output-path", type=Path, default=Path("video_samples/demo_5b/fox_5b_mlx.mp4"))
    parser.add_argument("--text-encoder-root", type=Path, default=fw21_default, help="Root with text_encoder/ + tokenizer/")
    parser.add_argument("--prompt-embeds-cache", type=Path, default=None,
                        help="Optional .npy UMT5 embedding cache for repeat generation.")
    parser.add_argument("--text-encoder-device", choices=("auto", "cpu", "mps"), default="cpu",
                        help="Device for UMT5 encoding. CPU is the safest default beside the 5B MLX DiT.")
    parser.add_argument("--enhance-prompt", action="store_true",
                        help="Apply deterministic local cinematic prompt enrichment before UMT5 encoding.")
    parser.add_argument("--enhance-prompt-backend", choices=("template",), default="template",
                        help="Prompt enrichment backend (template is local, deterministic, and dependency-free).")
    parser.add_argument("--dit-checkpoint", type=Path, default=dit_default / "diffusion_pytorch_model.safetensors")
    parser.add_argument("--dit-config", type=Path, default=dit_default / "config.json")
    parser.add_argument(
        "--mlx-checkpoint",
        type=Path,
        default=None,
        help="Pre-quantized MLX DiT checkpoint directory. Rewrapped with Wan2.2 per-token conditioning.",
    )
    parser.add_argument("--vae-root", type=Path, default=wan22_default / "vae")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=121, help="Pixel frames (121 at 24fps = 5.04 seconds)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--renoise-seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--dmd-denoising-steps", default="1000,757,522")
    parser.add_argument("--no-warp", action="store_true", help="Disable schedule warping (debug only).")
    parser.add_argument("--fast", action="store_true", help="Generate fewer frames then RIFE-interpolate to --num-frames.")
    parser.add_argument("--fast-factor", type=int, default=2)
    parser.add_argument("--fast-sharpen", type=float, default=0.6)
    parser.add_argument("--fast-spatial", action="store_true", help="Denoise at reduced spatial resolution then upsample latents.")
    parser.add_argument("--fast-spatial-scale", type=int, default=2)
    parser.add_argument("--fast-spatial-upsample-mode", choices=("bilinear", "nearest"), default="bilinear")
    parser.add_argument("--refine", action="store_true", help="Two-pass DMD: coarse denoise, upsample/re-noise, full-resolution denoise.")
    parser.add_argument("--refine-scale", type=int, default=2)
    parser.add_argument("--refine-upsample-mode", choices=("bilinear", "nearest"), default="bilinear")
    parser.add_argument("--no-refine-add-noise", action="store_true")
    parser.add_argument("--decode-backend", choices=("taehv", "taehv-torch", "wan-vae"), default="taehv")
    parser.add_argument("--save-latents", type=Path, default=None)
    parser.add_argument("--metrics-json", type=Path, default=None,
                        help="Write measured run metadata as JSON for reports or galleries.")
    parser.add_argument("--compile", action="store_true", help="Attempt mx.compile on DiT (experimental).")
    args = parser.parse_args()

    if args.fast_factor < 2:
        parser.error("--fast-factor must be at least 2")
    if args.fast_spatial:
        raise SystemExit(
            "--fast-spatial is disabled for Wan2.2-TI2V-5B: bilinearly upsampling "
            "a completed 48-channel Wan latent is out of distribution and produces "
            "black or noisy video. Use --refine (optionally with --fast) so the "
            "upsampled latent receives a valid high-resolution DMD denoise pass."
        )
    if args.refine and args.fast_spatial:
        print("[wan22] --refine takes precedence over --fast-spatial")
    target_frames = args.num_frames
    if args.fast:
        args.num_frames = (target_frames + args.fast_factor - 1) // args.fast_factor
        print(f"[wan22 fast] generating {args.num_frames} frames, RIFE {args.fast_factor}x -> {target_frames}")

    import mlx.core as mx
    import torch

    from examples.inference.basic.mlx_wan_prompt_to_video import (
        _rife_interpolate_video,
        encode_prompt,
        make_rotary_embeddings,
    )
    from fastvideo.mlx_runtime.fast_spatial import apply_fast_spatial_upsample, plan_fast_spatial
    from fastvideo.mlx_runtime.refine import plan_refine_resolutions, prepare_refine_latents
    from fastvideo.mlx_runtime.wan22 import (
        mlx_wan22_dit_from_diffusers_safetensors,
        mlx_wan22_dit_from_mlx_checkpoint,
    )
    from fastvideo.mlx_runtime.wan22_sample import build_wan22_dmd_schedule, sample_wan22_dmd
    from fastvideo.mlx_runtime.wan_vae import decode_latents_to_video

    if args.mlx_checkpoint is not None:
        config = json.loads((args.mlx_checkpoint / "mlx_dit.json").read_text())["config"]
    else:
        config = json.loads(args.dit_config.read_text())
    patch_size = tuple(config.get("patch_size", (1, 2, 2)))
    if args.refine:
        active_plan = plan_refine_resolutions(
            height=args.height, width=args.width, num_frames=args.num_frames,
            spatial_scale=args.refine_scale, vae_spatial_compression=16,
            vae_temporal_compression=4, patch_size=patch_size, enabled=True,
        )
        spatial_mode = "refine"
    elif args.fast_spatial:
        fast_spatial_plan = plan_fast_spatial(
            height=args.height, width=args.width, num_frames=args.num_frames,
            spatial_scale=args.fast_spatial_scale, vae_spatial_compression=16,
            vae_temporal_compression=4, patch_size=patch_size,
            upsample_mode=args.fast_spatial_upsample_mode, enabled=True,
        )
        active_plan = fast_spatial_plan.plan
        spatial_mode = "fast_spatial"
    else:
        active_plan = plan_refine_resolutions(
            height=args.height, width=args.width, num_frames=args.num_frames,
            spatial_scale=1, vae_spatial_compression=16, vae_temporal_compression=4,
            patch_size=patch_size, enabled=False,
        )
        spatial_mode = "off"
    lat_h, lat_w = active_plan.stage1_latent_height, active_plan.stage1_latent_width
    lat_t = active_plan.latent_frames
    in_ch = int(config["in_channels"])
    print(f"[5B] latent {in_ch}x{lat_t}x{lat_h}x{lat_w}", flush=True)

    total_start = time.perf_counter()
    prompt_for_encode = args.prompt
    enhance_backend = None
    enhance_elapsed_s = 0.0
    if args.enhance_prompt:
        from fastvideo.mlx_runtime.prompt_enhance import enhance_prompt

        enhancement = enhance_prompt(args.prompt, backend=args.enhance_prompt_backend)
        prompt_for_encode = enhancement.enhanced
        enhance_backend = enhancement.backend
        enhance_elapsed_s = enhancement.elapsed_s
        print(f"[enhance] backend={enhance_backend} in {enhance_elapsed_s:.2f}s", flush=True)
        print(f"[enhance] prompt: {prompt_for_encode}", flush=True)

    t0 = time.perf_counter()
    if args.prompt_embeds_cache is not None and args.prompt_embeds_cache.exists():
        embeds = torch.from_numpy(np.load(args.prompt_embeds_cache)).contiguous()
    else:
        embeds = encode_prompt(
            model_root=args.text_encoder_root,
            prompt=prompt_for_encode,
            max_sequence_length=512,
            device_arg=args.text_encoder_device,
            dtype_arg="fp16",
        )
        if args.prompt_embeds_cache is not None:
            args.prompt_embeds_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.prompt_embeds_cache, embeds.cpu().numpy())
    ehs = mx.array(embeds.numpy()).astype(mx.float16)
    prompt_encode_s = time.perf_counter() - t0
    print(f"[5B] prompt encoded {tuple(ehs.shape)} in {prompt_encode_s:.1f}s", flush=True)

    t1 = time.perf_counter()
    if args.mlx_checkpoint is not None:
        dit = mlx_wan22_dit_from_mlx_checkpoint(args.mlx_checkpoint)
    else:
        dit = mlx_wan22_dit_from_diffusers_safetensors(args.dit_checkpoint, args.dit_config, dtype="fp16")
    if args.compile:
        # Experimental: compile a pure function wrapper if shapes are static.
        print("[5B] note: mx.compile on full DiT is experimental; skipping if unsupported", flush=True)
    dit_load_s = time.perf_counter() - t1
    print(f"[5B] DiT loaded in {dit_load_s:.1f}s", flush=True)

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
    if spatial_mode == "refine":
        schedule, warped_steps = build_wan22_dmd_schedule(
            steps, flow_shift=args.flow_shift, warp_denoising_step=not args.no_warp,
        )
        sigma = schedule.sigma_for(warped_steps[0])
        latents = prepare_refine_latents(
            latents, scale=args.refine_scale, sigma=sigma,
            add_noise_flag=not args.no_refine_add_noise,
            upsample_mode=args.refine_upsample_mode, seed=args.renoise_seed + 1,
        )
        freqs_stage2 = make_rotary_embeddings(
            config, latent_frames=lat_t,
            latent_height=active_plan.stage2_latent_height,
            latent_width=active_plan.stage2_latent_width,
        )
        latents = sample_wan22_dmd(
            dit, ehs, latents, freqs_stage2, dmd_denoising_steps=steps,
            flow_shift=args.flow_shift, warp_denoising_step=not args.no_warp,
            seed=args.renoise_seed + 2,
        )
    elif spatial_mode == "fast_spatial":
        latents = apply_fast_spatial_upsample(latents, fast_spatial_plan)
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
    rife_s = 0.0
    if args.fast:
        rife_start = time.perf_counter()
        _rife_interpolate_video(
            video_path=args.output_path, target_frames=target_frames,
            factor=args.fast_factor, sharpen=args.fast_sharpen, fps=args.fps,
        )
        rife_s = time.perf_counter() - rife_start
    print(f"[5B] decoded via {metrics['backend']} in {metrics['decode_s']:.1f}s → {args.output_path}", flush=True)
    summary = {
        "output_path": str(args.output_path.resolve()),
        "prompt": args.prompt,
        "prompt_used": prompt_for_encode,
        "enhance_prompt": args.enhance_prompt,
        "enhance_backend": enhance_backend,
        "enhance_elapsed_s": round(enhance_elapsed_s, 3),
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "target_frames": target_frames,
        "generated_frames": args.num_frames,
        "seed": args.seed,
        "renoise_seed": args.renoise_seed,
        "dmd_denoising_steps": steps,
        "flow_shift": args.flow_shift,
        "warp": not args.no_warp,
        "spatial_mode": spatial_mode,
        "fast": args.fast,
        "fast_factor": args.fast_factor if args.fast else None,
        "fast_spatial_scale": args.fast_spatial_scale if args.fast_spatial else None,
        "refine_scale": args.refine_scale if args.refine else None,
        "decode_backend": args.decode_backend,
        "prompt_encode_s": round(prompt_encode_s, 3),
        "dit_load_s": round(dit_load_s, 3),
        "denoise_s": round(denoise_s, 3),
        "decode_s": round(metrics["decode_s"], 3),
        "rife_s": round(rife_s, 3),
        "wall_total_s": round(time.perf_counter() - total_start, 3),
        "peak_gib": round(peak, 3),
        "latent_shape": [in_ch, lat_t, lat_h, lat_w],
        "stage2_latent_shape": [in_ch, lat_t, active_plan.stage2_latent_height, active_plan.stage2_latent_width],
        "mlx_checkpoint": str(args.mlx_checkpoint.resolve()) if args.mlx_checkpoint else None,
    }
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[5B] wrote metrics {args.metrics_json}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
