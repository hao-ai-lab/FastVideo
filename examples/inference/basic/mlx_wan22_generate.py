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
import json
import time
from pathlib import Path

import numpy as np

from fastvideo.mlx_runtime.fast_spatial import DEFAULT_FAST_SPATIAL_SHARPEN
from fastvideo.mlx_runtime.frame_upsample import DEFAULT_PIXEL_UPSAMPLE_MODE, PIXEL_UPSAMPLE_MODES
from fastvideo.mlx_runtime.memory import cleanup_mlx
from fastvideo.mlx_runtime.prompt_cache import (
    fingerprint_digest,
    load_prompt_cache,
    save_prompt_cache,
    text_encoder_fingerprint,
)
from fastvideo.mlx_runtime.rife_interp import aligned_keyframe_count

FASTWAN21_MODEL_ID = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
FASTWAN22_MODEL_ID = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
DEFAULT_HEIGHT = 448
DEFAULT_WIDTH = 832
DEFAULT_NUM_FRAMES = 121

def _resolve_model_paths(
    *,
    text_encoder_root: Path | None,
    dit_checkpoint: Path | None,
    dit_config: Path | None,
    vae_root: Path | None,
    mlx_checkpoint: Path | None,
    decode_backend: str,
) -> tuple[Path, Path | None, Path | None, Path | None]:
    """Download only the missing assets required by the selected Wan2.2 path."""
    from huggingface_hub import snapshot_download

    if text_encoder_root is None:
        text_encoder_root = Path(snapshot_download(
            FASTWAN21_MODEL_ID,
            allow_patterns=["tokenizer/*", "text_encoder/*"],
        ))
    if mlx_checkpoint is None and (dit_checkpoint is None or dit_config is None):
        patterns = []
        if dit_checkpoint is None:
            patterns.append("transformer/diffusion_pytorch_model.safetensors")
        if dit_config is None:
            patterns.append("transformer/config.json")
        model_root = Path(snapshot_download(FASTWAN22_MODEL_ID, allow_patterns=patterns))
        dit_checkpoint = dit_checkpoint or model_root / "transformer/diffusion_pytorch_model.safetensors"
        dit_config = dit_config or model_root / "transformer/config.json"
    if decode_backend == "wan-vae" and vae_root is None:
        model_root = Path(snapshot_download(FASTWAN22_MODEL_ID, allow_patterns=["vae/*"]))
        vae_root = model_root / "vae"
    return text_encoder_root, dit_checkpoint, dit_config, vae_root


def _prompt_cache_fingerprint(
    *,
    prompt: str,
    prompt_used: str,
    enhance_prompt: bool,
    enhance_prompt_backend: str,
    text_encoder_root: Path,
    max_sequence_length: int,
    dtype: str,
) -> dict[str, object]:
    return {
        "prompt": prompt,
        "prompt_used": prompt_used,
        "enhance_prompt": enhance_prompt,
        "enhance_prompt_backend": enhance_prompt_backend,
        "text_encoder": text_encoder_fingerprint(text_encoder_root),
        "max_sequence_length": max_sequence_length,
        "dtype": dtype,
    }


def _default_prompt_cache_path(fingerprint: dict[str, object]) -> Path:
    """Content-addressed default cache file for a prompt fingerprint.

    The Wan2.1 entrypoint caches prompt embeddings by default; this one only
    did so when handed an explicit ``--prompt-embeds-cache`` path, so every 5B
    run paid a full UMT5 encode (~45s on an M4 Max) even for a repeat prompt.
    The fingerprint already covers everything that changes the embedding, so
    hash it for the filename.
    """
    digest = fingerprint_digest(fingerprint)[:32]
    return Path.home() / ".cache" / "fastvideo" / "prompt_embeds" / f"wan22_{digest}.npy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLX Wan2.2-5B T2V (encode → DiT DMD → TAEHV/VAE decode)"
    )
    parser.add_argument(
        "--prompt",
        default="A red fox trotting through a snowy pine forest at golden hour, cinematic",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("video_samples/demo_5b/fox_5b_mlx.mp4"),
    )
    parser.add_argument(
        "--text-encoder-root",
        type=Path,
        default=None,
        help="Root with text_encoder/ + tokenizer/",
    )
    parser.add_argument(
        "--prompt-embeds-cache",
        type=Path,
        default=None,
        help="Explicit .npy UMT5 embedding cache file. Overrides the automatic "
        "content-addressed cache (--prompt-cache).",
    )
    parser.add_argument(
        "--prompt-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache prompt embeddings under ~/.cache/fastvideo/prompt_embeds so "
        "repeat runs skip the text encoder entirely. Default: on.",
    )
    parser.add_argument(
        "--text-encoder-device",
        choices=("auto", "cpu", "mps"),
        default="cpu",
        help="Device for UMT5 encoding. CPU is safest beside the 5B MLX DiT.",
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Apply deterministic local cinematic prompt enrichment before UMT5.",
    )
    parser.add_argument(
        "--enhance-prompt-backend",
        choices=("template",),
        default="template",
        help="Prompt enrichment backend.",
    )
    parser.add_argument(
        "--dit-checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument("--dit-config", type=Path, default=None)
    parser.add_argument(
        "--mlx-checkpoint",
        type=Path,
        default=None,
        help="Pre-quantized MLX DiT checkpoint directory. Rewrapped with Wan2.2 per-token conditioning.",
    )
    parser.add_argument("--vae-root", type=Path, default=None)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Pixel frames (121 at 24fps = 5.04 seconds)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--renoise-seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--dmd-denoising-steps", default="1000,757,522")
    parser.add_argument(
        "--no-warp",
        action="store_true",
        help="Disable schedule warping (debug only).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Generate fewer frames then RIFE-interpolate to --num-frames.",
    )
    parser.add_argument("--fast-factor", type=int, default=2)
    parser.add_argument("--fast-sharpen", type=float, default=0.6)
    parser.add_argument(
        "--fast-spatial",
        action="store_true",
        help="Denoise and decode at reduced spatial resolution, then resample "
        "the decoded frames up to the target size.",
    )
    parser.add_argument("--fast-spatial-scale", type=int, default=2)
    parser.add_argument(
        "--fast-spatial-upsample-mode",
        choices=PIXEL_UPSAMPLE_MODES,
        default=DEFAULT_PIXEL_UPSAMPLE_MODE,
    )
    parser.add_argument("--fast-spatial-sharpen", type=float, default=DEFAULT_FAST_SPATIAL_SHARPEN)
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Two-pass DMD: coarse denoise, upsample/re-noise, full-res denoise.",
    )
    parser.add_argument("--refine-scale", type=int, default=2)
    parser.add_argument(
        "--refine-upsample-mode",
        choices=("bilinear", "nearest"),
        default="bilinear",
    )
    parser.add_argument("--no-refine-add-noise", action="store_true")
    parser.add_argument(
        "--decode-backend",
        choices=("taehv", "taehv-torch", "wan-vae"),
        default="taehv",
    )
    parser.add_argument("--save-latents", type=Path, default=None)
    parser.add_argument("--metrics-json", type=Path, default=None,
                        help="Write measured run metadata as JSON for reports or galleries.")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the DiT forward with mx.compile; fallback to eager on failure.",
    )
    args = parser.parse_args()

    if args.fast_factor < 2:
        parser.error("--fast-factor must be at least 2")
    # --fast-spatial used to be rejected here because it upsampled the completed
    # 48-channel latent, which is out of distribution for the decoder and gave
    # black or noisy video. The upsample now runs on decoded frames, so the
    # latent never leaves the grid it was denoised on and the mode is usable.
    if args.refine and args.fast_spatial:
        print("[wan22] --refine takes precedence over --fast-spatial")
    args.text_encoder_root, args.dit_checkpoint, args.dit_config, args.vae_root = _resolve_model_paths(
        text_encoder_root=args.text_encoder_root,
        dit_checkpoint=args.dit_checkpoint,
        dit_config=args.dit_config,
        vae_root=args.vae_root,
        mlx_checkpoint=args.mlx_checkpoint,
        decode_backend=args.decode_backend,
    )
    target_frames = args.num_frames
    if args.fast:
        args.num_frames = aligned_keyframe_count(target_frames, args.fast_factor)
        print(
            f"[wan22 fast] generating {args.num_frames} frames, "
            f"RIFE {args.fast_factor}x -> {target_frames}"
        )

    import mlx.core as mx
    import torch

    from examples.inference.basic.mlx_wan_prompt_to_video import (
        _postprocess_video,
        encode_prompt,
        make_rotary_embeddings,
    )
    from fastvideo.mlx_runtime.fast_spatial import plan_fast_spatial
    from fastvideo.mlx_runtime.refine import (
        default_refine_timesteps,
        plan_refine_resolutions,
        prepare_refine_latents,
    )
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
            upsample_mode=args.fast_spatial_upsample_mode,
            sharpen=args.fast_spatial_sharpen, enabled=True,
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
    prompt_cache_fingerprint = _prompt_cache_fingerprint(
        prompt=args.prompt,
        prompt_used=prompt_for_encode,
        enhance_prompt=args.enhance_prompt,
        enhance_prompt_backend=args.enhance_prompt_backend,
        text_encoder_root=args.text_encoder_root,
        max_sequence_length=512,
        dtype="fp16",
    )
    prompt_cache_path = args.prompt_embeds_cache
    if prompt_cache_path is None and args.prompt_cache:
        prompt_cache_path = _default_prompt_cache_path(prompt_cache_fingerprint)
    cached_embeds = load_prompt_cache(
        prompt_cache_path,
        prompt_cache_fingerprint,
    )
    if cached_embeds is not None:
        embeds = torch.from_numpy(cached_embeds).contiguous()
    else:
        embeds = encode_prompt(
            model_root=args.text_encoder_root,
            prompt=prompt_for_encode,
            max_sequence_length=512,
            device_arg=args.text_encoder_device,
            dtype_arg="fp16",
        )
        save_prompt_cache(
            prompt_cache_path,
            embeds.cpu().numpy(),
            prompt_cache_fingerprint,
        )
    ehs = mx.array(embeds.numpy()).astype(mx.float16)
    prompt_encode_s = time.perf_counter() - t0
    print(f"[5B] prompt encoded {tuple(ehs.shape)} in {prompt_encode_s:.1f}s", flush=True)

    t1 = time.perf_counter()
    if args.mlx_checkpoint is not None:
        dit = mlx_wan22_dit_from_mlx_checkpoint(
            args.mlx_checkpoint,
            compile=args.compile,
        )
    else:
        dit = mlx_wan22_dit_from_diffusers_safetensors(
            args.dit_checkpoint,
            args.dit_config,
            dtype="fp16",
            compile=args.compile,
        )
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
        # The grid opens at sigma == 1, where the hand-off
        # `(1 - sigma) * upsampled + sigma * noise` weights stage 1 at zero and
        # refine silently becomes a plain full-res run. Drop the leading
        # full-noise steps so stage 1 actually reaches stage 2.
        stage2_warped = default_refine_timesteps(schedule, warped_steps)
        stage2_steps = steps[len(warped_steps) - len(stage2_warped):]
        sigma = schedule.sigma_for(stage2_warped[0])
        print(f"[5B refine] stage-2 steps={stage2_steps} sigma={sigma:.4f} "
              f"(stage-1 weight {1.0 - sigma:.4f})", flush=True)
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
            dit, ehs, latents, freqs_stage2, dmd_denoising_steps=stage2_steps,
            flow_shift=args.flow_shift, warp_denoising_step=not args.no_warp,
            seed=args.renoise_seed + 2,
        )
    # spatial_mode == "fast_spatial" leaves the latents on the stage-1 grid;
    # the resample happens after decode, in _postprocess_video.
    denoise_s = time.perf_counter() - t2
    peak = mx.get_peak_memory() / (1024**3)
    print(f"[5B] denoise {len(steps)} steps in {denoise_s:.1f}s, peak {peak:.2f} GiB", flush=True)

    latents_np = np.array(latents.astype(mx.float32))
    if args.save_latents is not None:
        args.save_latents.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_latents, latents=latents_np, prompt=args.prompt, seed=args.seed)
        print(f"[5B] wrote latents {args.save_latents}", flush=True)

    if spatial_mode == "refine":
        del freqs_stage2
    del dit, latents, ehs, noise, freqs
    cleanup_mlx()

    metrics = decode_latents_to_video(
        latents_np,
        args.output_path,
        fps=args.fps,
        backend=args.decode_backend,
        vae_dir=args.vae_root if args.decode_backend == "wan-vae" else None,
        z_dim=in_ch,
    )
    # One h264 round-trip for both post-decode passes (see _postprocess_video).
    rife_s = 0.0
    rife_request = ({
        "factor": args.fast_factor,
        "target_frames": target_frames,
        "sharpen": args.fast_sharpen,
    } if args.fast else None)
    spatial_request = fast_spatial_plan if spatial_mode == "fast_spatial" else None
    if rife_request is not None or spatial_request is not None:
        rife_start = time.perf_counter()
        _postprocess_video(
            video_path=args.output_path, fps=args.fps,
            rife=rife_request, spatial=spatial_request,
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
