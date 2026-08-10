# SPDX-License-Identifier: Apache-2.0
"""Streaming (block-autoregressive) causal Wan generation on Apple Silicon (MLX).

Track C, rung 5 demo + rung 6 benchmark. Loads the released Self-Forcing causal
checkpoint into ``MLXCausalWanDiT`` and generates block-by-block, printing each
block's latency as it finalizes — the streaming behaviour a live preview would
show. Reports the headline latency numbers: time-to-first-block, steady per-block
latency, and peak memory.

The denoise-side metrics are content-independent, so ``--random-prompt`` (default
when no text encoder is present in the model root) uses random text embeddings —
enough to measure latency exactly. Supply a full pipeline root (transformer +
text_encoder + tokenizer [+ vae]) to encode a real prompt and decode frames.

Example (transformer-only root, latency benchmark):

    python examples/inference/basic/mlx_wan_streaming.py \
      --model-root ~/models/sfwan_t2v_1.3b \
      --height 480 --width 832 --num-frames 21
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def _mx_dtype(name: str):
    import mlx.core as mx

    return {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[name]


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX causal Wan streaming demo + latency benchmark.")
    parser.add_argument("--model-root", type=Path, required=True,
                        help="Root with transformer/ (and optionally text_encoder/, tokenizer/, vae/).")
    parser.add_argument("--prompt", default="A fox runs through a mossy forest.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=21, help="Latent frames (== blocks when nfb=1).")
    parser.add_argument("--dmd-denoising-steps", default="1000,750,500,250")
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--num-frames-per-block", type=int, default=1)
    parser.add_argument("--mlx-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--mlx-quantization", choices=("none", "int8", "int4"), default="none")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--metrics-out", type=Path, default=None)
    args = parser.parse_args()

    import mlx.core as mx

    from fastvideo.mlx_runtime.causal_dit import mlx_causal_dit_from_diffusers_safetensors
    from fastvideo.mlx_runtime.causal_sampler import build_dmd_schedule, stream_causal_latents

    dtype = _mx_dtype(args.mlx_dtype)
    quantization = None if args.mlx_quantization == "none" else args.mlx_quantization
    config = json.loads((args.model_root / "transformer" / "config.json").read_text())
    latent_h, latent_w = args.height // 8, args.width // 8
    patch = tuple(config["patch_size"])
    frame_seqlen = (latent_h // patch[1]) * (latent_w // patch[2])

    mx.clear_cache()
    mx.reset_peak_memory()
    load_start = time.perf_counter()
    model = mlx_causal_dit_from_diffusers_safetensors(
        args.model_root / "transformer" / "diffusion_pytorch_model.safetensors",
        args.model_root / "transformer" / "config.json",
        dtype=args.mlx_dtype, quantization=quantization,
        local_attn_size=-1, sink_size=0, num_frames_per_block=args.num_frames_per_block)
    load_s = time.perf_counter() - load_start
    print(f"loaded causal DiT ({config['num_layers']} layers, {args.mlx_quantization}) in {load_s:.1f}s")

    # Prompt embeddings: encode if a text encoder is present, else random (latency is content-independent).
    text_len = int(config.get("text_len", 512))
    if (args.model_root / "text_encoder").exists() and (args.model_root / "tokenizer").exists():
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from examples.inference.basic.mlx_wan_prompt_to_video import encode_prompt
        embeds = encode_prompt(model_root=args.model_root, prompt=args.prompt,
                               max_sequence_length=text_len, device_arg="auto", dtype_arg="fp16")
        text = mx.array(embeds.numpy()).astype(dtype)
        print(f"encoded prompt -> {tuple(text.shape)}")
    else:
        rng = np.random.default_rng(args.seed)
        text = mx.array((rng.standard_normal((1, text_len, int(config["text_dim"]))) * 0.1).astype(np.float16)).astype(
            dtype)
        print(f"no text encoder under {args.model_root}; using random prompt embeds {tuple(text.shape)} (latency only)")

    # Rotary tables for the whole clip (frame-major), and pure-noise latents.
    from examples.inference.basic.mlx_wan_prompt_to_video import make_rotary_embeddings
    cos_full, sin_full = make_rotary_embeddings(
        config, latent_frames=args.num_frames, latent_height=latent_h, latent_width=latent_w)
    gen = np.random.default_rng(args.seed + 1)
    noise = mx.array(
        gen.standard_normal((1, int(config["in_channels"]), args.num_frames, latent_h, latent_w)).astype(
            np.float32)).astype(dtype)

    dmd_steps = [int(s) for s in args.dmd_denoising_steps.split(",") if s.strip()]
    schedule, timesteps = build_dmd_schedule(dmd_steps, flow_shift=args.flow_shift, warp_denoising_step=True)

    print(f"streaming {args.num_frames} frames x {args.height}x{args.width} "
          f"({len(timesteps)}-step DMD, {frame_seqlen} tokens/frame)...")
    mx.reset_peak_memory()
    stream_start = time.perf_counter()
    block_latencies: list[float] = []
    blocks: list[np.ndarray] = []
    prev = stream_start
    time_to_first = None
    for block_index, latent in stream_causal_latents(
            model, text, noise, cos_full, sin_full, schedule, timesteps,
            frame_seqlen=frame_seqlen, seed=args.seed):
        now = time.perf_counter()
        block_latencies.append(now - prev)
        prev = now
        if time_to_first is None:
            time_to_first = now - stream_start
        blocks.append(np.array(latent.astype(mx.float32)))
        print(f"  block {block_index:2d} ready  (+{block_latencies[-1]:.2f}s)")

    total_s = time.perf_counter() - stream_start
    peak_gib = mx.get_peak_memory() / (1024**3)
    steady = statistics.median(block_latencies[1:]) if len(block_latencies) > 1 else block_latencies[0]
    metrics = {
        "num_frames": args.num_frames,
        "resolution": f"{args.height}x{args.width}",
        "dmd_steps": len(timesteps),
        "quantization": args.mlx_quantization,
        "load_s": round(load_s, 2),
        "time_to_first_block_s": round(time_to_first, 3),
        "block_latency_first_s": round(block_latencies[0], 3),
        "block_latency_steady_s": round(steady, 3),
        "total_stream_s": round(total_s, 2),
        "peak_gib": round(peak_gib, 3),
    }
    print("\n=== streaming metrics ===")
    print(json.dumps(metrics, indent=2))
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.metrics_out}")


if __name__ == "__main__":
    main()
