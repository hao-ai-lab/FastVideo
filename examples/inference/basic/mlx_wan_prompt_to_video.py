"""Generate a FastWan text-to-video clip with the Apple Silicon MLX runtime.

This is the supported source-tree entrypoint for the FastWan-QAD-INT8-1.3B
Apple release:

- Hugging Face/torch encodes the prompt with UMT5 (bf16 by default: fp32
  exponent range without fp16 overflow risk, at fp16 memory cost).
- MLX runs the FastWan DiT denoising loop (INT8 by default, compiled with
  ``mx.compile`` unless ``--no-mlx-compile``).
- TAEHV (default, fast/low-memory) or the full Wan VAE (``--decode-backend
  wan-vae``, higher fidelity, bf16) decodes the final latents.

Defaults produce the validated release shape: 480x832, 81 frames, 3-step DMD.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from fastvideo.mlx_runtime.memory import add_memory_limit_args, apply_memory_limits


DEFAULT_MODEL_ID = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"

# Legacy pinned-snapshot location, kept for callers that import it (the MLX
# benchmark harness). New code should prefer resolve_model_root(None), which
# resolves whatever snapshot the local HF cache has (downloading if needed).
DEFAULT_MODEL_ROOT = (
    Path.home()
    / ".cache/huggingface/hub/models--FastVideo--FastWan2.1-T2V-1.3B-Diffusers/"
    "snapshots/25e7ed7f41fd8ce2fdd108688c65e8caf0ce3aef"
)


def resolve_model_root(model_root: Path | None, *, model_id: str = DEFAULT_MODEL_ID) -> Path:
    """Return a usable model directory, resolving via the HF cache if unset.

    A user-supplied ``model_root`` is returned as-is. Otherwise the model is
    resolved through ``huggingface_hub.snapshot_download``, which reuses the
    local cache when present and downloads the current snapshot when not —
    no hardcoded snapshot hash.
    """
    if model_root is not None:
        return model_root
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id))


def _torch_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(device_arg)


def _torch_dtype(dtype_arg: str):
    import torch

    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_arg]


def _cleanup_torch() -> None:
    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def encode_prompt(
    *,
    model_root: Path,
    prompt: str,
    max_sequence_length: int,
    device_arg: str,
    dtype_arg: str,
):
    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel

    device = _torch_device(device_arg)
    dtype = _torch_dtype(dtype_arg)
    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_root / "text_encoder",
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    text_encoder.eval()

    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()

    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids, mask).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens, strict=False)]
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )
    if prompt_embeds.dtype == torch.bfloat16:
        # NumPy (and the .npy cache/subprocess transport) has no bfloat16;
        # fp32 is exact for every bf16 value.
        prompt_embeds = prompt_embeds.float()
    prompt_embeds = prompt_embeds.cpu().contiguous()
    del text_encoder, tokenizer, text_inputs, text_input_ids, mask
    _cleanup_torch()
    return prompt_embeds


def encode_prompt_subprocess(
    *,
    model_root: Path,
    prompt: str,
    max_sequence_length: int,
    device_arg: str,
    dtype_arg: str,
):
    import torch

    with tempfile.TemporaryDirectory(prefix="fastvideo_prompt_embeds_") as tmpdir:
        output_path = Path(tmpdir) / "prompt_embeds.npy"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--model-root",
                str(model_root),
                "--prompt",
                prompt,
                "--max-sequence-length",
                str(max_sequence_length),
                "--torch-device",
                device_arg,
                "--text-encoder-dtype",
                dtype_arg,
                "--encode-prompt-only",
                str(output_path),
            ],
            check=True,
        )
        prompt_embeds = np.load(output_path)
    return torch.from_numpy(prompt_embeds).contiguous()


def get_prompt_embeds(
    *,
    model_root: Path,
    prompt: str,
    max_sequence_length: int,
    device_arg: str,
    dtype_arg: str,
    encode_mode: str,
    cache_path: Path | None,
):
    import torch

    if cache_path is not None and cache_path.exists():
        return torch.from_numpy(np.load(cache_path)).contiguous()

    if encode_mode == "subprocess":
        prompt_embeds = encode_prompt_subprocess(
            model_root=model_root,
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device_arg=device_arg,
            dtype_arg=dtype_arg,
        )
    elif encode_mode == "inline":
        prompt_embeds = encode_prompt(
            model_root=model_root,
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device_arg=device_arg,
            dtype_arg=dtype_arg,
        )
    else:
        raise ValueError(f"Unsupported prompt encode mode: {encode_mode}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, prompt_embeds.cpu().numpy())
    return prompt_embeds


def make_rotary_embeddings(config: dict, *, latent_frames: int, latent_height: int, latent_width: int):
    import mlx.core as mx
    import torch

    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed

    num_heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])
    hidden_size = num_heads * head_dim
    patch_size = tuple(config["patch_size"])
    post_patch = (
        latent_frames // patch_size[0],
        latent_height // patch_size[1],
        latent_width // patch_size[2],
    )
    rope_dim_list = [head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)]
    freqs_cos, freqs_sin = get_rotary_pos_embed(
        post_patch,
        hidden_size,
        num_heads,
        rope_dim_list,
        dtype=torch.float32,
        rope_theta=10000,
    )
    return (
        mx.array(freqs_cos.numpy()).astype(mx.float32),
        mx.array(freqs_sin.numpy()).astype(mx.float32),
    )


def decode_latents_to_video(
    *,
    model_root: Path,
    latents_np: np.ndarray,
    output_path: Path,
    fps: int,
    device_arg: str,
    dtype_arg: str,
    backend: str,
    taehv_source_path: Path | None,
    taehv_checkpoint_path: Path | None,
    taehv_parallel: bool,
) -> None:
    import torch
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor
    from diffusers.utils import export_to_video

    device = _torch_device(device_arg)
    dtype = _torch_dtype(dtype_arg)
    if backend == "taehv":
        from fastvideo.mlx_runtime.taehv_decode import decode_latents_to_video_taehv

        decode_latents_to_video_taehv(
            latents_np=latents_np,
            output_path=output_path,
            fps=fps,
            device=device,
            dtype=dtype,
            parallel=taehv_parallel,
            source_path=taehv_source_path,
            checkpoint_path=taehv_checkpoint_path,
        )
        _cleanup_torch()
        return

    if backend != "wan-vae":
        raise ValueError(f"Unsupported decode backend: {backend}")

    vae = AutoencoderKLWan.from_pretrained(
        model_root / "vae",
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    vae.eval()

    latents = torch.from_numpy(latents_np).to(device=device, dtype=dtype)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents = latents / latents_std + latents_mean

    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]
    video = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial).postprocess_video(video, output_type="np")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video[0], str(output_path), fps=fps)
    del vae, latents, video
    _cleanup_torch()


def _unsharp(frame: np.ndarray, amount: float) -> np.ndarray:
    """Light unsharp mask to counter RIFE's optical-flow softening."""
    import cv2

    blur = cv2.GaussianBlur(frame, (0, 0), 1.0)
    return cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)


def _rife_interpolate_video(*, video_path: Path, target_frames: int, factor: int,
                            sharpen: float, fps: int) -> None:
    """Read the reduced-frame mp4, RIFE-interpolate up to ``target_frames`` on
    Apple Silicon, optionally light-sharpen, and rewrite the file in place."""
    import imageio.v3 as iio

    from fastvideo.mlx_runtime.rife_interp import interpolate as rife_interpolate, load_model

    frames = [frame for frame in iio.imread(video_path)]
    model = load_model()
    interp = rife_interpolate(frames, factor=factor, model=model)
    if len(interp) > target_frames:
        interp = interp[:target_frames]
    if sharpen and sharpen > 0:
        interp = [_unsharp(frame, sharpen) for frame in interp]
    iio.imwrite(video_path, np.stack(interp), fps=fps, codec="libx264")
    print(f"[fast] RIFE {factor}x -> {len(interp)} frames written to {video_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-to-video FastWan generation using MLX for the DiT")
    parser.add_argument("--model-root", type=Path, default=None,
                        help=f"Model directory. Defaults to the local HF cache for {DEFAULT_MODEL_ID} "
                        "(downloading it if missing).")
    parser.add_argument("--prompt", default="A paper boat sails through a shallow stream in a mossy forest.")
    parser.add_argument("--output-path", type=Path, default=Path("video_samples/mlx_fastwan_prompt_to_video.mp4"))
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=3)
    parser.add_argument("--dmd-denoising-steps", default="1000,757,522")
    parser.add_argument("--denoising-mode", choices=("dmd", "scheduler"), default="dmd")
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=False,
                        help="Fast mode: generate 1/factor of the frames, then RIFE-interpolate up "
                        "to --num-frames on Apple Silicon (~2.7x faster denoise, reconstruction "
                        "MS-SSIM ~0.97). Requires the rife-mlx package. See "
                        "docs/experiments/rife-speedup-summary.md.")
    parser.add_argument("--fast-factor", type=int, default=2,
                        help="Fast-mode interpolation factor (2 = generate half the frames).")
    parser.add_argument("--fast-sharpen", type=float, default=0.6,
                        help="Light unsharp strength to counter RIFE softness (0 disables).")
    parser.add_argument("--torch-device", default="auto", help="'auto', 'mps', or 'cpu' for text/VAE components.")
    parser.add_argument("--torch-dtype", choices=("fp16", "bf16", "fp32"), default="fp16",
                        help="Dtype for the TAEHV decode path (and legacy callers).")
    parser.add_argument("--text-encoder-dtype", choices=("bf16", "fp16", "fp32"), default="bf16",
                        help="UMT5 prompt-encode dtype. bf16 keeps the fp32 exponent range "
                        "(no fp16 overflow risk in the T5 stack) at fp16 memory cost; the "
                        "reference CUDA pipeline encodes in fp32. Pass fp16 if your "
                        "macOS/torch build lacks bf16 on MPS.")
    parser.add_argument("--vae-decode-dtype", choices=("bf16", "fp16", "fp32"), default="bf16",
                        help="Wan VAE decode dtype (wan-vae backend only). bf16 matches the "
                        "reference pipeline's effectively-lossless decode default.")
    parser.add_argument("--mlx-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument(
        "--mlx-quantization",
        choices=("none", "int8", "int4", "mxfp8", "mxfp4", "nvfp4"),
        default="int8",
    )
    parser.add_argument("--mlx-compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Compile the DiT forward with mx.compile (bit-identical to eager, "
                        "~1.4x faster denoise; falls back to eager if tracing fails). "
                        "Disable with --no-mlx-compile.")
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--decode-backend", choices=("wan-vae", "taehv"), default="taehv",
                        help="taehv (default): fast, low-memory tiny decoder. "
                        "wan-vae: full Wan VAE in bf16 — slower and heavier but higher fidelity.")
    parser.add_argument("--taehv-source-path", type=Path, default=None)
    parser.add_argument("--taehv-checkpoint-path", type=Path, default=None)
    parser.add_argument("--taehv-parallel", action="store_true", help="Decode all TAEHV frames at once; faster but higher memory.")
    parser.add_argument("--prompt-encode-mode", choices=("inline", "subprocess"), default="inline")
    parser.add_argument("--prompt-embeds-cache", type=Path, default=None)
    parser.add_argument("--mlx-checkpoint", type=Path, default=None,
                        help="Load the DiT from a pre-quantized MLX checkpoint directory "
                        "(created with --save-mlx-checkpoint) instead of casting/quantizing "
                        "the Diffusers weights on every run.")
    parser.add_argument("--save-mlx-checkpoint", type=Path, default=None,
                        help="After loading the DiT, save it (cast + quantized) as an MLX "
                        "checkpoint directory for fast reloads via --mlx-checkpoint.")
    add_memory_limit_args(parser)
    parser.add_argument("--encode-prompt-only", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Fast mode: generate fewer frames now, RIFE-interpolate back up after decode.
    fast_target_frames = None
    if args.fast:
        if args.fast_factor < 2:
            parser.error("--fast-factor must be >= 2")
        fast_target_frames = args.num_frames
        args.num_frames = (fast_target_frames + args.fast_factor - 1) // args.fast_factor
        print(f"[fast] generating {args.num_frames} frames, RIFE {args.fast_factor}x -> {fast_target_frames}")

    runtime_limits = apply_memory_limits(
        mlx_memory_limit_gib=args.mlx_memory_limit_gib,
        mlx_cache_limit_gib=args.mlx_cache_limit_gib,
        mlx_disable_cache=args.mlx_disable_cache,
        mlx_wired_limit_gib=args.mlx_wired_limit_gib,
        torch_mps_high_watermark_ratio=args.torch_mps_high_watermark_ratio,
        torch_mps_low_watermark_ratio=args.torch_mps_low_watermark_ratio,
    ).as_metrics()

    model_root = resolve_model_root(args.model_root)

    if args.encode_prompt_only is not None:
        prompt_embeds = encode_prompt(
            model_root=model_root,
            prompt=args.prompt,
            max_sequence_length=args.max_sequence_length,
            device_arg=args.torch_device,
            dtype_arg=args.text_encoder_dtype,
        )
        args.encode_prompt_only.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.encode_prompt_only, prompt_embeds.cpu().numpy())
        return

    import mlx.core as mx
    import torch
    from diffusers import UniPCMultistepScheduler

    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from fastvideo.mlx_runtime.fastwan import mlx_dit_from_diffusers_safetensors
    from fastvideo.mlx_runtime.sampling import MLXDMDSchedule, dmd_step

    mx.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_path = model_root / "transformer/config.json"
    checkpoint_path = model_root / "transformer/diffusion_pytorch_model.safetensors"
    config = json.loads(config_path.read_text())
    latent_frames = (args.num_frames - 1) // 4 + 1
    latent_height = args.height // 8
    latent_width = args.width // 8
    mx_dtype = {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[args.mlx_dtype]
    quantization = None if args.mlx_quantization == "none" else args.mlx_quantization

    total_start = time.perf_counter()
    prompt_start = time.perf_counter()
    prompt_embeds = get_prompt_embeds(
        model_root=model_root,
        prompt=args.prompt,
        max_sequence_length=args.max_sequence_length,
        device_arg=args.torch_device,
        dtype_arg=args.text_encoder_dtype,
        encode_mode=args.prompt_encode_mode,
        cache_path=args.prompt_embeds_cache,
    )
    prompt_time = time.perf_counter() - prompt_start

    load_start = time.perf_counter()
    mx.clear_cache()
    mx.reset_peak_memory()
    if args.mlx_checkpoint is not None:
        from fastvideo.mlx_runtime.checkpoint import load_mlx_dit_checkpoint

        dit = load_mlx_dit_checkpoint(args.mlx_checkpoint, compile=args.mlx_compile)
        config = dit.config
    else:
        dit = mlx_dit_from_diffusers_safetensors(
            checkpoint_path,
            config_path,
            dtype=args.mlx_dtype,
            quantization=quantization,
            compile=args.mlx_compile,
        )
    load_time = time.perf_counter() - load_start
    load_peak_memory = mx.get_peak_memory()

    if args.save_mlx_checkpoint is not None:
        from fastvideo.mlx_runtime.checkpoint import save_mlx_dit_checkpoint

        save_mlx_dit_checkpoint(dit, args.save_mlx_checkpoint)

    if args.denoising_mode == "dmd":
        scheduler = FlowMatchEulerDiscreteScheduler(shift=args.flow_shift)
        denoising_steps = [int(step.strip()) for step in args.dmd_denoising_steps.split(",") if step.strip()]
        timesteps = torch.tensor(denoising_steps, dtype=torch.long)
    else:
        scheduler = UniPCMultistepScheduler.from_pretrained(model_root / "scheduler", local_files_only=True)
        scheduler.set_timesteps(args.num_inference_steps, device="cpu")
        scheduler.set_begin_index(0)
        timesteps = scheduler.timesteps

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents_torch = torch.randn(
        (1, int(config["in_channels"]), latent_frames, latent_height, latent_width),
        generator=generator,
        dtype=torch.float32,
    )
    latents = mx.array(latents_torch.numpy()).astype(mx_dtype)
    encoder_hidden_states = mx.array(prompt_embeds.numpy()).astype(mx_dtype)
    freqs_cis = make_rotary_embeddings(
        config,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
    )

    # DMD keeps the whole update on the MLX device via the native sampler. Only
    # the (non-distilled) diffusers scheduler path still round-trips to torch.
    dmd_schedule = MLXDMDSchedule.from_torch_scheduler(scheduler) if args.denoising_mode == "dmd" else None

    denoise_start = time.perf_counter()
    mx.reset_peak_memory()
    for step_index, timestep in enumerate(timesteps):
        noise_input_latent = latents
        timestep_mx = mx.array([float(timestep.item())]).astype(mx.float32)
        noise_pred = dit(latents.astype(mx_dtype), encoder_hidden_states, timestep_mx, freqs_cis)

        if args.denoising_mode == "dmd":
            # On-device DMD update: no per-step MLX->torch->MLX round-trip. The
            # affine math runs in fp32 to match the torch reference precision,
            # then casts back to the runtime dtype. Re-noise is drawn with MLX's
            # RNG (seeded above) instead of the torch CPU generator.
            ts_val = float(timestep.item())
            noise_input_f32 = noise_input_latent.astype(mx.float32)
            pred_noise_f32 = noise_pred.astype(mx.float32)
            if step_index < len(timesteps) - 1:
                next_ts: float | None = float(timesteps[step_index + 1].item())
                renoise = mx.random.normal(noise_input_f32.shape).astype(mx.float32)
            else:
                next_ts, renoise = None, None
            latents = dmd_step(
                latents=noise_input_f32,
                noise_input_latent=noise_input_f32,
                pred_noise=pred_noise_f32,
                schedule=dmd_schedule,
                timestep=ts_val,
                next_timestep=next_ts,
                noise=renoise,
            ).astype(mx_dtype)
        else:
            mx.eval(noise_pred)
            noise_pred_torch = torch.from_numpy(np.array(noise_pred.astype(mx.float32)))
            latents_torch = torch.from_numpy(np.array(latents.astype(mx.float32)))
            latents_torch = scheduler.step(noise_pred_torch, timestep, latents_torch, return_dict=False)[0]
            latents = mx.array(latents_torch.numpy()).astype(mx_dtype)

        mx.eval(latents)
        print(f"denoise step {step_index + 1}/{len(timesteps)} complete")
    denoise_time = time.perf_counter() - denoise_start
    denoise_peak_memory = mx.get_peak_memory()
    active_memory = mx.get_active_memory()

    latents_np = np.array(latents.astype(mx.float32))
    if args.save_latents:
        latent_path = args.output_path.with_suffix(".latents.npy")
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(latent_path, latents_np)
        print(f"Saved latents to: {latent_path}")

    decode_start = time.perf_counter()
    decode_latents_to_video(
        model_root=model_root,
        latents_np=latents_np,
        output_path=args.output_path,
        fps=args.fps,
        device_arg=args.torch_device,
        dtype_arg=(args.vae_decode_dtype if args.decode_backend == "wan-vae" else args.torch_dtype),
        backend=args.decode_backend,
        taehv_source_path=args.taehv_source_path,
        taehv_checkpoint_path=args.taehv_checkpoint_path,
        taehv_parallel=args.taehv_parallel,
    )
    decode_time = time.perf_counter() - decode_start

    rife_time = 0.0
    if fast_target_frames is not None:
        rife_start = time.perf_counter()
        _rife_interpolate_video(
            video_path=args.output_path,
            target_frames=fast_target_frames,
            factor=args.fast_factor,
            sharpen=args.fast_sharpen,
            fps=args.fps,
        )
        rife_time = time.perf_counter() - rife_start
        print(f"RIFE fast-mode interpolate time: {rife_time:.2f}s")

    total_time = time.perf_counter() - total_start

    print(f"Prompt encode time: {prompt_time:.2f}s")
    print(f"MLX DiT load time: {load_time:.2f}s")
    print(f"MLX denoise time: {denoise_time:.2f}s")
    print(f"Decode/export time: {decode_time:.2f}s")
    print(f"Total prompt-to-video time: {total_time:.2f}s")
    print(f"MLX load peak memory: {load_peak_memory / (1024 ** 3):.2f} GiB")
    print(f"MLX denoise peak memory: {denoise_peak_memory / (1024 ** 3):.2f} GiB")
    print(f"MLX active memory after denoise: {active_memory / (1024 ** 3):.2f} GiB")
    print(f"Output written to: {args.output_path}")

    if args.metrics_json is not None:
        metrics = {
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "denoising_mode": args.denoising_mode,
            "dmd_denoising_steps": [int(step.strip()) for step in args.dmd_denoising_steps.split(",") if step.strip()],
            "mlx_dtype": args.mlx_dtype,
            "mlx_quantization": args.mlx_quantization,
            "mlx_compile": args.mlx_compile,
            "text_encoder_dtype": args.text_encoder_dtype,
            "vae_decode_dtype": args.vae_decode_dtype if args.decode_backend == "wan-vae" else None,
            "model_root": str(model_root),
            "decode_backend": args.decode_backend,
            "taehv_parallel": args.taehv_parallel if args.decode_backend == "taehv" else None,
            "prompt_encode_mode": args.prompt_encode_mode,
            "prompt_embeds_cache": str(args.prompt_embeds_cache) if args.prompt_embeds_cache else None,
            "prompt_encode_s": prompt_time,
            "mlx_dit_load_s": load_time,
            "mlx_denoise_s": denoise_time,
            "vae_decode_export_s": decode_time,
            "decode_export_s": decode_time,
            "total_s": total_time,
            "mlx_load_peak_bytes": int(load_peak_memory),
            "mlx_denoise_peak_bytes": int(denoise_peak_memory),
            "mlx_active_after_denoise_bytes": int(active_memory),
            "output_path": str(args.output_path),
            **runtime_limits,
        }
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2))
        print(f"Metrics written to: {args.metrics_json}")


if __name__ == "__main__":
    main()
