# SPDX-License-Identifier: Apache-2.0
"""Benchmark generate-fewer-frames plus MLX RIFE interpolation.

This script keeps the video diffusion path delegated to
``fastvideo.benchmarks.mlx_fastwan_bench._generate_cell``. It only orchestrates
two frame-count cells and the postprocess interpolation step.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np

from examples.inference.basic.mlx_wan_prompt_to_video import encode_prompt, make_rotary_embeddings
from fastvideo.benchmarks.mlx_fastwan_bench import _generate_cell, _ms_ssim
from fastvideo.mlx_runtime.rife_interp import RIFEBackendError, interpolate, load_model
from fastvideo.mlx_runtime.memory import add_memory_limit_args, apply_memory_limits

FOX_PROMPT = "A fox runs through a misty pine forest, leaves kicking up behind it."
DEFAULT_MODEL_ROOT = Path("/Users/aryank/models/qad_int8_v2")


def _parse_timesteps(raw: str) -> list[int]:
    timesteps = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not timesteps:
        raise SystemExit("No DMD timesteps parsed from --dmd-denoising-steps")
    return timesteps


def _read_video(path: Path) -> list[np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    frames = [np.asarray(frame[:, :, :3], dtype=np.uint8) for frame in imageio.mimread(path)]
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("Cannot write an empty frame list")
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, macro_block_size=1, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def _copy_video(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _make_generation_inputs(args, num_frames: int):
    import mlx.core as mx
    import torch

    config_path = args.model_root / "transformer" / "config.json"
    checkpoint_path = args.model_root / "transformer" / "diffusion_pytorch_model.safetensors"
    if not config_path.is_file():
        raise SystemExit(f"Missing DiT config: {config_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"Missing DiT checkpoint: {checkpoint_path}")

    config = json.loads(config_path.read_text())
    latent_frames = (num_frames - 1) // 4 + 1
    latent_height = args.height // 8
    latent_width = args.width // 8
    freqs_cis = make_rotary_embeddings(
        config,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents_seed = torch.randn(
        (1, int(config["in_channels"]), latent_frames, latent_height, latent_width),
        generator=generator,
        dtype=torch.float32,
    ).numpy()
    timesteps = _parse_timesteps(args.dmd_denoising_steps)
    renoise_by_step = [
        torch.randn(latents_seed.shape, generator=generator, dtype=torch.float32).numpy()
        for _ in range(max(0,
                           len(timesteps) - 1))
    ]
    prompt_embeds = encode_prompt(
        model_root=args.model_root,
        prompt=args.prompt,
        max_sequence_length=args.max_sequence_length,
        device_arg=args.torch_device,
        dtype_arg=args.torch_dtype,
    )
    return {
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "encoder_hidden_states": mx.array(prompt_embeds.numpy()),
        "freqs_cis": freqs_cis,
        "timesteps": timesteps,
        "latents_seed": latents_seed,
        "renoise_by_step": renoise_by_step,
        "latent_frames": latent_frames,
    }


def _generate(args, num_frames: int):
    cell_args = SimpleNamespace(
        model_root=args.model_root,
        height=args.height,
        width=args.width,
        num_frames=num_frames,
        fps=args.fps,
        flow_shift=args.flow_shift,
        torch_device=args.torch_device,
        torch_dtype=args.torch_dtype,
        taehv_source_path=args.taehv_source_path,
        taehv_checkpoint_path=args.taehv_checkpoint_path,
        taehv_parallel=args.taehv_parallel,
        mlx_checkpoint_cache=args.mlx_checkpoint_cache,
        mlx_memory_limit_gib=args.mlx_memory_limit_gib,
        mlx_cache_limit_gib=args.mlx_cache_limit_gib,
        mlx_disable_cache=args.mlx_disable_cache,
        mlx_wired_limit_gib=args.mlx_wired_limit_gib,
        torch_mps_high_watermark_ratio=args.torch_mps_high_watermark_ratio,
        torch_mps_low_watermark_ratio=args.torch_mps_low_watermark_ratio,
        benchmark_preset="metalfx-rife",
        current_prompt_id=f"fox-{num_frames}f",
        current_prompt=args.prompt,
        output_dir=args.output_dir,
    )
    inputs = _make_generation_inputs(args, num_frames)
    print(f"=== generate: {num_frames} frames ({inputs['latent_frames']} latent frames) ===", flush=True)
    return _generate_cell(
        args=cell_args,
        mode=args.mode,
        decoder=args.decoder,
        checkpoint_path=inputs["checkpoint_path"],
        config_path=inputs["config_path"],
        encoder_hidden_states=inputs["encoder_hidden_states"],
        freqs_cis=inputs["freqs_cis"],
        timesteps=inputs["timesteps"],
        latents_seed=inputs["latents_seed"],
        renoise_by_step=inputs["renoise_by_step"],
    )


def _relative_to_output(path: Path, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def _write_report(args, result: dict) -> Path:
    report_path = args.output_dir / "metalfx_rife_report.md"
    rows = result["speed_rows"]
    table_lines = [
        "| path | denoise_s | decode_s | gen_total_s | rife_s | net_s | speedup_vs_81 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table_lines.append(
            "| {path} | {denoise_s:.3f} | {decode_s:.3f} | {gen_total_s:.3f} | {rife_s:.3f} | {net_s:.3f} | {speedup_vs_81:.3f}x |"
            .format(**row))
    text = f"""# MetalFX/RIFE Generate-Fewer-Frames Benchmark Run

Prompt: {args.prompt}

Resolution: {args.height}x{args.width}, fps={args.fps}, mode={args.mode}, decoder={args.decoder}

| metric | value |
| --- | ---: |
| reconstruction_ms_ssim | {result['reconstruction_ms_ssim']:.6f} |
| reference_frames | {result['reference_frames']} |
| reduced_frames | {result['reduced_frames']} |
| interpolated_frames | {result['interpolated_frames']} |

{chr(10).join(table_lines)}

Videos:

- reference: `{result['videos']['reference']}`
- reference drop-41 RIFE reconstruction: `{result['videos']['drop41_rife81']}`
- generated 41 RIFE to 81: `{result['videos']['generated41_rife81']}`
- generated 41 direct: `{result['videos']['generated41']}`

Raw metrics are in `{_relative_to_output(args.output_dir / 'metrics.json', args.output_dir)}`.
"""
    report_path.write_text(text)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate 41-frame generation plus MLX RIFE interpolation vs 81-frame generation.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--prompt", default=FOX_PROMPT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--reference-frames", type=int, default=81)
    parser.add_argument("--reduced-frames", type=int, default=41)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--mode", default="int8", choices=("fp16", "bf16", "int8", "int4", "mxfp8", "mxfp4", "nvfp4"))
    parser.add_argument("--decoder", default="taehv", choices=("taehv", "wan-vae"))
    parser.add_argument("--dmd-denoising-steps", default="1000,757,522")
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--torch-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/metalfx_rife"))
    parser.add_argument("--mlx-checkpoint-cache", type=Path, default=None)
    parser.add_argument("--compile", action="store_true", help="Enable FASTVIDEO_MLX_COMPILE=1 for DiT denoise.")
    parser.add_argument("--rife-scale", type=float, default=1.0)
    parser.add_argument("--taehv-source-path", type=Path, default=None)
    parser.add_argument("--taehv-checkpoint-path", type=Path, default=None)
    parser.add_argument("--taehv-parallel", action="store_true")
    add_memory_limit_args(parser)
    args = parser.parse_args()

    if args.reference_frames != (args.reduced_frames - 1) * 2 + 1:
        raise SystemExit(
            "--reference-frames must equal (--reduced-frames - 1) * 2 + 1 for the default every-other-frame test")
    if args.compile:
        import os

        os.environ["FASTVIDEO_MLX_COMPILE"] = "1"

    args.model_root = args.model_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mlx_checkpoint_cache is None:
        args.mlx_checkpoint_cache = args.output_dir / "mlx_checkpoint_cache"
    else:
        args.mlx_checkpoint_cache = args.mlx_checkpoint_cache.expanduser().resolve()

    import mlx.core as mx
    import torch

    runtime_limits = apply_memory_limits(
        mlx_memory_limit_gib=args.mlx_memory_limit_gib,
        mlx_cache_limit_gib=args.mlx_cache_limit_gib,
        mlx_disable_cache=args.mlx_disable_cache,
        mlx_wired_limit_gib=args.mlx_wired_limit_gib,
        torch_mps_high_watermark_ratio=args.torch_mps_high_watermark_ratio,
        torch_mps_low_watermark_ratio=args.torch_mps_low_watermark_ratio,
        mx_module=mx,
    ).as_metrics()
    mx.random.seed(args.seed)
    torch.manual_seed(args.seed)

    reference_cell = _generate(args, args.reference_frames)
    reduced_cell = _generate(args, args.reduced_frames)

    reference_video = _copy_video(reference_cell.video_path, args.output_dir / "fox_reference_81.mp4")
    generated41_video = _copy_video(reduced_cell.video_path, args.output_dir / "fox_generated_41.mp4")

    reference_frames = _read_video(reference_video)
    dropped_reference_frames = reference_frames[::2]
    if len(dropped_reference_frames) != args.reduced_frames:
        raise RuntimeError(f"Expected {args.reduced_frames} dropped frames, got {len(dropped_reference_frames)}")

    try:
        rife_model = load_model("4.25")
    except RIFEBackendError:
        raise
    except Exception as exc:  # noqa: BLE001 - preserve exact backend failure.
        raise RIFEBackendError(f"Unexpected RIFE load failure: {exc}") from exc

    print("=== RIFE reconstruction: drop every other reference frame, 41 -> 81 ===", flush=True)
    start = time.perf_counter()
    reconstructed_frames = interpolate(dropped_reference_frames, factor=2, model=rife_model, scale=args.rife_scale)
    recon_rife_s = time.perf_counter() - start
    reconstructed_video = args.output_dir / "fox_reference_drop41_rife81.mp4"
    _write_video(reconstructed_video, reconstructed_frames, args.fps)
    reconstruction_ms_ssim = _ms_ssim(reference_video, reconstructed_video, required=True)
    if reconstruction_ms_ssim is None:
        raise RuntimeError("MS-SSIM returned None for reconstruction comparison")

    print("=== RIFE actual reduced path: generated 41 -> 81 ===", flush=True)
    generated41_frames = _read_video(generated41_video)
    start = time.perf_counter()
    generated41_rife_frames = interpolate(generated41_frames, factor=2, model=rife_model, scale=args.rife_scale)
    generated41_rife_s = time.perf_counter() - start
    generated41_rife_video = args.output_dir / "fox_generated41_rife81.mp4"
    _write_video(generated41_rife_video, generated41_rife_frames, args.fps)

    direct_denoise_s = float(reference_cell.metrics["denoise_s"])
    reduced_denoise_s = float(reduced_cell.metrics["denoise_s"])
    direct_gen_total_s = float(reference_cell.metrics["denoise_s"]) + float(reference_cell.metrics["decode_s"])
    reduced_gen_total_s = float(reduced_cell.metrics["denoise_s"]) + float(reduced_cell.metrics["decode_s"])
    net_denoise_rife_s = reduced_denoise_s + generated41_rife_s
    net_gen_rife_s = reduced_gen_total_s + generated41_rife_s

    result = {
        "prompt":
        args.prompt,
        "model_root":
        str(args.model_root),
        "rife_impl":
        "rife-mlx git+https://github.com/xocialize/rife-mlx.git, weights mlx-community/RIFE-4.25",
        "runtime_limits":
        runtime_limits,
        "reference_frames":
        len(reference_frames),
        "reduced_frames":
        len(generated41_frames),
        "interpolated_frames":
        len(generated41_rife_frames),
        "reconstruction_ms_ssim":
        reconstruction_ms_ssim,
        "reconstruction_rife_s":
        recon_rife_s,
        "generated41_rife_s":
        generated41_rife_s,
        "reference_metrics":
        reference_cell.metrics,
        "reduced_metrics":
        reduced_cell.metrics,
        "speed_rows": [
            {
                "path": "generate_81",
                "denoise_s": direct_denoise_s,
                "decode_s": float(reference_cell.metrics["decode_s"]),
                "gen_total_s": direct_gen_total_s,
                "rife_s": 0.0,
                "net_s": direct_gen_total_s,
                "speedup_vs_81": 1.0,
            },
            {
                "path": "generate_41_plus_rife81_denoise_only",
                "denoise_s": reduced_denoise_s,
                "decode_s": 0.0,
                "gen_total_s": reduced_denoise_s,
                "rife_s": generated41_rife_s,
                "net_s": net_denoise_rife_s,
                "speedup_vs_81": direct_denoise_s / net_denoise_rife_s,
            },
            {
                "path": "generate_41_plus_rife81_decode_included",
                "denoise_s": reduced_denoise_s,
                "decode_s": float(reduced_cell.metrics["decode_s"]),
                "gen_total_s": reduced_gen_total_s,
                "rife_s": generated41_rife_s,
                "net_s": net_gen_rife_s,
                "speedup_vs_81": direct_gen_total_s / net_gen_rife_s,
            },
        ],
        "videos": {
            "reference": _relative_to_output(reference_video, args.output_dir),
            "drop41_rife81": _relative_to_output(reconstructed_video, args.output_dir),
            "generated41": _relative_to_output(generated41_video, args.output_dir),
            "generated41_rife81": _relative_to_output(generated41_rife_video, args.output_dir),
        },
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2))
    report_path = _write_report(args, result)

    print(json.dumps(result["speed_rows"], indent=2))
    print(f"reconstruction_ms_ssim={reconstruction_ms_ssim:.6f}")
    print(f"wrote {metrics_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
