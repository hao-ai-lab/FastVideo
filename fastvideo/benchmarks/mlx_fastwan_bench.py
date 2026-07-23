# SPDX-License-Identifier: Apache-2.0
"""Prove-out benchmark for the MLX FastWan runtime (Apple Silicon).

Sweeps ``{dtype/quant} x {decoder}``, generates a clip per cell, and records the
latency breakdown, peak unified memory, and MS-SSIM (optionally LPIPS) against a
reference video. It emits a JSON blob and a markdown table -- the artifact that
turns "int8 + TAEHV looks good" into defensible numbers, and (via
``--assert-min-ssim``) a regression gate for the ``mx.compile`` work.

Design notes:
- Generation reuses the hybrid POC helpers in
  ``examples/inference/basic/mlx_wan_prompt_to_video.py`` (torch-MPS UMT5 encode
  and Wan-VAE/TAEHV decode) plus the on-device MLX DMD sampler
  (``fastvideo/mlx_runtime/sampling.py``); the denoise loop never leaves the
  device.
- Quality reuses the tested MS-SSIM primitive
  ``fastvideo/tests/utils.py::compute_video_ssim_torchvision``.
- Reference: by default each cell is scored against the highest-fidelity cell
  in the sweep (``fp16`` + ``wan-vae``), which needs no CUDA box and answers
  "how much does int8/int4/TAEHV degrade vs the best local config". Pass
  ``--reference PATH`` to score against an external clip instead (e.g. the
  torch-MPS or CUDA FastVideo output of the same model) for a "vs. the original
  model" column.

Run on an Apple Silicon Mac (needs ``mlx`` + a torch build with MPS):

    python fastvideo/benchmarks/mlx_fastwan_bench.py \
        --modes fp16,bf16,int8,int4 --decoders taehv,wan-vae
"""

from __future__ import annotations

import argparse
import html
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from examples.inference.basic.mlx_wan_prompt_to_video import (
    DEFAULT_MODEL_ROOT,
    decode_latents_to_video,
    encode_prompt,
    make_rotary_embeddings,
)
from fastvideo.mlx_runtime.memory import add_memory_limit_args, apply_memory_limits

# The highest-fidelity cell; used as the default SSIM reference when no external
# reference video is supplied.
REFERENCE_MODE = "fp16"
REFERENCE_DECODER = "wan-vae"

ALLOWED_MODES = ("fp16", "bf16", "int8", "int4", "mxfp8", "mxfp4", "nvfp4")
ALLOWED_DECODERS = ("taehv", "wan-vae")


@dataclass(frozen=True)
class PromptCase:
    id: str
    prompt: str


@dataclass(frozen=True)
class BenchmarkPreset:
    height: int
    width: int
    num_frames: int
    modes: str
    decoders: str
    mlx_memory_limit_gib: float | None = None
    mlx_disable_cache: bool = False
    torch_mps_high_watermark_ratio: float | None = None
    torch_mps_low_watermark_ratio: float | None = None


PROMPT_SETS = {
    "motion7": (
        PromptCase("beach-sunset", "A slow cinematic sunset over ocean waves at a quiet beach."),
        PromptCase("fox-forest", "A fox runs through a misty pine forest, leaves kicking up behind it."),
        PromptCase("raccoon-sunflowers", "A raccoon walks through a sunflower field as petals move in the wind."),
        PromptCase("surfing-cat", "A cat wearing sunglasses surfs across a bright blue ocean wave."),
        PromptCase("burning-clock", "A vintage table clock burns on a wooden desk, flames flickering realistically."),
        PromptCase("forest-walk", "Video game style footage of a man walking through a dense forest path."),
        PromptCase("sea-dock-yachts", "Several yachts are parked at a sea dock while water ripples around them."),
    ),
}

BENCHMARK_PRESETS = {
    "default":
    BenchmarkPreset(
        height=480,
        width=832,
        num_frames=81,
        modes="fp16,bf16,int8,int4",
        decoders="taehv,wan-vae",
    ),
    "mac-16gb":
    BenchmarkPreset(
        height=448,
        width=832,
        num_frames=61,
        modes="int8",
        decoders="taehv",
        mlx_memory_limit_gib=16.0,
        mlx_disable_cache=True,
        torch_mps_high_watermark_ratio=0.57,
        torch_mps_low_watermark_ratio=0.0,
    ),
    "mac-32gb":
    BenchmarkPreset(
        height=480,
        width=832,
        num_frames=81,
        modes="int8,fp16",
        decoders="taehv",
    ),
    "mac-64gb":
    BenchmarkPreset(
        height=480,
        width=832,
        num_frames=81,
        modes="int8,fp16",
        decoders="taehv,wan-vae",
    ),
}


@dataclass
class Cell:
    prompt_id: str
    prompt: str
    mode: str
    decoder: str
    video_path: Path
    latents: np.ndarray
    metrics: dict[str, float | int | str | bool | None] = field(default_factory=dict)


def _mode_to_dtype_quant(mode: str) -> tuple[str, str | None]:
    """Map a sweep mode to (MLX compute dtype, quantization spec).

    Quantized modes keep fp16 activations and quantize only the DiT linear
    weights (matching ``mlx_dit_from_diffusers_safetensors``).
    """
    if mode == "bf16":
        return "bf16", None
    if mode == "fp16":
        return "fp16", None
    # int8/int4/mxfp*/nvfp4 -> fp16 activations + quantized weights.
    return "fp16", mode


def _mx_dtype(mx, base: str):
    return {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[base]


def _parse_list(raw: str, allowed: tuple[str, ...], label: str) -> list[str]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = sorted(set(items) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported {label}: {unknown} (allowed: {list(allowed)})")
    return items


def _safe_slug(value: str, *, fallback: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:64] or fallback


def _load_prompt_cases(prompt: str, prompt_file: Path | None, prompt_set: str = "single") -> list[PromptCase]:
    """Load one prompt, a built-in prompt set, or a text/jsonl prompt file."""
    if prompt_file is None:
        if prompt_set == "single":
            return [PromptCase(id="prompt-001", prompt=prompt)]
        if prompt_set not in PROMPT_SETS:
            raise ValueError(f"Unsupported prompt set: {prompt_set} (allowed: {sorted(PROMPT_SETS) + ['single']})")
        return list(PROMPT_SETS[prompt_set])
    cases: list[PromptCase] = []
    for line_index, raw_line in enumerate(prompt_file.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        prompt_id = f"prompt-{len(cases) + 1:03d}"
        prompt_text = line
        if prompt_file.suffix.lower() == ".jsonl":
            item = json.loads(line)
            prompt_text = str(item.get("prompt") or item.get("text") or item.get("caption") or "").strip()
            if not prompt_text:
                raise ValueError(f"{prompt_file}:{line_index} has no prompt/text/caption field")
            prompt_id = str(item.get("id") or item.get("name") or prompt_id)
        cases.append(PromptCase(id=_safe_slug(prompt_id, fallback=f"prompt-{len(cases) + 1:03d}"), prompt=prompt_text))
    if not cases:
        raise ValueError(f"No prompts found in {prompt_file}")
    return cases


def denoise_dmd_on_device(
    *,
    mx,
    dit,
    latents,
    encoder_hidden_states,
    freqs_cis,
    timesteps: list[int],
    renoise_by_step: list[np.ndarray],
    schedule,
    dmd_step,
    mx_dtype,
) -> tuple[np.ndarray, list[float]]:
    """Run the FastWan DMD loop entirely on the MLX device.

    Mirrors the loop in ``mlx_wan_prompt_to_video.py`` (fp32 affine math, MLX RNG
    re-noise) so the benchmark measures exactly the shipped path.

    Returns the final latents plus per-step wall times. The first step carries
    one-time costs (mx.compile tracing, kernel warm-up), so first-vs-steady
    step timing is how the benchmark separates cold-start from steady-state
    denoise throughput.

    All host-side tensors (timesteps, re-noise draws) are uploaded before the
    loop starts, so the per-step body performs no bulk host->device transfers
    and step timings measure device work rather than staging copies.
    """
    timesteps_mx = [mx.array([float(timestep)]).astype(mx.float32) for timestep in timesteps]
    renoise_mx = [mx.array(renoise).astype(mx.float32) for renoise in renoise_by_step]
    if timesteps_mx or renoise_mx:
        mx.eval(*timesteps_mx, *renoise_mx)

    step_times: list[float] = []
    for step_index, timestep in enumerate(timesteps):
        step_start = time.perf_counter()
        noise_input_latent = latents
        noise_pred = dit(latents.astype(mx_dtype), encoder_hidden_states, timesteps_mx[step_index], freqs_cis)

        noise_input_f32 = noise_input_latent.astype(mx.float32)
        pred_noise_f32 = noise_pred.astype(mx.float32)
        if step_index < len(timesteps) - 1:
            next_ts: float | None = float(timesteps[step_index + 1])
            renoise = renoise_mx[step_index]
        else:
            next_ts, renoise = None, None
        latents = dmd_step(
            latents=noise_input_f32,
            noise_input_latent=noise_input_f32,
            pred_noise=pred_noise_f32,
            schedule=schedule,
            timestep=float(timestep),
            next_timestep=next_ts,
            noise=renoise,
        ).astype(mx_dtype)
        mx.eval(latents)
        step_times.append(time.perf_counter() - step_start)
    return np.array(latents.astype(mx.float32)), step_times


def _peak_memory_bytes(mx) -> int:
    try:
        return int(mx.get_peak_memory())
    except Exception:  # noqa: BLE001 - best-effort telemetry only.
        return 0


def _latent_delta_metrics(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    diff = candidate.astype(np.float32) - baseline.astype(np.float32)
    mse = float(np.mean(np.square(diff)))
    signal = float(np.mean(np.square(baseline.astype(np.float32))))
    return {
        "latent_mse_vs_ref_mode": mse,
        "latent_snr_db_vs_ref_mode": float(10.0 * np.log10(signal / mse)) if mse > 0 else float("inf"),
    }


def _ms_ssim(reference_video: Path, candidate_video: Path, *, required: bool = False) -> float | None:
    """Mean MS-SSIM between two mp4s, via the repo's tested helper."""
    if not reference_video.exists() or not candidate_video.exists():
        return None
    try:
        from fastvideo.tests.utils import compute_video_ssim_torchvision
    except ImportError as exc:
        message = ("MS-SSIM is unavailable because `pytorch-msssim` is not installed. "
                   "Install FastVideo with the test extra, e.g. `uv pip install -e '.[mlx,test]'`, "
                   "or run without an SSIM assertion.")
        if required:
            raise RuntimeError(message) from exc
        print(f"{message} Skipping MS-SSIM.")
        return None

    try:
        ssim_values = compute_video_ssim_torchvision(str(reference_video), str(candidate_video), use_ms_ssim=True)
    except ImportError as exc:
        message = ("MS-SSIM is unavailable because `pytorch-msssim` is not installed. "
                   "Install FastVideo with the test extra, e.g. `uv pip install -e '.[mlx,test]'`, "
                   "or run without an SSIM assertion.")
        if required:
            raise RuntimeError(message) from exc
        print(f"{message} Skipping MS-SSIM.")
        return None
    return float(ssim_values[0])


def _markdown_table(rows: list[dict]) -> str:
    columns = [
        ("prompt_id", "prompt"),
        ("mode", "mode"),
        ("decoder", "decoder"),
        ("status", "status"),
        ("denoise_s", "denoise s"),
        ("decode_s", "decode s"),
        ("total_s", "total s"),
        ("peak_gib", "peak GiB"),
        ("ms_ssim_vs_ref", "MS-SSIM"),
        ("lpips_vs_ref", "LPIPS"),
    ]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            elif value is None:
                cells.append("-")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_metric(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "-"
    return str(value)


def _html_grid(rows: list[dict]) -> str:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("prompt_id", "prompt")), []).append(row)

    sections = []
    for prompt_id, group_rows in groups.items():
        prompt = next((str(row.get("prompt", "")) for row in group_rows if row.get("prompt")), "")
        cards = []
        for row in group_rows:
            title = f"{row.get('mode', '-')} / {row.get('decoder', '-')}"
            status = row.get("status", "-")
            video_path = row.get("video_path")
            if video_path:
                media = f'<video src="{html.escape(str(video_path))}" muted loop controls playsinline></video>'
            else:
                media = f'<div class="missing">No video<br>{html.escape(str(row.get("error", "")))}</div>'
            metrics = (f"status={status} · total={_format_metric(row.get('total_s'))}s · "
                       f"denoise={_format_metric(row.get('denoise_s'))}s · "
                       f"decode={_format_metric(row.get('decode_s'))}s · "
                       f"peak={_format_metric(row.get('peak_gib'))}GiB")
            cards.append("<article>"
                         f"<h3>{html.escape(title)}</h3>"
                         f"{media}"
                         f"<p>{html.escape(metrics)}</p>"
                         "</article>")
        sections.append("<section>"
                        f"<h2>{html.escape(prompt_id)}</h2>"
                        f"<p class=\"prompt\">{html.escape(prompt)}</p>"
                        f"<div class=\"grid\">{''.join(cards)}</div>"
                        "</section>")

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FastVideo MLX benchmark grid</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; background: #111; color: #eee; }
    button { margin-right: 8px; padding: 8px 12px; border-radius: 8px; border: 1px solid #555; background: #222; color: #eee; }
    section { margin-top: 28px; }
    .prompt { color: #bbb; max-width: 900px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
    article { background: #1b1b1b; border: 1px solid #333; border-radius: 12px; padding: 12px; }
    h1, h2, h3 { margin: 0 0 10px; }
    video { width: 100%; border-radius: 8px; background: #000; }
    article p { color: #bbb; font-size: 13px; line-height: 1.4; }
    .missing { min-height: 160px; display: grid; place-items: center; text-align: center; color: #f5b5b5; background: #2a1515; border-radius: 8px; padding: 12px; }
  </style>
</head>
<body>
  <h1>FastVideo MLX benchmark grid</h1>
  <p>Use the controls below to start/stop every clip together for side-by-side inspection.</p>
  <button onclick="for (const v of document.querySelectorAll('video')) { v.currentTime = 0; v.play(); }">Restart + play all</button>
  <button onclick="for (const v of document.querySelectorAll('video')) v.pause();">Pause all</button>
""" + "\n".join(sections) + """
</body>
</html>
"""


def _write_html_grid(rows: list[dict], output_dir: Path) -> Path:
    html_path = output_dir / "index.html"
    html_path.write_text(_html_grid(rows))
    return html_path


def _generate_cell(
    *,
    args,
    mode: str,
    decoder: str,
    checkpoint_path: Path,
    config_path: Path,
    encoder_hidden_states,
    freqs_cis,
    timesteps: list[int],
    latents_seed: np.ndarray,
    renoise_by_step: list[np.ndarray],
) -> Cell:
    import mlx.core as mx

    from fastvideo.mlx_runtime.fastwan import mlx_dit_from_diffusers_safetensors
    from fastvideo.mlx_runtime.sampling import MLXDMDSchedule, dmd_step
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    base_dtype, quantization = _mode_to_dtype_quant(mode)
    mx_dtype = _mx_dtype(mx, base_dtype)

    mx.clear_cache()
    mx.reset_peak_memory()
    load_start = time.perf_counter()
    load_source = "diffusers"
    if args.mlx_checkpoint_cache is not None:
        from fastvideo.mlx_runtime.checkpoint import (
            load_mlx_dit_checkpoint,
            save_mlx_dit_checkpoint,
        )

        mode_ckpt_dir = args.mlx_checkpoint_cache / mode
        if (mode_ckpt_dir / "mlx_dit.json").exists():
            dit = load_mlx_dit_checkpoint(mode_ckpt_dir)
            load_source = "mlx_checkpoint"
        else:
            dit = mlx_dit_from_diffusers_safetensors(
                checkpoint_path,
                config_path,
                dtype=base_dtype,
                quantization=quantization,
            )
            save_mlx_dit_checkpoint(dit, mode_ckpt_dir)
            load_source = "diffusers_then_saved"
    else:
        dit = mlx_dit_from_diffusers_safetensors(
            checkpoint_path,
            config_path,
            dtype=base_dtype,
            quantization=quantization,
        )
    load_s = time.perf_counter() - load_start
    load_peak = _peak_memory_bytes(mx)

    scheduler = FlowMatchEulerDiscreteScheduler(shift=args.flow_shift)
    schedule = MLXDMDSchedule.from_torch_scheduler(scheduler)

    latents = mx.array(latents_seed).astype(mx_dtype)
    mx.reset_peak_memory()
    denoise_start = time.perf_counter()
    latents_np, step_times = denoise_dmd_on_device(
        mx=mx,
        dit=dit,
        latents=latents,
        encoder_hidden_states=encoder_hidden_states.astype(mx_dtype),
        freqs_cis=freqs_cis,
        timesteps=timesteps,
        renoise_by_step=renoise_by_step,
        schedule=schedule,
        dmd_step=dmd_step,
        mx_dtype=mx_dtype,
    )
    denoise_s = time.perf_counter() - denoise_start
    denoise_peak = _peak_memory_bytes(mx)

    video_path = (args.output_dir / f"{args.current_prompt_id}" /
                  f"video_{mode}_{decoder}_{args.height}x{args.width}x{args.num_frames}.mp4")
    decode_start = time.perf_counter()
    decode_latents_to_video(
        model_root=args.model_root,
        latents_np=latents_np,
        output_path=video_path,
        fps=args.fps,
        device_arg=args.torch_device,
        dtype_arg=args.torch_dtype,
        backend=decoder,
        taehv_source_path=args.taehv_source_path,
        taehv_checkpoint_path=args.taehv_checkpoint_path,
        taehv_parallel=args.taehv_parallel,
    )
    decode_s = time.perf_counter() - decode_start

    metrics: dict[str, float | int | str | bool | None] = {
        "prompt_id": args.current_prompt_id,
        "prompt": args.current_prompt,
        "benchmark_preset": args.benchmark_preset,
        "mode": mode,
        "decoder": decoder,
        "status": "ok",
        "video_path": str(video_path.relative_to(args.output_dir)),
        "load_s": load_s,
        "load_source": load_source,
        "denoise_s": denoise_s,
        # The first step carries one-time costs (mx.compile tracing, kernel
        # warm-up); steady-state throughput is the median of the rest.
        "denoise_first_step_s": step_times[0] if step_times else None,
        "denoise_steady_step_s": (float(np.median(step_times[1:])) if len(step_times) > 1 else None),
        "decode_s": decode_s,
        "total_s": load_s + denoise_s + decode_s,
        "load_peak_gib": load_peak / (1024**3),
        "peak_gib": max(load_peak, denoise_peak) / (1024**3),
        "quantization": quantization or "none",
        "compute_dtype": base_dtype,
        "compile": os.environ.get("FASTVIDEO_MLX_COMPILE", "0") == "1",
        "fast_norm": os.environ.get("FASTVIDEO_MLX_FAST_NORM", "0") == "1",
        "mlx_memory_limit_gib": args.mlx_memory_limit_gib,
        "mlx_cache_limit_gib": args.mlx_cache_limit_gib,
        "mlx_disable_cache": args.mlx_disable_cache,
        "mlx_wired_limit_gib": args.mlx_wired_limit_gib,
        "torch_mps_high_watermark_ratio": args.torch_mps_high_watermark_ratio,
        "torch_mps_low_watermark_ratio": args.torch_mps_low_watermark_ratio,
    }
    return Cell(
        prompt_id=args.current_prompt_id,
        prompt=args.current_prompt,
        mode=mode,
        decoder=decoder,
        video_path=video_path,
        latents=latents_np,
        metrics=metrics,
    )


def main() -> None:
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--benchmark-preset", choices=tuple(BENCHMARK_PRESETS), default="default")
    preset_args, _ = preset_parser.parse_known_args()
    preset = BENCHMARK_PRESETS[preset_args.benchmark_preset]

    parser = argparse.ArgumentParser(description="MLX FastWan prove-out benchmark (latency + quality).")
    parser.add_argument("--benchmark-preset",
                        choices=tuple(BENCHMARK_PRESETS),
                        default=preset_args.benchmark_preset,
                        help="Memory-tier benchmark defaults. Explicit CLI flags override preset values.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--prompt", default="A paper boat sails through a shallow stream in a mossy forest.")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help=
        "Optional prompt set. Plain text uses one prompt per non-empty line; .jsonl accepts prompt/text/caption plus optional id/name.",
    )
    parser.add_argument(
        "--prompt-set",
        choices=("single", *PROMPT_SETS.keys()),
        default="single",
        help="Built-in standard prompt set. Ignored when --prompt-file is supplied.",
    )
    parser.add_argument("--height", type=int, default=preset.height)
    parser.add_argument("--width", type=int, default=preset.width)
    parser.add_argument("--num-frames", type=int, default=preset.num_frames)
    parser.add_argument("--dmd-denoising-steps", default="1000,757,522")
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--modes", default=preset.modes)
    parser.add_argument("--decoders", default=preset.decoders)
    parser.add_argument("--output-dir", type=Path, default=Path("video_samples/mlx_fastwan_bench"))
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--torch-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="External reference mp4 to score every cell against. Defaults to the fp16+wan-vae cell.",
    )
    parser.add_argument("--assert-min-ssim",
                        type=float,
                        default=None,
                        help="Fail if any cell's MS-SSIM vs the reference falls below this value.")
    parser.add_argument("--compile",
                        action="store_true",
                        help="Enable mx.compile on the DiT forward (sets FASTVIDEO_MLX_COMPILE=1).")
    parser.add_argument("--lpips", action="store_true", help="Also compute LPIPS (needs the `lpips` package).")
    parser.add_argument("--taehv-source-path", type=Path, default=None)
    parser.add_argument("--taehv-checkpoint-path", type=Path, default=None)
    parser.add_argument("--taehv-parallel", action="store_true")
    parser.add_argument(
        "--mlx-checkpoint-cache",
        type=Path,
        default=None,
        help="Directory of per-mode pre-quantized MLX checkpoints. The first cell of a mode "
        "converts from Diffusers weights and saves here (load_source=diffusers_then_saved); "
        "later cells and later runs reload without requantizing (load_source=mlx_checkpoint), "
        "which is also how the checkpoint load-time win is measured.",
    )
    add_memory_limit_args(
        parser,
        mlx_memory_limit_gib=preset.mlx_memory_limit_gib,
        mlx_disable_cache=preset.mlx_disable_cache,
        torch_mps_high_watermark_ratio=preset.torch_mps_high_watermark_ratio,
        torch_mps_low_watermark_ratio=preset.torch_mps_low_watermark_ratio,
    )
    args = parser.parse_args()

    if args.compile:
        os.environ["FASTVIDEO_MLX_COMPILE"] = "1"

    import mlx.core as mx

    runtime_limits = apply_memory_limits(
        mlx_memory_limit_gib=args.mlx_memory_limit_gib,
        mlx_cache_limit_gib=args.mlx_cache_limit_gib,
        mlx_disable_cache=args.mlx_disable_cache,
        mlx_wired_limit_gib=args.mlx_wired_limit_gib,
        torch_mps_high_watermark_ratio=args.torch_mps_high_watermark_ratio,
        torch_mps_low_watermark_ratio=args.torch_mps_low_watermark_ratio,
        mx_module=mx,
    ).as_metrics()
    import torch

    mx.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    modes = _parse_list(args.modes, ALLOWED_MODES, "modes")
    decoders = _parse_list(args.decoders, ALLOWED_DECODERS, "decoders")

    config_path = args.model_root / "transformer/config.json"
    checkpoint_path = args.model_root / "transformer/diffusion_pytorch_model.safetensors"
    config = json.loads(config_path.read_text())
    latent_frames = (args.num_frames - 1) // 4 + 1
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

    timesteps = [int(step.strip()) for step in args.dmd_denoising_steps.split(",") if step.strip()]
    # Keep DMD stochasticity identical across benchmark cells. Without this,
    # FP16/INT8/decoder comparisons can accidentally measure different re-noise
    # samples instead of only quantization or decoder differences.
    renoise_by_step = [
        torch.randn(latents_seed.shape, generator=generator, dtype=torch.float32).numpy()
        for _ in range(max(0,
                           len(timesteps) - 1))
    ]

    from fastvideo.mlx_runtime.fastwan import UnsupportedMLXQuantizationError

    prompt_cases = _load_prompt_cases(args.prompt, args.prompt_file, args.prompt_set)
    cells: list[Cell] = []
    unsupported_rows: list[dict] = []
    for prompt_case in prompt_cases:
        args.current_prompt_id = prompt_case.id
        args.current_prompt = prompt_case.prompt
        prompt_embeds = encode_prompt(
            model_root=args.model_root,
            prompt=prompt_case.prompt,
            max_sequence_length=args.max_sequence_length,
            device_arg=args.torch_device,
            dtype_arg=args.torch_dtype,
        )
        encoder_hidden_states = mx.array(prompt_embeds.numpy())

        for mode in modes:
            for decoder in decoders:
                print(f"=== cell: prompt={prompt_case.id} mode={mode} decoder={decoder} ===")
                try:
                    cells.append(
                        _generate_cell(
                            args=args,
                            mode=mode,
                            decoder=decoder,
                            checkpoint_path=checkpoint_path,
                            config_path=config_path,
                            encoder_hidden_states=encoder_hidden_states,
                            freqs_cis=freqs_cis,
                            timesteps=timesteps,
                            latents_seed=latents_seed,
                            renoise_by_step=renoise_by_step,
                        ))
                except UnsupportedMLXQuantizationError as exc:
                    # Record the cell as unsupported and keep sweeping: a partial
                    # report on this MLX build beats crashing the whole run.
                    print(f"skipping cell (unsupported by this MLX build): {exc}")
                    unsupported_rows.append({
                        "prompt_id": prompt_case.id,
                        "prompt": prompt_case.prompt,
                        "mode": mode,
                        "decoder": decoder,
                        "status": "unsupported_by_mlx",
                        "error": str(exc),
                    })

    if not cells:
        metrics_path = args.output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(unsupported_rows, indent=2))
        raise SystemExit(f"No benchmark cell could run: every requested mode is unsupported by this MLX build. "
                         f"Wrote {metrics_path}.")

    # Resolve one internal reference per prompt. A single external reference, if
    # supplied, is used for every prompt and only video metrics are computed.
    reference_by_prompt: dict[str, tuple[Path, np.ndarray | None]] = {}
    if args.reference is not None:
        for prompt_case in prompt_cases:
            reference_by_prompt[prompt_case.id] = (args.reference, None)
    else:
        for prompt_case in prompt_cases:
            prompt_cells = [c for c in cells if c.prompt_id == prompt_case.id]
            if not prompt_cells:
                continue
            ref_cell = next(
                (c for c in prompt_cells if c.mode == REFERENCE_MODE and c.decoder == REFERENCE_DECODER),
                prompt_cells[0],
            )
            reference_by_prompt[prompt_case.id] = (ref_cell.video_path, ref_cell.latents)
            print(f"Using internal reference cell for {prompt_case.id}: "
                  f"mode={ref_cell.mode} decoder={ref_cell.decoder}")

    lpips_fn = _load_lpips() if args.lpips else None

    rows: list[dict] = []
    failures: list[str] = []
    for cell in cells:
        reference_video, reference_latents = reference_by_prompt[cell.prompt_id]
        ms_ssim = _ms_ssim(Path(reference_video), cell.video_path, required=args.assert_min_ssim is not None)
        cell.metrics["ms_ssim_vs_ref"] = ms_ssim
        cell.metrics.update(runtime_limits)
        if reference_latents is not None:
            cell.metrics.update(_latent_delta_metrics(cell.latents, reference_latents))
        cell.metrics["lpips_vs_ref"] = (_lpips_between(lpips_fn, Path(reference_video), cell.video_path)
                                        if lpips_fn else None)
        if args.assert_min_ssim is not None and ms_ssim is not None and ms_ssim < args.assert_min_ssim:
            failures.append(
                f"{cell.prompt_id}/{cell.mode}/{cell.decoder}: MS-SSIM {ms_ssim:.4f} < {args.assert_min_ssim}")
        rows.append(dict(cell.metrics))
        print(json.dumps(cell.metrics, indent=2))
    rows.extend(unsupported_rows)

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(rows, indent=2))
    table_path = args.output_dir / "metrics.md"
    table = _markdown_table(rows)
    table_path.write_text(table + "\n")
    html_path = _write_html_grid(rows, args.output_dir)
    print("\n" + table)
    print(f"\nWrote {metrics_path}, {table_path}, and {html_path}")

    if failures:
        raise SystemExit("SSIM regression gate failed:\n  " + "\n  ".join(failures))


def _load_lpips() -> object | None:
    """Return an LPIPS model, or ``None`` if the optional dep is unavailable."""
    try:
        import lpips  # noqa: PLC0415 - optional dependency.
    except ImportError:
        print("LPIPS requested but the `lpips` package is not installed; skipping (install `.[eval]`).")
        return None
    return lpips.LPIPS(net="alex")


def _lpips_between(lpips_fn, reference_video: Path, candidate_video: Path) -> float | None:
    if lpips_fn is None or not reference_video.exists() or not candidate_video.exists():
        return None
    import torch

    ref = _read_video_frames(reference_video)
    cand = _read_video_frames(candidate_video)
    if ref is None or cand is None or ref.shape != cand.shape:
        return None
    # LPIPS expects NCHW in [-1, 1].
    ref_t = torch.from_numpy(ref).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    cand_t = torch.from_numpy(cand).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    with torch.no_grad():
        scores = lpips_fn(ref_t, cand_t)
    return float(scores.mean().item())


def _read_video_frames(path: Path) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        return None
    return np.stack(frames, axis=0)


if __name__ == "__main__":
    main()
