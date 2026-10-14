# SPDX-License-Identifier: Apache-2.0
"""Speed and fidelity of MiniMax-H3 video decoders on the same latents.

Every variant decodes identical latents. Fidelity is measured against the full
36-layer decoder run in FP32 without autocast (``full_fp32``), and against the
source pixels when the latents come from ``--video``. Timing covers the decoder
call only, after one warmup, synchronized on the device.

Variants are ``<family>_<modifier>_<modifier>...`` with family ``full`` (36-layer)
or ``light`` (26-layer), plus ``taeh3``. Modifiers:
  fp32      no autocast (``full_fp32`` is the fidelity reference)
  prod      fp16 autocast, as the H3 decode stage runs (default when no precision modifier)
  fp16w     decoder linears stored fp16
  bf16      decoder linears stored bf16, bf16 autocast
  int8q     quantize the decoder block linears to INT8 ConvRot here (Hadamard 256, per-channel abs-max),
            run by the same kernels as the shipped overlay; works for any decoder (e.g. full_int8q)
  int8      INT8 ConvRot overlay (V2 / Trim release decode)
  nvfp4     NVFP4 block linears (post-training, or --nvfp4-checkpoint), bf16 autocast
  r<N>      Hadamard rotation group N for nvfp4 (e.g. r256)
  unit      nvfp4 activation global scale 1.0 (default: dynamic per call)
  static    nvfp4 activation scale calibrated on the benchmark latents (one decode)
  unfused   nvfp4 without the fused inference kernels (the op-by-op eager path)
  d<K>      keep K evenly spaced decoder blocks (speed only unless trained)
  compile   torch.compile the ViT decoder

Example:
  python scripts/benchmarks/minimax_h3_vae/bench_decoder.py \\
      --full-vae-dir $FULL/vae --light-vae-dir $V2/vae --video assets/videos/robot_pouring.mp4 \\
      --variants full_fp32,full_prod,light_prod,light_int8,taeh3 --tile-batch 1,12
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REFERENCE = "full_fp32"
LPIPS_FRAME_STRIDE = 4


def load_h3_vae(vae_dir: str, device: torch.device, *, int8_overlay: bool = False) -> nn.Module:
    from safetensors.torch import load_file

    from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEConfig
    from fastvideo.models.vaes.minimax_h3_int8_convrot import (
        dense_vae_safetensors,
        find_int8_convrot_vae_path,
        overlay_minimax_h3_int8_convrot_decoder,
    )
    from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3

    config = MiniMaxH3VideoVAEConfig()
    with open(os.path.join(vae_dir, "config.json")) as handle:
        config.update_model_arch(json.load(handle))
    vae = AutoencoderKLMiniMaxH3(config).to(device)
    state: dict[str, torch.Tensor] = {}
    for path in dense_vae_safetensors(sorted(glob.glob(os.path.join(vae_dir, "*.safetensors")))):
        state.update(load_file(path))
    vae.load_state_dict(state, strict=True)
    del state
    if int8_overlay:
        overlay_path = find_int8_convrot_vae_path(vae_dir)
        if overlay_path is None:
            raise FileNotFoundError(f"No INT8 ConvRot overlay next to {vae_dir}")
        overlay_minimax_h3_int8_convrot_decoder(vae, overlay_path)
    return vae.eval().requires_grad_(False)


def store_decoder_linears_fp16(vae: nn.Module, dtype: torch.dtype = torch.float16) -> None:
    """Keep decoder GEMM weights in ``dtype`` so autocast does not recast them per call."""
    for module in vae.decoder.modules():
        if type(module) is nn.Linear:
            nn.Module.to(module, dtype=dtype)


@torch.no_grad()
def quantize_decoder_int8_convrot(vae: nn.Module, group_size: int = 256) -> int:
    """Replace decoder block linears with INT8 ConvRot linears (per-channel abs-max scales)."""
    from fastvideo.models.vaes.minimax_h3_int8_convrot import _int8_linear_from_tensors, rotate_activation
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import nvfp4_decoder_linear_names

    marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size}
    names = nvfp4_decoder_linear_names(vae.decoder)
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = vae.decoder.get_submodule(parent_name)
        linear = parent[int(child)] if child.isdigit() else getattr(parent, child)
        weight = rotate_activation(linear.weight.float(), group_size)
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
        codes = (weight / scale).round().clamp(-127, 127).to(torch.int8)
        bias = linear.bias.float() if linear.bias is not None else None
        replacement = _int8_linear_from_tensors(codes, scale, bias, marker).to(linear.weight.device)
        if child.isdigit():
            parent[int(child)] = replacement
        else:
            setattr(parent, child, replacement)
    return len(names)


def read_frames(path: str, num_frames: int) -> torch.Tensor:
    """First ``num_frames`` RGB frames as ``[T, C, H, W]`` uint8 (torchcodec, else PyAV)."""
    try:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(path)
        if len(decoder) < num_frames:
            raise ValueError(f"{path} has {len(decoder)} frames, need {num_frames}")
        return decoder.get_frames_in_range(0, num_frames).data
    except (ImportError, OSError, RuntimeError):
        import av

        frames = []
        with av.open(path) as container:
            for frame in container.decode(video=0):
                frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
                if len(frames) == num_frames:
                    break
        if len(frames) < num_frames:
            raise ValueError(f"{path} has {len(frames)} frames, need {num_frames}") from None
        return torch.stack(frames)


def read_video(path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    """Return ``[1, 3, T, H, W]`` uint8 on CPU, center-cropped and resized."""
    frames = read_frames(path, num_frames)  # T, C, H, W uint8
    _, _, src_h, src_w = frames.shape
    scale = max(height / src_h, width / src_w)
    resized = F.interpolate(frames.float(),
                            size=(round(src_h * scale), round(src_w * scale)),
                            mode="bilinear",
                            antialias=True,
                            align_corners=False)
    top = (resized.shape[-2] - height) // 2
    left = (resized.shape[-1] - width) // 2
    cropped = resized[..., top:top + height, left:left + width].clamp(0, 255).round().to(torch.uint8)
    return cropped.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def timed(fn: Callable[[], torch.Tensor], repeats: int) -> tuple[torch.Tensor, list[float]]:
    out = fn()  # warmup
    times = []
    for _ in range(repeats):
        del out
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return out, times


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.float(), b.float()).item()
    return float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def frames_for_metrics(video: torch.Tensor) -> torch.Tensor:
    """``[1, 3, T, H, W]`` in [0, 1] -> ``[T', 3, H, W]`` every ``LPIPS_FRAME_STRIDE`` frames."""
    return video[0, :, ::LPIPS_FRAME_STRIDE].permute(1, 0, 2, 3)


class Fidelity:

    def __init__(self, device: torch.device) -> None:
        try:
            import lpips
            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except ImportError:
            self.lpips = None
        try:
            from pytorch_msssim import ssim
            self.ssim = ssim
        except ImportError:
            self.ssim = None

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        if pred.shape != target.shape:
            return {"shape_mismatch": 1.0}
        result = {"psnr": psnr(pred, target)}
        p, t = frames_for_metrics(pred).float(), frames_for_metrics(target).float()
        if self.ssim is not None:
            result["ssim"] = float(self.ssim(p, t, data_range=1.0).item())
        if self.lpips is not None:
            result["lpips"] = float(self.lpips(p * 2 - 1, t * 2 - 1).mean().item())
        return result


def decode_fn(vae: nn.Module, z: torch.Tensor, *, autocast: torch.dtype | None) -> Callable[[], torch.Tensor]:

    @torch.no_grad()
    def run() -> torch.Tensor:
        ctx = (torch.autocast(device_type="cuda", dtype=autocast)
               if autocast is not None else contextlib.nullcontext())
        with ctx:
            sample = vae.decode(z, return_dict=False)[0]
        return vae.denormalize_pixels(sample.float()).clamp_(0, 1)

    return run


def taeh3_fn(vae: nn.Module, z: torch.Tensor) -> Callable[[], torch.Tensor]:
    from fastvideo.models.vaes.minimax_h3_taeh3 import decode_ncthw_latents_taeh3

    normalized = vae.normalize_latents(z)

    @torch.no_grad()
    def run() -> torch.Tensor:
        return decode_ncthw_latents_taeh3(normalized, device=z.device).float()

    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full-vae-dir", required=True, help="vae/ folder of the full 36-layer H3 VAE")
    parser.add_argument("--light-vae-dir", help="vae/ folder of the 26-layer light VAE (+ INT8 overlay)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="encode this clip with the full VAE encoder")
    source.add_argument("--latents", help="glob of raw (denormalized) NCTHW latents saved with torch.save")
    source.add_argument("--holdout-data-root",
                        action="append",
                        help="train_qad.py data roots: decode its held-out clips and compare to their source videos")
    parser.add_argument("--holdout-count", type=int, default=6, help="must match train_qad.py --eval-clips")
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--variants", default="full_fp32,full_prod,full_fp16w,light_prod,light_int8,taeh3")
    parser.add_argument("--tile-batch",
                        default="1",
                        help="comma list of FASTVIDEO_H3_VAE_TILE_BATCH values; 'auto' = one call per tile grid")
    parser.add_argument("--overlap", default="64", help="comma list of spatial tile overlaps in pixels")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--nvfp4-checkpoint", help="QAD checkpoint loaded into every nvfp4 variant")
    parser.add_argument("--checkpoint",
                        action="append",
                        default=[],
                        help="VARIANT=PATH QAD checkpoint for one variant (also used by its _compile form)")
    parser.add_argument("--profile", help="comma list of variants to profile (first tile batch/overlap only)")
    parser.add_argument("--wandb-project", help="log every row and a summary table to this W&B project")
    parser.add_argument("--wandb-name")
    return parser.parse_args()


def flatten(row: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in row.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}/"))
        elif not isinstance(value, list):
            flat[f"{prefix}{key}"] = value
    return flat


def keep_blocks(decoder: nn.Module, count: int) -> None:
    """Keep ``count`` evenly spaced transformer blocks (first and last included)."""
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import keep_evenly_spaced_blocks

    keep_evenly_spaced_blocks(decoder, count)


def variant_checkpoint(name: str, args: argparse.Namespace) -> str | None:
    base = "_".join(mod for mod in name.split("_") if mod not in ("compile", "unfused"))
    mapping = dict(item.split("=", 1) for item in args.checkpoint)
    return mapping.get(base) or mapping.get(name) or args.nvfp4_checkpoint


def build_variant(name: str, args: argparse.Namespace, device: torch.device,
                  cache: dict[str, nn.Module]) -> tuple[nn.Module, torch.dtype | None]:
    """Return (vae, autocast dtype or None) for a variant name; VAEs are cached by name."""
    family, *mods = name.split("_")
    if family not in ("full", "light"):
        raise ValueError(f"Unknown variant {name}")
    if family == "light" and not args.light_vae_dir:
        raise ValueError(f"{name} needs --light-vae-dir")
    autocast: torch.dtype | None = torch.bfloat16 if ("nvfp4" in mods or "bf16" in mods) else torch.float16
    if "fp32" in mods:
        autocast = None
    if name not in cache:
        vae = load_h3_vae(args.full_vae_dir if family == "full" else args.light_vae_dir,
                          device,
                          int8_overlay="int8" in mods)
        for mod in mods:
            if mod.startswith("d") and mod[1:].isdigit():
                keep_blocks(vae.decoder, int(mod[1:]))
        if "fp16w" in mods:
            store_decoder_linears_fp16(vae)
        if "bf16" in mods:
            store_decoder_linears_fp16(vae, torch.bfloat16)
        if "int8q" in mods:
            quantize_decoder_int8_convrot(vae)
        if "nvfp4" in mods:
            from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import convert_decoder_to_nvfp4

            rotation = next((int(m[1:]) for m in mods if m.startswith("r") and m[1:].isdigit()), None)
            act_scale = "unit" if "unit" in mods else "static" if "static" in mods else "dynamic"
            convert_decoder_to_nvfp4(vae.decoder,
                                     rotation_group=rotation,
                                     compute_dtype=torch.bfloat16,
                                     act_scale=act_scale)
            checkpoint = variant_checkpoint(name, args)
            if checkpoint:
                state = torch.load(checkpoint, map_location=device)
                vae.decoder.load_state_dict(state["decoder"], strict=True)
                vae.post_quant_conv.load_state_dict(state["post_quant_conv"], strict=True)
            vae.requires_grad_(False)
            if "unfused" in mods:
                vae.decoder.fused_blocks_forward = None
        if "compile" in mods:
            vae.decoder = torch.compile(vae.decoder, dynamic=False)
        cache[name] = vae
    return cache[name], autocast


def profile_decode(fn: Callable[[], torch.Tensor], path: Path) -> None:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    path.write_text(table)
    print(table, flush=True)


class Clip(NamedTuple):
    name: str
    latents: torch.Tensor  # raw NCTHW on CPU
    source: torch.Tensor | None  # [1, 3, T, H, W] uint8 on CPU, aligned to the decoded geometry


def holdout_clips(args: argparse.Namespace, vae: nn.Module) -> list[Clip]:
    """The exact held-out clips of train_qad.py (never trained on), with their source videos."""
    import sys

    import pyarrow.parquet as pq

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "distill" / "minimax_h3_nvfp4_decoder"))
    from train_qad import list_parquets, load_holdout

    files = list_parquets(args.holdout_data_root)
    latents, held = load_holdout(files, args.holdout_count, True, vae.latents_mean.cpu(), vae.latents_std.cpu())
    clips = []
    for z, (path, row_index) in zip(latents, sorted(held, key=lambda item: files.index(item[0])), strict=True):
        row = pq.ParquetFile(path).read_row_group(0, columns=["id"]).slice(row_index, 1).to_pylist()[0]
        source_dir = Path(path).parents[2]
        raw_path = next(
            json.loads(line)["raw_video_path"] for line in open(source_dir / "MANIFEST_rows.jsonl")
            if json.loads(line).get("conditioning_id") == row["id"])
        _, _, frames, height, width = vae.decoded_pixel_shape(z.shape)
        pixels = read_frames(raw_path, frames).float()
        pixels = F.interpolate(pixels, size=(height, width), mode="bilinear", antialias=True, align_corners=False)
        source = pixels.clamp(0, 255).round().to(torch.uint8).permute(1, 0, 2, 3).unsqueeze(0)
        clips.append(Clip(f"{source_dir.name}_{row['id']}", z, source))
    return clips


def load_clips(args: argparse.Namespace, vae: nn.Module, out_dir: Path) -> list[Clip]:
    if args.video:
        pixels = read_video(args.video, args.num_frames, args.height, args.width)
        with torch.no_grad():
            z = vae.encode_pixels(pixels).latent_dist.mode().cpu()
        torch.save(z, out_dir / "latents.pt")
        return [Clip(Path(args.video).stem, z, pixels)]
    if args.latents:
        paths = sorted(glob.glob(args.latents))
        if not paths:
            raise FileNotFoundError(args.latents)
        return [Clip(Path(path).stem, torch.load(path, map_location="cpu").float(), None) for path in paths]
    return holdout_clips(args, vae)


def grid_tiles(vae: nn.Module, latent_shape: torch.Size) -> int:
    height = latent_shape[-2] * vae.spatial_compression_ratio
    width = latent_shape[-1] * vae.spatial_compression_ratio
    rows = len(vae._split_tiles(height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)[0])
    cols = len(vae._split_tiles(width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width)[0])
    return rows * cols


def summarize(rows: list[dict]) -> str:
    """Markdown table: mean over clips per (resolution, variant)."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["resolution"], row["variant"]), []).append(row)
    lines = [
        "| resolution | variant | clips | decode s | LPIPS vs original | PSNR vs original | LPIPS vs source | "
        "PSNR vs source |", "|---|---|---|---|---|---|---|---|"
    ]

    def mean(items: list[dict], key: str, metric: str) -> str:
        values = [r[key][metric] for r in items if key in r and metric in r[key]]
        return f"{sum(values) / len(values):.4f}" if values and metric == "lpips" else (
            f"{sum(values) / len(values):.2f}" if values else "-")

    for (resolution, variant), items in groups.items():
        seconds = sum(r["median_s"] for r in items) / len(items)
        lines.append(f"| {resolution} | {variant} | {len(items)} | {seconds:.3f} | "
                     f"{mean(items, 'vs_full_fp32', 'lpips')} | {mean(items, 'vs_full_fp32', 'psnr')} | "
                     f"{mean(items, 'vs_source', 'lpips')} | {mean(items, 'vs_source', 'psnr')} |")
    return "\n".join(lines)


def main() -> None:  # noqa: C901 - one benchmark loop
    args = parse_args()
    device = torch.device("cuda")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, nn.Module] = {}
    reference_vae, _ = build_variant(REFERENCE, args, device, cache)
    clips = load_clips(args, reference_vae, out_dir)
    print(f"{len(clips)} clips: {[(c.name, tuple(c.latents.shape)) for c in clips]}", flush=True)

    run = None
    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_name, job_type="decoder-bench", config=vars(args))

    fidelity = Fidelity(device)
    rows = []
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if REFERENCE in variants:
        variants.remove(REFERENCE)
    variants.insert(0, REFERENCE)
    profiled = set((args.profile or "").split(",")) - {""}
    done_profiles: set[str] = set()
    calibrated: set[str] = set()
    for clip in clips:
        z = clip.latents.to(device=device, dtype=torch.float32)
        source = clip.source.to(device).float().div(255) if clip.source is not None else None
        resolution = f"{z.shape[-1] * 16}x{z.shape[-2] * 16}"
        if args.save_videos and clip.source is not None:
            torch.save(clip.source, out_dir / f"{clip.name}__source.pt")
        reference = None
        for name in variants:
            batches = [grid_tiles(reference_vae, z.shape)] if args.tile_batch == "auto" else [
                int(v) for v in args.tile_batch.split(",")
            ]
            for tile_batch in batches:
                for overlap in (int(v) for v in args.overlap.split(",")):
                    if name == "taeh3" and (tile_batch, overlap) != (batches[0], 64):
                        continue
                    os.environ["FASTVIDEO_H3_VAE_TILE_BATCH"] = str(tile_batch)
                    if name == "taeh3":
                        fn = taeh3_fn(reference_vae, z)
                    else:
                        vae, autocast = build_variant(name, args, device, cache)
                        vae.tile_sample_min_overlap_height = overlap
                        vae.tile_sample_min_overlap_width = overlap
                        fn = decode_fn(vae, z, autocast=autocast)
                        if ("static" in name.split("_") and name not in calibrated
                                and not variant_checkpoint(name, args)):
                            from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import calibrate_static_scales
                            calibrate_static_scales(vae, fn)
                            calibrated.add(name)
                    torch.cuda.synchronize()
                    resident = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    video, times = timed(fn, args.repeats)
                    decode_peak = (torch.cuda.max_memory_allocated() - resident) / 2**30
                    if name in profiled and name not in done_profiles:
                        profile_decode(fn, out_dir / f"profile_{name}_{resolution}.txt")
                        done_profiles.add(name)
                    row = {
                        "clip": clip.name,
                        "resolution": resolution,
                        "variant": name,
                        "tile_batch": tile_batch,
                        "overlap": overlap,
                        "median_s": sorted(times)[len(times) // 2],
                        "decode_peak_gib": decode_peak,
                        "shape": list(video.shape),
                    }
                    if reference is None:
                        reference = video
                    else:
                        row["vs_full_fp32"] = fidelity(video, reference)
                    if source is not None:
                        row["vs_source"] = fidelity(video, source)
                    # Compiled variants decode the same numbers as their eager form; save one copy.
                    if (args.save_videos and "compile" not in name.split("_")
                            and (tile_batch, overlap) == (batches[0], int(args.overlap.split(",")[0]))):
                        torch.save((video * 255).round().to(torch.uint8).cpu(), out_dir / f"{clip.name}__{name}.pt")
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    if run is not None:
                        run.log(flatten(row))
                    del video
        del reference, z, source
        torch.cuda.empty_cache()
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    table = summarize(rows)
    (out_dir / "summary.md").write_text(table + "\n")
    print(table, flush=True)
    if run is not None:
        flat_rows = [flatten(row) for row in rows]
        columns = sorted({key for row in flat_rows for key in row})
        run.log({"results": wandb.Table(columns=columns, data=[[row.get(c) for c in columns] for row in flat_rows])})
        run.finish()


if __name__ == "__main__":
    main()
