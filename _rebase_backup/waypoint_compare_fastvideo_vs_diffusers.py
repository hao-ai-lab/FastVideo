# SPDX-License-Identifier: Apache-2.0
"""
Run FastVideo Waypoint and Diffusers Waypoint with same prompt/seed, then compare.

Generates two videos and reports pixel-level diff + SSIM.

Usage:
  python scripts/waypoint_compare_fastvideo_vs_diffusers.py
  python scripts/waypoint_compare_fastvideo_vs_diffusers.py --prompt "..." --seed 1024 --num-frames 16
  python scripts/waypoint_compare_fastvideo_vs_diffusers.py --debug  # prompt_emb, VAE config, denoised drift
  python scripts/waypoint_compare_fastvideo_vs_diffusers.py --compare-only --fastvideo path1.mp4 --diffusers path2.mp4

On RunPod: run from repo root after cloning. Use --low-memory if 16GB VRAM.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure we run from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_REPO_ROOT)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def run_fastvideo(
    prompt: str,
    seed: int,
    num_frames: int,
    output_path: str,
    low_memory: bool = False,
    debug_output: str | None = None,
) -> bool:
    """Run FastVideo Waypoint pipeline."""
    cmd = [
        sys.executable,
        "examples/inference/basic/basic_waypoint.py",
        "--prompt", prompt,
        "--seed", str(seed),
        "--num_steps", str(num_frames),
        "--frames-per-step", "1",
        "--output", os.path.basename(output_path),
        "--output_dir", os.path.dirname(output_path) or ".",
    ]
    if low_memory:
        cmd.append("--low-memory")
    if debug_output:
        cmd.extend(["--debug-output", debug_output])
    print(f"[FastVideo] Running: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=_REPO_ROOT)
    return r.returncode == 0


def run_diffusers(
    prompt: str,
    seed: int,
    num_frames: int,
    output_path: str,
    debug_output: str | None = None,
) -> bool:
    """Run Diffusers Waypoint pipeline."""
    cmd = [
        sys.executable,
        "scripts/run_diffusers_waypoint.py",
        "--prompt", prompt,
        "--seed", str(seed),
        "--num-frames", str(num_frames),
        "--output", output_path,
    ]
    if debug_output:
        cmd.extend(["--debug-output", debug_output])
    print(f"[Diffusers] Running: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=_REPO_ROOT)
    return r.returncode == 0


def print_debug_comparison(
    fastvideo_debug_path: str,
    diffusers_debug_path: str,
) -> None:
    """Load debug JSONs and print prompt_emb, VAE config, denoised latent drift."""
    import json

    def load_json(path: str) -> dict | None:
        if not path or not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    fv = load_json(fastvideo_debug_path)
    df = load_json(diffusers_debug_path)
    if not fv and not df:
        print("\n--- Debug comparison: no debug files found ---")
        return
    print("\n--- Debug comparison ---")
    # prompt_emb
    fv_prompt = None
    for e in fv.get("entries", []) if fv else []:
        if e.get("tag") == "prompt_emb" and "prompt_emb" in e:
            fv_prompt = e["prompt_emb"]
            break
    df_prompt = df.get("prompt_emb") if df else None
    print("\nPrompt emb (mean, std, min, max):")
    if fv_prompt:
        print(
            "  FastVideo:  mean=%.4f std=%.4f min=%.4f max=%.4f" % (
                fv_prompt.get("mean", 0),
                fv_prompt.get("std", 0),
                fv_prompt.get("min", 0),
                fv_prompt.get("max", 0),
            ))
    else:
        print("  FastVideo:  (not available)")
    if df_prompt:
        print(
            "  Diffusers:  mean=%.4f std=%.4f min=%.4f max=%.4f" % (
                df_prompt.get("mean", 0),
                df_prompt.get("std", 0),
                df_prompt.get("min", 0),
                df_prompt.get("max", 0),
            ))
    else:
        print("  Diffusers:  (not available)")
    # VAE config
    fv_vae = None
    for e in fv.get("entries", []) if fv else []:
        if e.get("tag") == "vae_config":
            fv_vae = e
            break
    df_vae = df.get("vae_config", {}) if df else {}
    print("\nWorldEngine VAE scaling_factor / shift:")
    print(
        "  FastVideo:  scaling_factor=%s shift_factor=%s" % (
            fv_vae.get("scaling_factor", "N/A") if fv_vae else "N/A",
            fv_vae.get("shift_factor", "N/A") if fv_vae else "N/A",
        ))
    print(
        "  Diffusers:  scaling_factor=%s shift_factor=%s" % (
            df_vae.get("scaling_factor", "N/A"),
            df_vae.get("shift_factor", "N/A"),
        ))
    # Denoised latent drift
    fv_denoised = [
        e for e in (fv.get("entries", []) if fv else [])
        if e.get("tag") == "denoised"
    ]
    fv_denoised.sort(key=lambda x: x.get("frame", 0))
    df_denoised = df.get("frames", []) if df else []
    df_denoised = [d for d in df_denoised if "denoised" in d]
    if not df_denoised:
        df_denoised = [
            {**d, "denoised": d.get("vae_out")}
            for d in (df.get("frames", []) if df else [])
        ]
    print("\nDenoised latent mean by frame (brightness drift):")
    for i, d in enumerate(fv_denoised[:16]):
        den = d.get("denoised", {})
        mean = den.get("mean", "N/A")
        mean_s = "%.4f" % mean if isinstance(mean, (int, float)) else str(mean)
        print("  FastVideo frame %2d: mean=%s" % (d.get("frame", i), mean_s))
    if fv_denoised and len(fv_denoised) > 16:
        print("  ... (%d more frames)" % (len(fv_denoised) - 16))
    if df_denoised:
        print("  (Diffusers: decoded image mean; latent not exposed)")
    for i, d in enumerate(df_denoised[:16]):
        den = d.get("denoised") or d.get("vae_out", {})
        mean = den.get("mean", "N/A") if den else "N/A"
        mean_s = "%.4f" % mean if isinstance(mean, (int, float)) else str(mean)
        print("  Diffusers frame %2d: mean=%s" % (d.get("frame", i), mean_s))
    if df_denoised and len(df_denoised) > 16:
        print("  ... (%d more frames)" % (len(df_denoised) - 16))


def compare_videos(fastvideo_path: str, diffusers_path: str) -> None:
    """Compare two videos: pixel diff and SSIM."""
    import numpy as np
    from torchvision.io import read_video

    def load_frames(path: str):
        frames, _, _ = read_video(
            path, pts_unit="sec", output_format="TCHW"
        )
        return frames.numpy()

    if not os.path.exists(fastvideo_path):
        print(f"ERROR: FastVideo output not found: {fastvideo_path}")
        return
    if not os.path.exists(diffusers_path):
        print(f"ERROR: Diffusers output not found: {diffusers_path}")
        return

    f1 = load_frames(fastvideo_path)
    f2 = load_frames(diffusers_path)
    min_t = min(f1.shape[0], f2.shape[0])
    f1, f2 = f1[:min_t].astype(np.float32), f2[:min_t].astype(np.float32)

    print(f"\n--- Comparison ({min_t} frames) ---")
    print(f"FastVideo shape: {f1.shape}")
    print(f"Diffusers shape: {f2.shape}")

    if f1.shape != f2.shape:
        print("WARNING: Shape mismatch. Resize Diffusers to match FastVideo.")
        # Resize f2 to f1 shape if needed
        if f2.shape[2:] != f1.shape[2:]:
            import torch
            f2_t = torch.from_numpy(f2)
            f2_t = torch.nn.functional.interpolate(
                f2_t.permute(0, 2, 3, 1),
                size=(f1.shape[2], f1.shape[3]),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 3, 1, 2)
            f2 = f2_t.numpy()

    diff = np.abs(f1 - f2)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    exact_match = np.allclose(f1, f2, rtol=0, atol=0)

    print(f"\nPixel-level:")
    print(f"  Exact match: {exact_match}")
    print(f"  Max pixel diff: {max_diff:.2f}")
    print(f"  Mean pixel diff: {mean_diff:.4f}")

    # SSIM if available
    try:
        from fastvideo.tests.utils import compute_video_ssim_torchvision
        mean_ssim, min_ssim, max_ssim = compute_video_ssim_torchvision(
            fastvideo_path, diffusers_path, use_ms_ssim=True
        )
        print(f"\nSSIM (MS-SSIM):")
        print(f"  Mean: {mean_ssim:.4f}")
        print(f"  Min:  {min_ssim:.4f}")
        print(f"  Max:  {max_ssim:.4f}")
    except ImportError:
        print("\n(Install pytorch-msssim for SSIM comparison)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare FastVideo vs Diffusers Waypoint"
    )
    parser.add_argument(
        "--prompt",
        default="A first-person view of walking through a grassy field.",
        help="Text prompt (same for both)",
    )
    parser.add_argument("--seed", type=int, default=1024, help="Random seed")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames (16 = quick, 60 = 1 sec)",
    )
    parser.add_argument(
        "--output-dir",
        default="waypoint_compare_output",
        help="Directory for output videos",
    )
    parser.add_argument(
        "--skip-fastvideo",
        action="store_true",
        help="Skip FastVideo generation (use existing)",
    )
    parser.add_argument(
        "--skip-diffusers",
        action="store_true",
        help="Skip Diffusers generation (use existing)",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only run comparison (requires --fastvideo and --diffusers)",
    )
    parser.add_argument(
        "--fastvideo",
        default=None,
        help="Path to FastVideo output (for --compare-only)",
    )
    parser.add_argument(
        "--diffusers",
        default=None,
        help="Path to Diffusers output (for --compare-only)",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Use low-memory mode for FastVideo (16GB VRAM)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save and compare debug stats (prompt_emb, VAE config, denoised drift)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fastvideo_path = args.fastvideo or os.path.join(
        args.output_dir, "fastvideo_waypoint.mp4"
    )
    diffusers_path = args.diffusers or os.path.join(
        args.output_dir, "diffusers_waypoint.mp4"
    )

    if args.compare_only:
        if not args.fastvideo or not args.diffusers:
            parser.error("--compare-only requires --fastvideo and --diffusers")
        compare_videos(args.fastvideo, args.diffusers)
        return

    debug_dir = args.output_dir if args.debug else None
    fv_debug = (
        os.path.join(debug_dir, "fastvideo_debug.json")
        if debug_dir else None)
    df_debug = (
        os.path.join(debug_dir, "diffusers_debug.json")
        if debug_dir else None)

    ok = True
    if not args.skip_fastvideo:
        fv_dir = os.path.dirname(fastvideo_path) or args.output_dir
        fv_name = os.path.basename(fastvideo_path)
        ok = run_fastvideo(
            args.prompt,
            args.seed,
            args.num_frames,
            os.path.join(fv_dir, fv_name),
            low_memory=args.low_memory,
            debug_output=fv_debug,
        )
        if not ok:
            print("FastVideo generation failed.")
            sys.exit(1)

    if not args.skip_diffusers:
        ok = run_diffusers(
            args.prompt,
            args.seed,
            args.num_frames,
            diffusers_path,
            debug_output=df_debug,
        )
        if not ok:
            print("Diffusers generation failed.")
            sys.exit(1)

    compare_videos(fastvideo_path, diffusers_path)
    if args.debug and (fv_debug or df_debug):
        print_debug_comparison(fv_debug or "", df_debug or "")
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
