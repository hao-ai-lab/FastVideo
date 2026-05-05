"""Zero-shot I2V smoke test on 5 SoccerNet seed frames using FastWan2.2-TI2V-5B.

Sequential, single-GPU + CPU offloading. Loads VideoGenerator once and
iterates over the seed PNGs produced by sample_seed_frames.py.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

from fastvideo import VideoGenerator

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.environ.get(
    "WAN22_MODEL_PATH",
    "/leonardo_work/AIFAC_P02_082/.cache/FastWan2.2-TI2V-5B-Diffusers",
)
SEED_DIR = REPO_ROOT / "outputs/wan22_soccernet_smoke/seeds"
OUT_DIR = REPO_ROOT / "outputs/wan22_soccernet_smoke/videos"
MANIFEST_PATH = REPO_ROOT / "outputs/wan22_soccernet_smoke/manifest.csv"

PROMPT = (
    "A wide broadcast shot of a professional soccer match. "
    "Players move across the green pitch under stadium lights, "
    "the camera follows the play, realistic motion."
)


def main() -> None:
    seeds_path = SEED_DIR / "seeds.json"
    if not seeds_path.exists():
        raise FileNotFoundError(
            f"{seeds_path} not found. Run sample_seed_frames.py "
            f"--out-dir {SEED_DIR} first (or use submit_wan22_i2v.sbatch, "
            "which chains the two steps).")
    seeds = json.loads(seeds_path.read_text())
    print(f"[plan] {len(seeds)} generations, model={MODEL_PATH}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_load = time.time()
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )
    print(f"[load] VideoGenerator ready in {time.time() - t_load:.1f}s", flush=True)

    rows = []
    for i, s in enumerate(seeds):
        league = s["league"]
        png = Path(s["png"])
        if not png.is_absolute():
            png = REPO_ROOT / png
        out_sub = OUT_DIR / league
        out_sub.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i+1}/{len(seeds)}] {league}: {png.name}", flush=True)
        t0 = time.time()
        generator.generate_video(
            PROMPT,
            image_path=str(png),
            output_path=str(out_sub),
            save_video=True,
            seed=42,
        )
        wall = time.time() - t0
        print(f"[{i+1}/{len(seeds)}] {league}: done in {wall:.1f}s", flush=True)

        # Find the produced mp4 (FastVideo names it from prompt; we just glob).
        mp4s = sorted(out_sub.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        out_mp4 = str(mp4s[0]) if mp4s else ""
        rows.append({
            "league": league,
            "seed_png": str(png),
            "out_dir": str(out_sub),
            "out_mp4": out_mp4,
            "wallclock_s": f"{wall:.2f}",
            "game": s["game"],
            "half": s["half"],
            "pos_ms": s["pos_ms"],
        })

        # Write manifest after every step so partial results are durable.
        with MANIFEST_PATH.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\n[done] {len(rows)} videos -> {OUT_DIR} | manifest: {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
