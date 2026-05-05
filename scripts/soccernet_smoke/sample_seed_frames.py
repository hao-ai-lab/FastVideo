"""Sample 1 random main-camera-center frame per domestic league as I2V seeds.

Reads action-refined segments (v4_all), picks one fragment per league
(stratified across the 5 domestic leagues; UEFA Champions League dropped),
picks a uniform random timestamp inside the fragment, and ffmpeg-extracts
a single PNG. Outputs `{out_dir}/{league}.png` plus `seeds.json` manifest.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import pandas as pd

DEFAULT_LEAGUES = (
    "england_epl",
    "france_ligue-1",
    "germany_bundesliga",
    "italy_serie-a",
    "spain_laliga",
)


def hybrid_seek_extract(mkv: Path, target_s: float, out_png: Path) -> None:
    """Mirror analysis/scripts/20_refine_validate_clips.py:183-219 but for 1 PNG."""
    coarse = max(0.0, target_s - 5.0)
    fine = target_s - coarse
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{coarse:.3f}",
        "-i", str(mkv),
        "-ss", f"{fine:.3f}",
        "-frames:v", "1",
        "-q:v", "2",
        "-an",
        "-hide_banner", "-loglevel", "error",
        str(out_png),
    ]
    subprocess.run(cmd, check=True, timeout=120)
    if not out_png.exists() or out_png.stat().st_size < 2048:
        raise RuntimeError(f"ffmpeg produced no/tiny output for {mkv} @ {target_s:.2f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--segments",
        default="/leonardo_work/AIFAC_P02_082/data/analysis/out/19_refine/v4_all/refined_segments.parquet",
    )
    ap.add_argument(
        "--video-root",
        default="/leonardo_work/AIFAC_P02_082/data/SoccerNet_raw/games_720p",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-dur-s", type=float, default=5.0)
    ap.add_argument("--margin-s", type=float, default=1.0,
                    help="Avoid the first/last N seconds of the fragment when picking pos.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(args.video_root)

    df = pd.read_parquet(args.segments)
    df = df[df["league"].isin(DEFAULT_LEAGUES)]
    df = df[df["dur_s"] >= args.min_dur_s].reset_index(drop=True)
    print(f"[load] {len(df)} fragments in {df['league'].nunique()} leagues "
          f"after dur_s >= {args.min_dur_s} filter")

    rng = random.Random(args.seed)
    seeds = []

    for league in DEFAULT_LEAGUES:
        league_df = df[df["league"] == league]
        if league_df.empty:
            print(f"[skip] {league}: no fragments")
            continue
        row = league_df.sample(n=1, random_state=rng.randint(0, 2**31 - 1)).iloc[0]
        lo = row["pos_start_ms"] + int(args.margin_s * 1000)
        hi = row["pos_end_ms"] - int(args.margin_s * 1000)
        pos_ms = rng.randint(lo, hi)
        target_s = pos_ms / 1000.0

        mkv = video_root / row["league"] / row["season"] / row["game"] / f"{row['half']}_720p.mkv"
        if not mkv.exists():
            raise FileNotFoundError(mkv)

        out_png = out_dir / f"{league}.png"
        print(f"[extract] {league}: {mkv.name} @ {target_s:.2f}s -> {out_png.name}")
        hybrid_seek_extract(mkv, target_s, out_png)

        seeds.append({
            "league": league,
            "season": row["season"],
            "game": row["game"],
            "half": int(row["half"]),
            "pos_ms": int(pos_ms),
            "frag_pos_start_ms": int(row["pos_start_ms"]),
            "frag_pos_end_ms": int(row["pos_end_ms"]),
            "frag_dur_s": float(row["dur_s"]),
            "mkv": str(mkv),
            "png": str(out_png),
        })

    manifest = out_dir / "seeds.json"
    manifest.write_text(json.dumps(seeds, indent=2))
    print(f"[done] {len(seeds)} seeds -> {manifest}")


if __name__ == "__main__":
    main()
