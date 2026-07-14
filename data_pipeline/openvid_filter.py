# SPDX-License-Identifier: Apache-2.0
"""Stage-1 (metadata) filter for OpenVid — from the dataset CSV.

The OpenVid CSV columns are: video, caption, aesthetic score, motion score,
temporal consistency score, camera motion, frame, fps, seconds.
There is NO resolution / aspect-ratio column, so 720p + 16:9 CANNOT be filtered
here — those are enforced downstream by probing (extract_tracks_mp.py
--min-height / --aspect-tol). This stage keeps clips that are long enough to yield
`num_frames` at the target fps, with optional motion/quality gates (tracking wants
motion; static/low-motion clips give degenerate tracks).

Output: newline-delimited video basenames to keep (feeds the shard extraction +
tracking stages). Also writes a captions sidecar (video -> caption) for later use.
"""
from __future__ import annotations
import argparse, csv, json, sys

csv.field_size_limit(sys.maxsize)  # captions are long


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="OpenVid-1M.csv or OpenVidHD.csv")
    ap.add_argument("--out", required=True, help="output: one video basename per line")
    ap.add_argument("--captions-out", default=None, help="optional JSON {video: caption}")
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--target-fps", type=int, default=24)
    ap.add_argument("--min-seconds", type=float, default=None,
                    help="override; default = num_frames/target_fps (+2%% margin)")
    ap.add_argument("--min-motion", type=float, default=0.0, help="drop clips below this motion score")
    ap.add_argument("--min-aesthetic", type=float, default=0.0)
    ap.add_argument("--exclude-static", action="store_true", help="drop camera motion == 'static'")
    a = ap.parse_args()

    min_secs = a.min_seconds if a.min_seconds is not None else (a.num_frames / a.target_fps) * 1.02
    kept = 0; total = 0; caps = {}
    with open(a.csv, newline="") as f, open(a.out, "w") as o:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            try:
                secs = float(row["seconds"]); motion = float(row["motion score"]); aes = float(row["aesthetic score"])
            except (KeyError, ValueError):
                continue
            if secs < min_secs:            continue
            if motion < a.min_motion:      continue
            if aes < a.min_aesthetic:      continue
            if a.exclude_static and row.get("camera motion", "").strip().lower() == "static":
                continue
            o.write(row["video"] + "\n"); kept += 1
            if a.captions_out is not None:
                caps[row["video"]] = row.get("caption", "")
    if a.captions_out is not None:
        json.dump(caps, open(a.captions_out, "w"))
    print(f"[filter] min_seconds={min_secs:.3f} min_motion={a.min_motion} min_aes={a.min_aesthetic} "
          f"exclude_static={a.exclude_static}")
    print(f"[filter] kept {kept}/{total} ({100*kept/max(total,1):.1f}%) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
