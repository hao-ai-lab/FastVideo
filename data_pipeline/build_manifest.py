# SPDX-License-Identifier: Apache-2.0
"""Build videos2caption.json + merge.txt for the final track dataset.

Pairs each 121f/720p clip (--clips-dir) with its CoTracker npz (--tracks-dir) and
caption (--captions JSON {video.mp4: caption}). Output is exactly the manifest
`fastvideo/pipelines/preprocess/v1_preprocess.py --preprocess_task i2v_track` consumes
(same format used for the overfit run), so the dataset is training-ready.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--captions", required=True, help="JSON {video.mp4: caption}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    a = ap.parse_args()

    caps = json.load(open(a.captions))
    tracks = {p.stem: p for p in Path(a.tracks_dir).glob("*.npz")}
    items, missing_track, missing_cap = [], 0, 0
    for clip in sorted(Path(a.clips_dir).glob("*.mp4")):
        tr = tracks.get(clip.stem)
        if tr is None:
            missing_track += 1; continue
        cap = caps.get(clip.name, "")
        if not cap:
            missing_cap += 1
        items.append({
            "path": clip.name,
            "cap": [cap],
            "points_path": str(tr.resolve()),
            "fps": float(a.fps),
            "num_frames": a.num_frames,
            "resolution": {"width": a.width, "height": a.height},
        })
    outd = Path(a.out_dir); outd.mkdir(parents=True, exist_ok=True)
    (outd / "videos2caption.json").write_text(json.dumps(items, indent=2))
    (outd / "merge.txt").write_text(f"{a.clips_dir},{outd / 'videos2caption.json'}")
    print(f"[manifest] {len(items)} clips paired (clip+track+caption); "
          f"skipped {missing_track} clips w/o track, {missing_cap} w/o caption")
    print(f"[manifest] wrote {outd}/videos2caption.json + merge.txt")


if __name__ == "__main__":
    main()
