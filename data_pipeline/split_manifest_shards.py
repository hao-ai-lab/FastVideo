# SPDX-License-Identifier: Apache-2.0
"""Split a videos2caption.json manifest into NUM_SHARDS shards for data-parallel preprocess.

The i2v_track preprocess reads a merge.txt (``<clips_dir>,<json>``) whose JSON is a
*list* of dicts (json.load, NOT jsonlines) — see
fastvideo/dataset/preprocessing_datasets.py:_load_raw_data. So each shard is itself a
JSON array + its own merge.txt, and each shard is fed to one single-GPU v1_preprocess.py
process.

Also injects a ``duration`` field when missing: FrameSamplingStage.should_keep drops any
row with ``duration is None`` (preprocessing_datasets.py:183), and OpenVid's
videos2caption.json carries only num_frames/fps. duration = num_frames / fps.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="videos2caption.json (JSON array)")
    ap.add_argument("--clips-dir", required=True, help="dir with the .mp4 clips (merge.txt col 1)")
    ap.add_argument("--out-dir", required=True, help="output shards dir (shard_XXXXX/ created inside)")
    ap.add_argument("--num-shards", type=int, required=True)
    a = ap.parse_args()

    items = json.load(open(a.manifest))
    filled = 0
    for it in items:
        if it.get("duration") is None:
            nf, fps = it.get("num_frames"), it.get("fps")
            if nf and fps:
                it["duration"] = float(nf) / float(fps)
                filled += 1

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Contiguous shards keep each shard's clips locally clustered on disk.
    n = a.num_shards
    total = len(items)
    per = (total + n - 1) // n
    written = 0
    for k in range(n):
        chunk = items[k * per:(k + 1) * per]
        sd = out / f"shard_{k:05d}"
        sd.mkdir(parents=True, exist_ok=True)
        mani = sd / "videos2caption.json"
        mani.write_text(json.dumps(chunk, indent=2))
        (sd / "merge.txt").write_text(f"{a.clips_dir},{mani}")
        written += len(chunk)
    print(f"[split] {total} rows ({filled} got injected duration) -> {n} shards "
          f"(~{per}/shard) under {out}; wrote {written} rows total")


if __name__ == "__main__":
    main()
