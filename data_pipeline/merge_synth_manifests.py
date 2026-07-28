# SPDX-License-Identifier: Apache-2.0
"""Compile per-worker manifest shards -> FastVideo videos2caption.json + merge.txt.
Idempotent; run anytime (progress check) or at the end. Dedups by idx, verifies mp4 exists."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    videos_dir = a.output_dir / "videos"
    shard_dir = a.output_dir / "manifest_shards"
    recs: dict[int, dict] = {}
    for sh in sorted(shard_dir.glob("worker_*.jsonl")):
        for ln in sh.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if (videos_dir / r["path"]).exists():
                recs[int(r["idx"])] = r
    ordered = [recs[k] for k in sorted(recs)]
    j = a.output_dir / "videos2caption.json"
    j.write_text(json.dumps(ordered, indent=2))
    (a.output_dir / "merge.txt").write_text(f"{videos_dir.resolve()},{j.resolve()}\n")
    # also count mp4s on disk (may exceed manifest if a worker died mid-write)
    on_disk = len(list(videos_dir.glob("vid_*.mp4")))
    print(f"manifest: {len(ordered)} entries; {on_disk} mp4s on disk -> {j}")

if __name__ == "__main__":
    main()
