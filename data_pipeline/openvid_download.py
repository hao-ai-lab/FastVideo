# SPDX-License-Identifier: Apache-2.0
"""Download + extract OpenVid / OpenVidHD video shards from HuggingFace.

Videos are only distributed as ~30-50 GB zip shards (no per-file fetch), so this
pulls whole shards, extracts the .mp4s into --videos-dir, and (unless --keep-zip)
deletes the zip. Pair with openvid_filter.py (which clips to keep) + extract_tracks_mp.py
(resolution/aspect probe + tracking). Idempotent: skips shards already extracted.

Some parts are HF-split (e.g. OpenVid_part102_partaa/ab) — those must be
concatenated before unzip; this handles single-file parts (all OpenVidHD parts).
"""
from __future__ import annotations
import argparse, zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "nkp37/OpenVid-1M"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True,
                    help="repo paths, e.g. OpenVidHD/OpenVidHD_part_1.zip OpenVid_part0.zip")
    ap.add_argument("--videos-dir", required=True, help="where .mp4s are extracted")
    ap.add_argument("--zip-dir", default=None, help="scratch for downloaded zips (default: <videos-dir>/../_zips)")
    ap.add_argument("--keep-zip", action="store_true", help="don't delete the zip after extract")
    ap.add_argument("--limit", type=int, default=0, help="extract at most N mp4s per shard (0=all; for smoke tests)")
    ap.add_argument("--only-list", default=None, help="file of basenames; extract only these (e.g. the ≥5s filtered list)")
    a = ap.parse_args()
    only = None
    if a.only_list:
        only = {l.strip() for l in open(a.only_list) if l.strip()}

    vdir = Path(a.videos_dir); vdir.mkdir(parents=True, exist_ok=True)
    zdir = Path(a.zip_dir) if a.zip_dir else vdir.parent / "_zips"
    zdir.mkdir(parents=True, exist_ok=True)
    marker_dir = vdir.parent / "_extracted"; marker_dir.mkdir(parents=True, exist_ok=True)

    for shard in a.shards:
        done_marker = marker_dir / (Path(shard).name + ".done")
        if done_marker.exists():
            print(f"[dl] skip {shard} (already extracted)", flush=True); continue
        print(f"[dl] downloading {shard} ...", flush=True)
        zp = hf_hub_download(REPO, shard, repo_type="dataset", local_dir=str(zdir))
        print(f"[dl] extracting {zp} -> {vdir}", flush=True)
        n = 0
        with zipfile.ZipFile(zp) as z:
            for m in z.namelist():
                if m.lower().endswith(".mp4"):
                    if only is not None and Path(m).name not in only:
                        continue
                    # flatten: write basename directly into videos-dir
                    (vdir / Path(m).name).write_bytes(z.read(m))
                    n += 1
                    if a.limit and n >= a.limit:
                        break
        done_marker.write_text(str(n))
        if not a.keep_zip:
            Path(zp).unlink(missing_ok=True)
        print(f"[dl] {shard}: extracted {n} mp4s", flush=True)
    print("[dl] done", flush=True)


if __name__ == "__main__":
    main()
