# SPDX-License-Identifier: Apache-2.0
"""Robust OpenVidHD (1080p, 16:9) full downloader for the trackwan bidir dataset.

OpenVidHD parts are a mix of single-file `.zip` (parts 1-14) and SPLIT parts
(`OpenVidHD_part_<i>_part_aa` + `_part_ab`) that must be concatenated before unzip
(mirrors the official download_scripts/download_OpenVid.py). This enumerates the HF
repo, plans each part, downloads (single or split+cat), extracts only the mp4s in
--only-list (the >=5s filtered set), writes them into --videos-dir, and deletes the
zip. Idempotent via per-part .done markers, so it resumes after interruption.

Shard by --shard/--num-shards (e.g. SLURM env) to spread the download across nodes.
"""
from __future__ import annotations
import argparse, os, re, subprocess, zipfile, collections
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "nkp37/OpenVid-1M"


def plan_parts():
    """Return {part_num: [repo_files]} for OpenVidHD (single .zip or split _part_aa/ab)."""
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    parts = collections.defaultdict(list)
    for f in files:
        if not f.startswith("OpenVidHD/"):
            continue
        m = re.search(r"OpenVidHD_part_(\d+)", f)
        if m:
            parts[int(m.group(1))].append(f)
    return {k: sorted(v) for k, v in parts.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--zip-dir", required=True, help="scratch for zips (deleted after extract unless --keep-zip)")
    ap.add_argument("--only-list", required=True, help="basenames to keep (the >=5s filtered list)")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_NTASKS", "1")))
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_PROCID", "0")))
    ap.add_argument("--parts", default="all", help="'all' or comma range like 1-14 / 1,2,3")
    a = ap.parse_args()

    vdir = Path(a.videos_dir); vdir.mkdir(parents=True, exist_ok=True)
    zdir = Path(a.zip_dir); zdir.mkdir(parents=True, exist_ok=True)
    mdir = vdir.parent / "_extracted"; mdir.mkdir(parents=True, exist_ok=True)
    only = {l.strip() for l in open(a.only_list) if l.strip()}

    parts = plan_parts()
    keys = sorted(parts)
    if a.parts != "all":
        want = set()
        for tok in a.parts.split(","):
            if "-" in tok:
                lo, hi = tok.split("-"); want |= set(range(int(lo), int(hi) + 1))
            else:
                want.add(int(tok))
        keys = [k for k in keys if k in want]
    keys = keys[a.shard::a.num_shards]
    print(f"[hd-dl shard {a.shard}/{a.num_shards}] {len(keys)} parts: {keys[:6]}{'...' if len(keys)>6 else ''}", flush=True)

    for i in keys:
        marker = mdir / f"OpenVidHD_part_{i}.done"
        if marker.exists():
            continue
        fs = parts[i]
        zip_path = zdir / f"OpenVidHD_part_{i}.zip"
        try:
            if len(fs) == 1 and fs[0].endswith(".zip"):
                p = hf_hub_download(REPO, fs[0], repo_type="dataset", local_dir=str(zdir))
                zip_path = Path(p)
            else:  # split: download _part_aa/_part_ab, concat
                locals_ = []
                for f in fs:
                    locals_.append(hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(zdir)))
                with open(zip_path, "wb") as out:
                    for lp in sorted(locals_):
                        with open(lp, "rb") as src:
                            while (chunk := src.read(1 << 24)):
                                out.write(chunk)
                for lp in locals_:
                    Path(lp).unlink(missing_ok=True)
            n = 0
            with zipfile.ZipFile(zip_path) as z:
                for m in z.namelist():
                    if m.lower().endswith(".mp4") and Path(m).name in only:
                        (vdir / Path(m).name).write_bytes(z.read(m))
                        n += 1
            marker.write_text(str(n))
            if not a.keep_zip:
                zip_path.unlink(missing_ok=True)
            print(f"[hd-dl] part_{i}: extracted {n} clips", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[hd-dl] part_{i} FAILED: {repr(e)[:150]}", flush=True)
    print(f"[hd-dl shard {a.shard}] done", flush=True)


if __name__ == "__main__":
    main()
