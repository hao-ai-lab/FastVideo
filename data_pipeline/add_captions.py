# SPDX-License-Identifier: Apache-2.0
"""Fill in real captions on a videos2caption.json built from OpenVid clips.

The OpenVid-WanTrack shards ship mp4s only -- no metadata -- so a manifest built by
scanning the clip directory has empty ``cap`` fields. Stage 5 would then encode empty
strings through T5, giving every clip an identical null text embedding: text conditioning
(and the joint text+motion CFG that depends on a meaningful conditional/null contrast)
would be silently dead.

OpenVid-1M's caption CSV keys on exactly the same filenames (``---_iRTHryQ_13_0to241.mp4``),
so the join is 1:1 on basename -- no id parsing required.

    python data_pipeline/add_captions.py --manifest <root>/videos2caption.json

The CSV is downloaded once from the Hub (~300 MB) and cached; pass --captions-csv to use
a local copy. Clips with no caption keep "" and are counted, never silently dropped.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CAPTION_REPO = "nkp37/OpenVid-1M"
CAPTION_FILES = ["data/train/OpenVid-1M.csv", "data/train/OpenVidHD.csv"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True, help="videos2caption.json to patch in place.")
    p.add_argument("--captions-csv", type=Path, nargs="*", default=None,
                   help="Local caption CSV(s). Default: download+cache from the Hub.")
    p.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/openvid_captions",
                   help="Where downloaded caption CSVs are cached.")
    p.add_argument("--min-coverage", type=float, default=0.9,
                   help="Fail if fewer than this fraction of clips get a caption (0 = never fail).")
    p.add_argument("--dry-run", action="store_true", help="Report coverage without writing.")
    return p.parse_args()


def caption_paths(args: argparse.Namespace) -> list[Path]:
    if args.captions_csv:
        return list(args.captions_csv)
    from huggingface_hub import hf_hub_download
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for f in CAPTION_FILES:
        try:
            out.append(Path(hf_hub_download(CAPTION_REPO, f, repo_type="dataset",
                                            cache_dir=str(args.cache_dir))))
        except Exception as e:  # noqa: BLE001
            print(f"[caption] warn: could not fetch {f} ({e})", flush=True)
    if not out:
        sys.exit("[caption] ERROR: no caption CSV available")
    return out


def load_captions(paths: list[Path]) -> dict[str, str]:
    """basename -> caption. Later files do not overwrite earlier hits."""
    caps: dict[str, str] = {}
    csv.field_size_limit(10 * 1024 * 1024)      # captions can be long
    for p in paths:
        n0 = len(caps)
        with p.open(newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                key, cap = row.get("video"), row.get("caption")
                if key and cap and key not in caps:
                    caps[key] = cap.strip()
        print(f"[caption] {p.name}: +{len(caps) - n0} captions (total {len(caps)})", flush=True)
    return caps


def main() -> None:
    args = parse_args()
    items = json.loads(args.manifest.read_text())
    caps = load_captions(caption_paths(args))

    hit = 0
    missing: list[str] = []
    for it in items:
        name = Path(it.get("path", "")).name
        cap = caps.get(name)
        if cap:
            it["cap"] = [cap]
            hit += 1
        else:
            it.setdefault("cap", [""])
            missing.append(name)

    cov = hit / max(len(items), 1)
    print(f"[caption] matched {hit}/{len(items)} clips ({cov*100:.1f}%)", flush=True)
    if missing:
        print(f"[caption] first few unmatched: {missing[:3]}", flush=True)

    if cov < args.min_coverage:
        sys.exit(f"[caption] ERROR: coverage {cov*100:.1f}% < required {args.min_coverage*100:.0f}%. "
                 "Refusing to write -- training on empty captions silently breaks text conditioning.")

    if args.dry_run:
        print("[caption] dry run, manifest not written", flush=True)
        return
    tmp = args.manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, indent=2))
    tmp.replace(args.manifest)
    print(f"[caption] wrote {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
