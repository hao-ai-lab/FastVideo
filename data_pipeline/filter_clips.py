# SPDX-License-Identifier: Apache-2.0
"""Select clips that already match the target geometry; symlink them for downstream stages.

Real shards are usually uniform but not guaranteed. Rather than re-encoding every clip to
force conformity (a lossy no-op when the clip already matches -- measured 40 dB on the
OpenVid shard, worse than the VAE round-trip's own distortion), this scans container
metadata (fast: no frame decode) and links through only the clips that conform.

Non-conforming clips are reported and listed in ``skipped_clips.json`` so nothing is
silently dropped -- re-run them through ``resize_videos.py`` if you want them included.

    python data_pipeline/filter_clips.py \\
        --src-dir <root>/raw_videos --out-dir <root>/videos \\
        --height 720 --width 1280 --num-frames 121
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True, help="Symlinks to conforming clips land here.")
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--num-frames", type=int, default=121,
                   help="Required exact frame count (0 = don't check).")
    p.add_argument("--report", type=str, default="skipped_clips.json",
                   help="Written next to --out-dir; lists every skipped clip and why.")
    p.add_argument("--needs-resize-list", type=str, default="needs_resize.txt",
                   help="Written next to --out-dir; names of clips that are readable but at the "
                        "wrong geometry, i.e. rescuable by resize_videos.py --include-list. "
                        "Unreadable clips are excluded (nothing can rescue those).")
    p.add_argument("--clean", action="store_true",
                   help="Empty --out-dir of *.mp4 first (links AND regular files -- a stale real "
                        "file would otherwise shadow the link and be used silently). --out-dir is "
                        "a derived directory; never point it at original footage.")
    return p.parse_args()


def probe(path: Path) -> tuple[int, int, int]:
    """Return (width, height, n_frames) from container metadata; (0,0,0) if unreadable."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return (0, 0, 0)
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
          int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return wh


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for old in args.out_dir.glob("*.mp4"):
            old.unlink()

    clips = sorted(args.src_dir.glob("*.mp4"))
    ok = 0
    skipped: list[dict] = []
    for c in clips:
        w, h, n = probe(c)
        if (w, h) == (0, 0):
            skipped.append({"clip": c.name, "reason": "unreadable"})
            continue
        if (w, h) != (args.width, args.height):
            skipped.append({"clip": c.name, "reason": "resolution", "got": f"{w}x{h}"})
            continue
        if args.num_frames and n != args.num_frames:
            skipped.append({"clip": c.name, "reason": "frames", "got": n})
            continue
        link = args.out_dir / c.name
        target = c.resolve()
        if link.is_symlink() and link.readlink() == target:
            pass                       # already correct
        else:
            if link.exists() or link.is_symlink():
                link.unlink()          # replace a stale file/link rather than trusting it
            link.symlink_to(target)
        ok += 1

    report_path = args.out_dir.parent / args.report
    report_path.write_text(json.dumps(
        {"src": str(args.src_dir), "required": f"{args.width}x{args.height}@{args.num_frames}f",
         "total": len(clips), "kept": ok, "skipped": skipped}, indent=2))

    rescuable = [s["clip"] for s in skipped if s["reason"] in ("resolution", "frames")]
    list_path = args.out_dir.parent / args.needs_resize_list
    list_path.write_text("\n".join(rescuable) + ("\n" if rescuable else ""))

    by_reason: dict[str, int] = {}
    for s in skipped:
        by_reason[s["reason"]] = by_reason.get(s["reason"], 0) + 1
    detail = ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items())) or "none"
    print(f"[filter] {ok}/{len(clips)} clips match {args.width}x{args.height}"
          f"@{args.num_frames}f -> {args.out_dir}", flush=True)
    print(f"[filter] skipped: {detail}  (details in {report_path})", flush=True)
    if rescuable:
        print(f"[filter] {len(rescuable)} rescuable by resize -> {list_path}", flush=True)


if __name__ == "__main__":
    main()
