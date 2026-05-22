"""Compute Fréchet Video Distance (FVD) over a folder of generated videos.

Replaces the standalone ``benchmarks/fvd/run_fvd.py`` + ``benchmarks/scripts/run.sh``
that lived in the repo before ``common.fvd`` became a registered eval metric.

Run::

    pip install -e .[eval]
    python examples/inference/eval/eval_fvd.py \\
        --gen-dir   path/to/generated_videos/ \\
        --reference-dir path/to/real_videos/ \\
        --extractor i3d \\
        --output    fvd_scores.json

The first call extracts I3D (or CLIP / VideoMAE) features over every
file under ``--reference-dir`` and caches them to
``${FASTVIDEO_EVAL_CACHE}/fvd/real_features_{extractor}.pt``.  Subsequent
runs with the same ``--extractor`` reuse the cache — pass
``--reference-dir`` only on the first call (or whenever the reference
set changes).

For paper-grade FVD scores use ``--extractor i3d`` (the literature
default) and at least 256 generated + 256 reference videos.  CLIP and
VideoMAE extractors are research-grade and not directly comparable to
published FVD numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fastvideo.eval import get_metric
from fastvideo.eval.io import load_video


_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".gif"}


def _list_videos(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"{directory} is not a directory")
    out = sorted(p for p in directory.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
    if not out:
        raise SystemExit(f"No videos under {directory} (looked for {sorted(_VIDEO_EXTS)})")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen-dir", type=Path, required=True,
                   help="Directory of generated videos (.mp4, .avi, .mov, .mkv, .gif).")
    p.add_argument("--reference-dir", type=Path, default=None,
                   help="Directory of reference videos. Required on first run for a given "
                        "extractor; subsequent runs reuse the cached features.")
    p.add_argument("--extractor", choices=["i3d", "clip", "videomae"], default="i3d",
                   help="Feature backbone (default: i3d, the standard FVD spec).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--chunk-size", type=int, default=32,
                   help="Videos per forward pass. Reduce if GPU OOMs.")
    p.add_argument("--cache-path", type=Path, default=None,
                   help="Override the reference-feature cache path. "
                        "Defaults to ${FASTVIDEO_EVAL_CACHE}/fvd/real_features_{extractor}.pt.")
    p.add_argument("--output", type=Path, default=None,
                   help="Write the result as JSON to this path (default: stdout only).")
    args = p.parse_args()

    metric = get_metric(
        "common.fvd",
        extractor=args.extractor,
        cache_path=str(args.cache_path) if args.cache_path else None,
        chunk_size=args.chunk_size,
    )
    metric.to(args.device)
    metric.setup()
    metric.reset()

    gen_paths = _list_videos(args.gen_dir)
    print(f"Found {len(gen_paths)} generated videos under {args.gen_dir}")

    # Build the reference cache on the first generated sample, then never
    # touch it again — common.fvd takes one reference *set* and reuses it
    # across all subsequent accumulate() calls.
    ref_tensor: torch.Tensor | None = None
    if args.reference_dir is not None:
        ref_paths = _list_videos(args.reference_dir)
        print(f"Found {len(ref_paths)} reference videos under {args.reference_dir}")
        ref_tensor = torch.stack([load_video(str(p)) for p in ref_paths])

    for i, gp in enumerate(gen_paths):
        sample: dict = {"video": load_video(str(gp))}
        if i == 0 and ref_tensor is not None:
            sample["reference"] = ref_tensor
        metric.accumulate(sample)
        if (i + 1) % 32 == 0 or i == len(gen_paths) - 1:
            print(f"  accumulated {i + 1}/{len(gen_paths)} generated videos")

    result = metric.finalize()

    if result.score is None:
        print(f"\nFVD ({args.extractor}): SKIPPED — {result.details.get('skipped')}")
    else:
        print(f"\nFVD ({args.extractor}): {result.score:.4f}")
        print(f"  details: {result.details}")

    if args.output is not None:
        payload = {
            "metric": result.name,
            "score": result.score,
            "details": result.details,
            "extractor": args.extractor,
            "gen_dir": str(args.gen_dir),
            "reference_dir": str(args.reference_dir) if args.reference_dir else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
