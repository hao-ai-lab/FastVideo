"""Compute Fréchet Video Distance (FVD) over a folder of generated videos.

Run::

    pip install -e .[eval]
    python examples/inference/eval/eval_fvd.py \\
        --gen-dir   path/to/generated_videos/ \\
        --reference-dir path/to/real_videos/ \\
        --output    fvd_scores.json

Demonstrates the canonical pattern: turn two directories into a samples
list with :func:`samples_from`, hand it to :class:`Evaluator`.  The
Evaluator decodes through its :class:`VideoPool` and runs FVD's
``accumulate`` / ``finalize`` end-to-end.

The first call extracts I3D features over every file under
``--reference-dir`` and caches them to
``${FASTVIDEO_EVAL_CACHE}/fvd/real_features_i3d.pt``.  Subsequent runs
reuse the cache — omit ``--reference-dir`` to score new generations
against the cached reference set.

For paper-grade FVD scores use at least 256 generated + 256 reference
videos.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from fastvideo.eval import create_evaluator, samples_from


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen-dir", type=Path, required=True,
                   help="Directory of generated videos (.mp4, .avi, .mov, .mkv, .gif).")
    p.add_argument("--reference-dir", type=Path, default=None,
                   help="Directory of reference videos. Omit to score against the cached "
                        "reference features (built on a previous run).")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="Number of GPU replicas. >1 fans extraction out across devices.")
    p.add_argument("--cache-path", type=Path, default=None,
                   help="Override the reference-feature cache path. "
                        "Defaults to ${FASTVIDEO_EVAL_CACHE}/fvd/real_features_i3d.pt.")
    p.add_argument("--output", type=Path, default=None,
                   help="Write the result as JSON to this path (default: stdout only).")
    args = p.parse_args()

    if args.cache_path is not None:
        # FVDMetric resolves cache_path via this env-var when no constructor
        # kwarg is set; create_evaluator doesn't forward kwargs through to
        # the metric, so the env-var is the only path here.
        os.environ["FASTVIDEO_FVD_REF_FEATURES"] = str(args.cache_path)

    samples = samples_from(
        video=args.gen_dir,
        reference=args.reference_dir,
    )

    ev = create_evaluator(metrics=["common.fvd"], device=args.device, num_gpus=args.num_gpus)
    results = ev.evaluate(samples=samples)
    result = results.corpus["common.fvd"]

    if result.score is None:
        print(f"\nFVD: SKIPPED — {result.details.get('skipped')}")
    else:
        print(f"\nFVD: {result.score:.4f}")
        print(f"  details: {result.details}")

    if args.output is not None:
        payload = {
            "metric": result.name,
            "score": result.score,
            "details": result.details,
            "gen_dir": str(args.gen_dir),
            "reference_dir": str(args.reference_dir) if args.reference_dir else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
