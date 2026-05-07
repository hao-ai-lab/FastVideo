"""Score a folder of videos in parallel across multiple GPUs.

Uses :meth:`Evaluator.evaluate(samples=[...])`, which round-robins each
sample dict across the GPU replicas the evaluator was built with.

Example::

    python examples/inference/eval/score_folder.py \\
        --videos generated/ \\
        --metrics vbench.aesthetic_quality,vbench.subject_consistency \\
        --num-gpus 4 \\
        --output scores.json

Pair each generated video with a same-name reference video (e.g.
``ref/<stem>.mp4``) by passing ``--reference-dir``::

    python examples/inference/eval/score_folder.py \\
        --videos generated/ --reference-dir ref/ \\
        --metrics common.psnr,common.ssim,common.lpips \\
        --num-gpus 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastvideo.eval import create_evaluator


def _list_videos(directory: Path) -> list[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".gif"}
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", type=Path, required=True,
                   help="Directory of generated videos.")
    p.add_argument("--reference-dir", type=Path, default=None,
                   help="Directory of reference videos with matching stems.")
    p.add_argument("--metrics", default="vbench.aesthetic_quality")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--fps", type=float, default=None,
                   help="Frame-rate annotation for fps-aware metrics.")
    p.add_argument("--output", type=Path, default=Path("scores.json"))
    args = p.parse_args()

    video_paths = _list_videos(args.videos)
    if not video_paths:
        raise SystemExit(f"No videos under {args.videos}")
    print(f"Found {len(video_paths)} videos in {args.videos}")

    metrics: list[str] | str = (
        args.metrics if args.metrics == "all"
        else [m.strip() for m in args.metrics.split(",") if m.strip()]
    )
    evaluator = create_evaluator(metrics=metrics, num_gpus=args.num_gpus)

    # Build per-video sample dicts holding *paths*, not pre-loaded
    # tensors. Each path is decoded inside the worker thread that picks
    # up its sample, so peak resident memory is bounded by num_gpus
    # rather than scaling with the size of the folder.
    samples: list[dict] = []
    for vp in video_paths:
        sample: dict = {"video": str(vp)}
        if args.reference_dir is not None:
            ref_path = args.reference_dir / vp.name
            if not ref_path.is_file():
                raise FileNotFoundError(f"Missing reference for {vp.name} at {ref_path}")
            sample["reference"] = str(ref_path)
        if args.fps is not None:
            sample["fps"] = args.fps
        samples.append(sample)

    print(f"Scoring with {len(evaluator.metric_names)} metric(s) "
          f"on {evaluator.num_gpus} GPU(s)...")
    all_results = evaluator.evaluate(samples=samples)
    evaluator.shutdown()

    payload = [
        {
            "video": str(vp),
            "scores": {name: r.score for name, r in results.items()},
        }
        for vp, results in zip(video_paths, all_results)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
