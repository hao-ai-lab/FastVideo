"""Score one video on one GPU.

Smallest possible use of ``fastvideo.eval``: load an mp4, build an
:class:`Evaluator` for the requested metric set, run it.

Examples::

    # Reference-free (just the generated video):
    python examples/inference/eval/score_video.py \\
        --video clip.mp4 \\
        --metrics vbench.aesthetic_quality,vbench.imaging_quality

    # Reference-paired (compare against ground truth):
    python examples/inference/eval/score_video.py \\
        --video gen.mp4 --reference ref.mp4 \\
        --metrics common.psnr,common.ssim,common.lpips
"""
from __future__ import annotations

import argparse
import json

from fastvideo.eval import create_evaluator
from fastvideo.eval.io import load_video


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="Path to the generated mp4.")
    p.add_argument("--reference", default=None,
                   help="Optional path to a reference mp4 (for paired metrics).")
    p.add_argument("--metrics", default="common.psnr,common.ssim",
                   help="Comma-separated metric names, or a group name like 'vbench'.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--text-prompt", default=None,
                   help="Text prompt for prompt-aware metrics "
                        "(vbench.overall_consistency, etc.).")
    p.add_argument("--fps", type=float, default=None,
                   help="Frame-rate annotation for fps-aware metrics "
                        "(vbench.dynamic_degree, etc.).")
    args = p.parse_args()

    metrics: list[str] | str = (
        args.metrics if args.metrics in ("all",)
        else [m.strip() for m in args.metrics.split(",") if m.strip()]
    )
    evaluator = create_evaluator(metrics=metrics, device=args.device)

    sample: dict = {"video": load_video(args.video)}
    if args.reference is not None:
        sample["reference"] = load_video(args.reference)
    if args.text_prompt is not None:
        sample["text_prompt"] = args.text_prompt
    if args.fps is not None:
        sample["fps"] = args.fps

    results = evaluator.evaluate(**sample)
    evaluator.shutdown()

    print(json.dumps(
        {name: r.score for name, r in results.items()},
        indent=2,
    ))


if __name__ == "__main__":
    main()
