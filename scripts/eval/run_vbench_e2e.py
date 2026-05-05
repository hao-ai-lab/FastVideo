"""End-to-end VBench evaluation: prompt → generate → score.

    # Generate + score:
    python scripts/eval/run_vbench_e2e.py \\
        --model <hf-id-or-path> \\
        --dimensions subject_consistency,aesthetic_quality \\
        --videos-dir runs/vbench/videos \\
        --results-json runs/vbench/results.json

    # Generate only:
    python scripts/eval/run_vbench_e2e.py \\
        --model <hf-id-or-path> \\
        --dimensions subject_consistency \\
        --videos-dir runs/vbench/videos \\
        --skip-scoring

    # Score existing videos:
    python scripts/eval/run_vbench_e2e.py \\
        --dimensions subject_consistency \\
        --videos-dir runs/vbench/videos \\
        --results-json runs/vbench/results.json \\
        --skip-generation
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser(
        __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dimensions", default="all",
                    help="Comma-separated VBench dimensions, or 'all'.")
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--videos-dir", required=True, type=Path)
    ap.add_argument("--results-json", type=Path, default=None)
    ap.add_argument("--metrics", default="vbench")

    ap.add_argument("--model", default=None,
                    help="HF id / path for VideoGenerator.from_pretrained.")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--num-frames", type=int, default=None)
    ap.add_argument("--num-inference-steps", type=int, default=None)

    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--num-gpus", type=int, default=1)

    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--skip-scoring", action="store_true")
    args = ap.parse_args()

    from fastvideo.eval import EvalRunner
    from fastvideo.eval.datasets import VBenchPromptDataset

    dims = ("all" if args.dimensions == "all"
            else [d.strip() for d in args.dimensions.split(",") if d.strip()])
    dataset = VBenchPromptDataset(dimensions=dims)
    if args.max_prompts:
        dataset._rows = dataset._rows[:args.max_prompts]

    if args.metrics.strip() == "all":
        metrics: str | list[str] = "all"
    elif "," in args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        metrics = args.metrics.strip()

    generator = None
    if not args.skip_generation:
        if not args.model:
            ap.error("--model is required unless --skip-generation is set")
        from fastvideo.entrypoints.video_generator import VideoGenerator
        generator = VideoGenerator.from_pretrained(args.model,
                                                   num_gpus=args.num_gpus)

    gen_kwargs = {k: v for k, v in {
        "height": args.height, "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "fps": args.fps,
    }.items() if v is not None}

    runner = EvalRunner.from_dataset(
        dataset,
        videos_dir=args.videos_dir,
        generator=generator,
        metrics=metrics if not args.skip_scoring else [],
        num_gpus=args.num_gpus,
        fps=args.fps,
        seed=args.seed,
        n_samples=args.n_samples,
        gen_kwargs=gen_kwargs,
    )

    if not args.skip_generation:
        runner.generate()
    if not args.skip_scoring:
        result = runner.score()
        result.print()
        if args.results_json:
            result.save(args.results_json)


if __name__ == "__main__":
    main()
