"""End-to-end VBench: dataset → generate → score → aggregate.

Iterates the VBench prompt corpus, generates one video per prompt with
LTX2, scores each generated video against the requested ``vbench.*``
sub-metrics, and prints per-metric averages over the run.

Re-running with ``--skip-generation`` reuses any mp4 already on disk
under ``--videos-dir``, so you can iterate on metric selection without
re-paying the generation cost.

Example — quick smoke run on 4 prompts from the ``aesthetic_quality``
dimension across 2 GPUs::

    python examples/inference/eval/bench_vbench.py \\
        --dimensions aesthetic_quality \\
        --limit 4 --num-gpus 2 \\
        --videos-dir outputs_video/vbench_smoke

Full benchmark on a single dimension::

    python examples/inference/eval/bench_vbench.py \\
        --dimensions subject_consistency --num-gpus 8
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from fastvideo.eval import create_evaluator
from fastvideo.eval.datasets import get_dataset


def _slugify(prompt: str, max_len: int = 100) -> str:
    """Filesystem-safe filename stem; mirrors VBench's official convention."""
    s = re.sub(r'[\\/:*?"<>|]', "", prompt[:max_len]).strip().strip(".")
    return re.sub(r"\s+", " ", s) or "output"


def _generate_videos(prompts: list[str], videos_dir: Path,
                     model: str, num_gpus: int,
                     num_frames: int, height: int, width: int) -> None:
    from fastvideo import VideoGenerator

    videos_dir.mkdir(parents=True, exist_ok=True)
    todo = [(p, videos_dir / f"{_slugify(p)}.mp4") for p in prompts]
    todo = [(p, out) for (p, out) in todo if not out.is_file()]
    if not todo:
        print(f"[gen] all {len(prompts)} videos already present; skipping.")
        return

    print(f"[gen] {len(todo)}/{len(prompts)} prompts to render with {model} "
          f"({num_frames}x{height}x{width})...")
    gen = VideoGenerator.from_pretrained(model, num_gpus=num_gpus)
    try:
        for prompt, out_path in todo:
            gen.generate_video(
                prompt=prompt, output_path=str(out_path), save_video=True,
                num_frames=num_frames, height=height, width=width,
            )
    finally:
        gen.shutdown()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dimensions", default="aesthetic_quality,subject_consistency",
                   help="Comma-separated VBench dimensions (or 'all').")
    p.add_argument("--limit", type=int, default=None,
                   help="Truncate to first N prompts for smoke runs.")
    p.add_argument("--videos-dir", type=Path,
                   default=Path("outputs_video/bench_vbench"))
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--model", default="Davids048/LTX2-Base-Diffusers",
                   help="HF repo id of the text→video generator to use.")
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--height", type=int, default=1088)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--fps", type=float, default=24.0,
                   help="Frame-rate annotation passed to fps-aware metrics.")
    p.add_argument("--skip-generation", action="store_true",
                   help="Re-score existing videos under --videos-dir without "
                        "regenerating.")
    p.add_argument("--scores-out", type=Path, default=None,
                   help="Where to dump per-prompt scores as JSON. "
                        "Defaults to <videos-dir>/scores.json.")
    args = p.parse_args()

    # 1. Pull prompts from VBench.
    dims_arg: list[str] | str = (
        args.dimensions if args.dimensions == "all"
        else [d.strip() for d in args.dimensions.split(",") if d.strip()]
    )
    ds = get_dataset("vbench", dimensions=dims_arg)
    rows = list(ds)[: args.limit]
    print(f"[load] VBench: {len(rows)} prompts across {ds.dimensions}")

    # 2. Generate (or reuse) one mp4 per prompt.
    if not args.skip_generation:
        _generate_videos(
            [row["prompt"] for row in rows],
            args.videos_dir, args.model, args.num_gpus,
            args.num_frames, args.height, args.width,
        )

    # 3. Score each video against the requested vbench sub-metrics.
    metric_names = sorted(set(f"vbench.{d}" for d in ds.dimensions))
    print(f"[eval] metrics: {metric_names}")
    evaluator = create_evaluator(metrics=metric_names, num_gpus=args.num_gpus)

    samples: list[dict] = []
    matched_rows: list[dict] = []
    for row in rows:
        video_path = args.videos_dir / f"{_slugify(row['prompt'])}.mp4"
        if not video_path.is_file():
            print(f"[eval] missing {video_path}; skipping this row.")
            continue
        # Pass the path; the worker decodes lazily so memory stays bounded.
        samples.append({
            "video": str(video_path),
            "fps": args.fps,
            **row,                                       # prompt / aux / dims
        })
        matched_rows.append(row)

    all_results = evaluator.evaluate(samples=samples)
    evaluator.shutdown()

    # 4. Aggregate per-metric.
    by_metric: dict[str, list[float]] = defaultdict(list)
    detailed: list[dict] = []
    for row, results in zip(matched_rows, all_results):
        scores = {name: r.score for name, r in results.items()}
        detailed.append({"prompt": row["prompt"], "scores": scores})
        for name, score in scores.items():
            if score is not None:
                by_metric[name].append(score)

    print()
    print("=== per-metric averages ===")
    for name in sorted(by_metric):
        avg = sum(by_metric[name]) / len(by_metric[name])
        print(f"  {name:42s}  {avg:.4f}   (n={len(by_metric[name])})")

    out = args.scores_out or (args.videos_dir / "scores.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(detailed, indent=2))
    print(f"\n[done] per-prompt scores → {out}")


if __name__ == "__main__":
    main()
