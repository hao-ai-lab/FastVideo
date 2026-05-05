"""End-to-end VBench evaluation: prompt → generate → score.

Plain script. No runner class — just :class:`fastvideo.VideoGenerator`,
:class:`fastvideo.eval.Evaluator`, and a :class:`PromptDataset` glued by
two for-loops.

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
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vbench-e2e")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dimensions", default="all")
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--videos-dir", required=True, type=Path)
    ap.add_argument("--results-json", type=Path, default=None)
    ap.add_argument("--metrics", default="vbench")

    ap.add_argument("--model", default=None)
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Override per-prompt n_samples (default: per-row).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--num-frames", type=int, default=None)
    ap.add_argument("--num-inference-steps", type=int, default=None)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--num-gpus", type=int, default=1)

    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--skip-scoring", action="store_true")
    return ap.parse_args()


def parse_metrics(spec: str) -> str | list[str]:
    spec = spec.strip()
    if spec == "all":
        return "all"
    if "," in spec:
        return [m.strip() for m in spec.split(",") if m.strip()]
    return spec


def generate(dataset, generator, videos_dir: Path, *, base_seed: int,
             n_override: int | None, gen_kwargs: dict) -> None:
    from fastvideo.eval.io import default_filename

    videos_dir.mkdir(parents=True, exist_ok=True)
    total = sum((n_override or row.get("n_samples", 1)) for row in dataset)
    log.info("[gen] %d prompts -> %d videos in %s",
             len(dataset), total, videos_dir)

    global_idx = 0
    manifest: dict[str, list[str]] = {}
    for row in dataset:
        n = n_override or row.get("n_samples", 1)
        paths: list[str] = []
        for k in range(n):
            target = videos_dir / default_filename(row, k)
            paths.append(str(target))
            seed = base_seed + global_idx
            global_idx += 1
            if target.exists():
                log.info("[gen %d/%d] SKIP exists: %s",
                         global_idx, total, target.name)
                continue
            log.info("[gen %d/%d] prompt=%r seed=%d -> %s",
                     global_idx, total, row["prompt"][:60], seed,
                     target.name)
            generator.generate_video(
                prompt=row["prompt"],
                output_path=str(target),
                save_video=True,
                seed=seed,
                **gen_kwargs,
            )
            if target.exists():
                log.info("[gen %d/%d] OK %.1f MB",
                         global_idx, total, target.stat().st_size / 1e6)
            else:
                log.warning("[gen %d/%d] missing after generate: %s",
                            global_idx, total, target)
        manifest[row["prompt"]] = paths
    (videos_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def score(dataset, evaluator, videos_dir: Path, *, fps: float):
    from fastvideo.eval.io import build_eval_kwargs, glob_videos
    from fastvideo.eval.types import EvalResult

    samples: list[dict] = []
    metas: list[dict] = []
    for row in dataset:
        for vp in glob_videos(videos_dir, row):
            samples.append(build_eval_kwargs(row, vp, fps=fps))
            metas.append({
                "prompt": row.get("prompt"),
                "video": str(vp),
                "dimensions": list(row.get("dimensions", [])),
            })
    if not samples:
        log.warning("[score] no videos matched in %s", videos_dir)
        return EvalResult.from_raw({}, [])

    results = evaluator.evaluate(samples)

    by_metric: dict[str, list[float]] = {}
    per_video: list[dict] = []
    for meta, res in zip(metas, results, strict=True):
        scores: dict[str, float] = {}
        for name, mr in res.items():
            if mr is not None and mr.score is not None:
                scores[name] = float(mr.score)
                by_metric.setdefault(name, []).append(scores[name])
        per_video.append({**meta, "scores": scores})
        log.info("[score] %s -> %s", meta["video"],
                 {k: round(v, 4) for k, v in scores.items()})
    log.info("[score] done. videos_scored=%d", len(per_video))
    return EvalResult.from_raw(by_metric, per_video)


def main() -> None:
    args = parse_args()
    from fastvideo.eval.datasets import VBenchPromptDataset

    dims = ("all" if args.dimensions == "all"
            else [d.strip() for d in args.dimensions.split(",") if d.strip()])
    dataset = VBenchPromptDataset(dimensions=dims)
    log.info("dataset: %d prompts across %d dimensions",
             len(dataset), len(dataset.dimensions))
    if args.max_prompts:
        dataset._rows = dataset._rows[:args.max_prompts]
        log.info("dataset: truncated to first %d prompts", len(dataset))

    if not args.skip_generation:
        if not args.model:
            raise SystemExit("--model is required unless --skip-generation is set")
        from fastvideo.entrypoints.video_generator import VideoGenerator
        log.info("[gen] loading %s with num_gpus=%d",
                 args.model, args.num_gpus)
        generator = VideoGenerator.from_pretrained(args.model,
                                                   num_gpus=args.num_gpus)
        gen_kwargs = {k: v for k, v in {
            "height": args.height, "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "fps": args.fps,
        }.items() if v is not None}
        generate(dataset, generator, args.videos_dir,
                 base_seed=args.seed,
                 n_override=args.n_samples,
                 gen_kwargs=gen_kwargs)

    if not args.skip_scoring:
        from fastvideo.eval import Evaluator
        evaluator = Evaluator(metrics=parse_metrics(args.metrics),
                              num_gpus=args.num_gpus)
        result = score(dataset, evaluator, args.videos_dir, fps=args.fps)
        result.print()
        if args.results_json:
            result.save(args.results_json)


if __name__ == "__main__":
    main()
