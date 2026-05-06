"""Score every video in a folder with VBench metrics (no prompts).

Multi-GPU bulk eval over a flat folder of mp4s. The default metric
set is the prompt-free VBench subset — anything that needs a source
prompt or aux info is excluded by default.

    python scripts/eval/score_folder.py \\
        --videos-dir runs/my_outputs \\
        --num-gpus 4 \\
        --fps 24 \\
        --results-json runs/my_outputs/scores.json

Pass ``--metrics`` to override the default set; pass a custom JSON
mapping ``filename → prompt`` via ``--prompts-json`` to enable
prompt-aware metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("score-folder")

# Metrics that need only the generated frames (and optionally fps).
DEFAULT_METRICS = [
    "vbench.aesthetic_quality",
    "vbench.subject_consistency",
    "vbench.background_consistency",
    "vbench.imaging_quality",
    "vbench.temporal_flickering",
    "vbench.motion_smoothness",
    "vbench.dynamic_degree",
]

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", required=True, type=Path)
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                    help="Comma-separated metric names, or 'all', or a "
                         "group prefix like 'vbench'.")
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--prompts-json", type=Path, default=None,
                    help="Optional path to a JSON {filename: prompt} mapping. "
                         "If given, prompts attach to each sample as "
                         "text_prompt (enables prompt-aware metrics).")
    ap.add_argument("--results-json", type=Path, default=None)
    return ap.parse_args()


def parse_metrics(spec: str) -> str | list[str]:
    spec = spec.strip()
    if spec == "all":
        return "all"
    if "," in spec:
        return [m.strip() for m in spec.split(",") if m.strip()]
    return spec


def main() -> None:
    args = parse_args()
    from fastvideo.eval import Evaluator
    from fastvideo.eval.io import build_eval_kwargs
    from fastvideo.eval.types import EvalResult

    videos_dir: Path = args.videos_dir
    if not videos_dir.is_dir():
        raise SystemExit(f"--videos-dir {videos_dir} is not a directory")

    paths = sorted(p for p in videos_dir.iterdir()
                   if p.suffix.lower() in VIDEO_EXTS)
    if not paths:
        raise SystemExit(f"no videos found in {videos_dir} "
                         f"(looked for {sorted(VIDEO_EXTS)})")
    log.info("[score] %d videos found in %s", len(paths), videos_dir)

    prompts: dict[str, str] = {}
    if args.prompts_json:
        prompts = json.loads(args.prompts_json.read_text())
        log.info("[score] loaded %d prompts from %s",
                 len(prompts), args.prompts_json)

    samples: list[dict] = []
    metas: list[dict] = []
    for vp in paths:
        row: dict = {}
        if vp.name in prompts:
            row["prompt"] = prompts[vp.name]
        samples.append(build_eval_kwargs(row, vp, fps=args.fps))
        metas.append({
            "video": str(vp),
            "prompt": row.get("prompt"),
        })

    log.info("[score] building evaluator: %s (num_gpus=%d)",
             args.metrics, args.num_gpus)
    evaluator = Evaluator(metrics=parse_metrics(args.metrics),
                          num_gpus=args.num_gpus)
    log.info("[score] dispatching %d samples...", len(samples))
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

    result = EvalResult.from_raw(by_metric, per_video)
    result.print()
    if args.results_json:
        result.save(args.results_json)
        log.info("[score] wrote %s", args.results_json)


if __name__ == "__main__":
    main()
