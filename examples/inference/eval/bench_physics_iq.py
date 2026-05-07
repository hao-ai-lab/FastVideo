"""End-to-end Physics-IQ: dataset → generate → score → aggregate.

Walks the on-disk Physics-IQ release, generates one video per take-1
scenario with LTX2 (using the scenario caption as the prompt), scores
each generated video against the take-1 reference and the take-2
"physical-variance" reference, and prints aggregate scores using
:meth:`PhysicsIQMetric.aggregate_components` — the official scoring
recipe from the upstream benchmark.

The Physics-IQ release ships at 30 FPS; the dataset transcodes once
into ``<dataset_root>/.physics_iq_cache/`` on first access.

Quick smoke run on 4 scenarios across 2 GPUs::

    python examples/inference/eval/bench_physics_iq.py \\
        --dataset-root /path/to/physics-IQ-benchmark \\
        --limit 4 --num-gpus 2 \\
        --videos-dir outputs_video/physics_iq_smoke

Re-score existing generations without regenerating::

    python examples/inference/eval/bench_physics_iq.py \\
        --dataset-root /path/to/physics-IQ-benchmark \\
        --videos-dir outputs_video/physics_iq_smoke \\
        --skip-generation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastvideo.eval import create_evaluator, get_metric
from fastvideo.eval.datasets import get_dataset


def _expected_filename(row: dict) -> str:
    """Filename Physics-IQ expects for the generated video for *row*.

    Uses the dataset's own ``expected_gen_filename`` annotation so the
    output filenames match the benchmark's manifest convention.
    """
    return row["auxiliary_info"]["expected_gen_filename"]


def _generate_videos(rows: list[dict], videos_dir: Path,
                     model: str, num_gpus: int) -> None:
    from fastvideo import VideoGenerator

    videos_dir.mkdir(parents=True, exist_ok=True)
    todo = [(row, videos_dir / _expected_filename(row)) for row in rows]
    todo = [(row, out) for (row, out) in todo if not out.is_file()]
    if not todo:
        print(f"[gen] all {len(rows)} videos already present; skipping.")
        return

    print(f"[gen] {len(todo)}/{len(rows)} scenarios to render with {model}...")
    gen = VideoGenerator.from_pretrained(model, num_gpus=num_gpus)
    try:
        for row, out_path in todo:
            gen.generate_video(
                prompt=row["prompt"], output_path=str(out_path), save_video=True,
            )
    finally:
        gen.shutdown()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Path to the Physics-IQ release (the directory "
                        "containing descriptions/ and split-videos/).")
    p.add_argument("--videos-dir", type=Path,
                   default=Path("outputs_video/bench_physics_iq"),
                   help="Where to read/write generated videos.")
    p.add_argument("--limit", type=int, default=None,
                   help="Truncate to first N scenarios for smoke runs.")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--model", default="Davids048/LTX2-Base-Diffusers",
                   help="HF repo id of the text→video generator to use.")
    p.add_argument("--skip-generation", action="store_true",
                   help="Re-score existing videos under --videos-dir.")
    p.add_argument("--scores-out", type=Path, default=None,
                   help="Where to write per-scenario scores (JSON). "
                        "Defaults to <videos-dir>/scores.json.")
    args = p.parse_args()

    # 1. Walk the Physics-IQ corpus.
    ds = get_dataset("physics_iq", dataset_root=args.dataset_root)
    rows = list(ds)[: args.limit]
    print(f"[load] Physics-IQ: {len(rows)} scenarios from {ds.dataset_dir}")

    # 2. Generate (or reuse) one mp4 per scenario.
    if not args.skip_generation:
        _generate_videos(rows, args.videos_dir, args.model, args.num_gpus)

    # 3. Score each scenario. The metric reads file paths directly out
    #    of the row dict (reference, reference_take2, masks), so we
    #    just attach the generated video path and forward.
    evaluator = create_evaluator(metrics=["physics_iq"], num_gpus=args.num_gpus)

    samples: list[dict] = []
    matched: list[dict] = []
    for row in rows:
        video_path = args.videos_dir / _expected_filename(row)
        if not video_path.is_file():
            print(f"[eval] missing {video_path}; skipping.")
            continue
        # The physics_iq metric accepts file paths via its polymorphic
        # input handling — no need to load the tensors here.
        samples.append({"video": str(video_path), **row})
        matched.append(row)

    all_results = evaluator.evaluate(samples=samples)
    evaluator.shutdown()

    # 4. Aggregate per the upstream scoring recipe.
    metric = get_metric("physics_iq")
    components = metric.aggregate_components(
        [r["physics_iq"] for r in all_results]
    )

    print()
    print("=== Physics-IQ aggregate ===")
    for name, value in components.items():
        print(f"  {name:24s}  {value:.4f}")

    detailed = [
        {
            "scenario": row["auxiliary_info"]["scenario_id"],
            "view": row["view"],
            "scenario_name": row["auxiliary_info"]["scenario_name"],
            "score": results["physics_iq"].score,
        }
        for row, results in zip(matched, all_results)
    ]
    out = args.scores_out or (args.videos_dir / "scores.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"aggregate": components, "per_scenario": detailed},
        indent=2,
    ))
    print(f"\n[done] per-scenario scores → {out}")


if __name__ == "__main__":
    main()
