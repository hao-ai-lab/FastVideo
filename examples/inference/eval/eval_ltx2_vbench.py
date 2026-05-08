"""End-to-end: generate a video with LTX2 and score it with VBench.

Pipeline:
    prompt → LTX2-Base → mp4 → fastvideo.eval → vbench scores

Run::

    pip install -e .[eval]
    git submodule update --init fastvideo/third_party/eval/vbench

    python examples/inference/eval/eval_ltx2_vbench.py
    # or with 4 GPUs and the distilled checkpoint:
    python examples/inference/eval/eval_ltx2_vbench.py \
        --model FastVideo/LTX2-Distilled-Diffusers --num-gpus 4

The default metric set covers the vbench sub-metrics that are
meaningful for an arbitrary text→video sample — i.e. those that need
only the generated video (and optionally fps + the source prompt).
Structured-prompt metrics like ``vbench.color``, ``vbench.scene``,
``vbench.multiple_objects`` etc. are *not* on by default — they only
make sense when the prompt is built to a specific schema, and they
require GRiT/detectron2 setup. Pass them via ``--metrics`` if you have
a matching prompt.

First-time runs download CLIP, DINO, RAFT, AMT, ViCLIP, and MUSIQ
weights to ``~/.cache/fastvideo/eval/models/`` and
``~/.cache/torch/hub/`` (~few GB total). Subsequent runs are fast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.eval import create_evaluator
from fastvideo.eval.io import load_video


PROMPT = (
    "A warm sunny backyard. The camera starts in a tight cinematic close-up "
    "of a woman and a man in their 30s, facing each other with serious "
    "expressions. The woman, emotional and dramatic, says softly, \"That's "
    "it... Dad's lost it. And we've lost Dad.\" The man exhales, slightly "
    "annoyed: \"Stop being so dramatic, Jess.\" A beat. He glances aside, "
    "then mutters defensively, \"He's just having fun.\" The camera slowly "
    "pans right, revealing the grandfather in the garden wearing enormous "
    "butterfly wings, waving his arms in the air like he's trying to take "
    "off. He shouts, \"Wheeeew!\" as he flaps his wings with full commitment. "
    "The woman covers her face, on the verge of tears. The tone is deadpan, "
    "absurd, and quietly tragic."
)

DEFAULT_METRICS = [
    # No-input metrics: just need the generated frames.
    "vbench.aesthetic_quality",       # CLIP + LAION aesthetic head
    "vbench.subject_consistency",     # DINO frame-to-first cosine
    "vbench.background_consistency",  # DINO on background patches
    "vbench.imaging_quality",         # pyiqa MUSIQ
    "vbench.temporal_flickering",     # pixel-wise frame deltas
    "vbench.motion_smoothness",       # AMT frame interpolator residual
    # Need fps annotation:
    "vbench.dynamic_degree",          # RAFT optical-flow magnitude
    # Need the source prompt:
    "vbench.overall_consistency",     # ViCLIP video↔prompt similarity
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Davids048/LTX2-Base-Diffusers",
                   help="HF repo id of the LTX2 checkpoint.")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--output", default="outputs_video/ltx2_eval/clip.mp4",
                   help="Where to save the generated mp4.")
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--height", type=int, default=1088)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--fps", type=float, default=24.0,
                   help="Frame-rate annotation passed to fps-aware metrics "
                        "(e.g. vbench.dynamic_degree). LTX2 outputs at 24 fps "
                        "by default.")
    p.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                   help="Comma-separated metric names. Pass 'all' for every "
                        "registered metric, or e.g. 'vbench' for the whole group.")
    p.add_argument("--scores-out", default="outputs_video/ltx2_eval/scores.json")
    p.add_argument("--skip-generation", action="store_true",
                   help="Reuse an existing --output video instead of regenerating.")
    return p.parse_args()


def generate(args: argparse.Namespace) -> Path:
    out = Path(args.output)
    if args.skip_generation and out.is_file():
        print(f"[gen] reusing existing video at {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[gen] loading {args.model} ({args.num_gpus} GPU)...")
    generator = VideoGenerator.from_pretrained(args.model, num_gpus=args.num_gpus)
    try:
        print(f"[gen] generating to {out}...")
        generator.generate_video(
            prompt=args.prompt,
            output_path=str(out),
            save_video=True,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
        )
    finally:
        generator.shutdown()
    return out


def evaluate_video(video_path: Path, prompt: str, fps: float,
                   metric_names) -> dict:
    print(f"[eval] loading video from {video_path}...")
    video = load_video(str(video_path))                  # (T, C, H, W) in [0, 1]
    video = video.unsqueeze(0)                            # → (1, T, C, H, W)

    print(f"[eval] building evaluator: {metric_names}")
    evaluator = create_evaluator(metrics=metric_names, device="cuda")

    print(f"[eval] running ({video.shape[1]} frames @ {fps} fps)...")
    results = evaluator.evaluate(
        video=video,
        text_prompt=[prompt],
        fps=fps,
    )

    if isinstance(results, list):
        results = results[0]                              # batch of 1

    return {
        name: {"score": r.score, "details": r.details}
        for name, r in results.items()
    }


def main() -> None:
    args = parse_args()
    if args.metrics.strip() == "all":
        metric_names = "all"
    else:
        metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

    video_path = generate(args)
    scores = evaluate_video(video_path, args.prompt, args.fps, metric_names)

    print("\n=== VBench scores ===")
    for name, payload in scores.items():
        print(f"  {name}: {payload['score']}")

    out = Path(args.scores_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"video": str(video_path), "prompt": args.prompt, "scores": scores},
        indent=2,
    ))
    print(f"[done] scores written to {out}")


if __name__ == "__main__":
    main()
