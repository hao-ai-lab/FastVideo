"""Generate one Wan video and score it with VBench metrics.

This example mirrors ``examples/inference/eval/basic_ltx2_eval.py`` but uses
Wan as the text-to-video generator.

Wan outputs are video-only by default, so this script evaluates video metrics
rather than audio metrics. Audio metrics should only be used if an audio track
is explicitly available or muxed into the generated mp4.

The first run downloads the VBench-related checkpoints to
``~/.cache/fastvideo/eval/``.
"""

from pathlib import Path

import torch

from fastvideo import VideoGenerator
from fastvideo.eval import Evaluator
from fastvideo.eval.io import build_eval_kwargs

PROMPT = (
    "A cinematic shot of a small dog running through a sunny park, "
    "with realistic motion, natural lighting, and a shallow depth of field. "
    "The camera follows the dog from the side as it runs across the grass."
)

# VBench sub-metrics meaningful for an arbitrary text-to-video sample.
# Structured-prompt metrics such as vbench.color, vbench.multiple_objects,
# and vbench.scene are excluded because they need prompts built to a
# specific schema.
METRICS = [
    "vbench.aesthetic_quality",
    "vbench.subject_consistency",
    "vbench.background_consistency",
    "vbench.imaging_quality",
    "vbench.temporal_flickering",
    "vbench.motion_smoothness",
    "vbench.dynamic_degree",
    "vbench.overall_consistency",
]

MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
OUTPUT_PATH = Path("outputs_video/wan_basic/output_wan_t2v_480_832.mp4")
FPS = 16.0


def _print_results(results) -> None:
    print("\n=== VBench scores ===")

    for name in METRICS:
        if name not in results:
            print(f"  {name}: MISSING")
            continue

        r = results[name]

        if r.score is None:
            reason = (
                r.details.get("skipped", "no score")
                if isinstance(r.details, dict)
                else "no score"
            )
            print(f"  {name}: SKIPPED ({reason})")
        else:
            print(f"  {name}: {r.score:.4f}")

        if r.details:
            for key, value in r.details.items():
                print(f"      {key}: {value}")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ----- generation -----
    if OUTPUT_PATH.exists():
        print(f"[eval] using existing video: {OUTPUT_PATH}")
    else:
        print("[eval] generating Wan video...")

        generator = VideoGenerator.from_pretrained(
            MODEL_PATH,
            num_gpus=1,
        )

        generator.generate_video(
            prompt=PROMPT,
            output_path=str(OUTPUT_PATH),
            save_video=True,
            num_frames=81,
            height=480,
            width=832,
        )

        generator.shutdown()

        # Free residual CUDA memory before building the evaluator.
        torch.cuda.empty_cache()

    # ----- scoring -----
    print(f"\n[eval] building evaluator: {METRICS}")
    evaluator = Evaluator(metrics=METRICS)

    sample = build_eval_kwargs(
        {"prompt": PROMPT},
        str(OUTPUT_PATH),
        fps=FPS,
    )

    print(f"[eval] running ({sample['video'].shape[1]} frames @ {FPS:g} fps)...")
    results = evaluator.evaluate(**sample)

    _print_results(results)


if __name__ == "__main__":
    main()