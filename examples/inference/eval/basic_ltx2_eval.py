"""Generate one LTX2 video and score it with VBench metrics.

The generation block is the same as
``examples/inference/basic/basic_ltx2.py`` — same prompt, same model,
same shape, same num_frames. After ``shutdown()`` the script loads the
mp4 back, builds a single :class:`fastvideo.eval.Evaluator`, and runs
the prompt-aware VBench subset that's meaningful for an arbitrary
text→video sample.

The first run downloads CLIP / DINO / RAFT / AMT / ViCLIP / MUSIQ
weights to ``~/.cache/fastvideo/eval/`` (~few GB total).

GPU memory caveat
-----------------
Scoring 1088×1920×121 with all 8 metrics needs a dedicated GPU (~80 GB).
On a shared GPU, ``vbench.motion_smoothness`` (AMT correlation volume)
will OOM — its memory autoscale reads ``total_memory`` rather than
``mem_get_info()`` free memory and therefore underestimates the
required scale-down. Drop ``motion_smoothness`` from ``METRICS`` if
sharing, or run on a smaller-resolution generation.
"""
import torch

from fastvideo import VideoGenerator
from fastvideo.eval import Evaluator
from fastvideo.eval.io import build_eval_kwargs

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

# VBench sub-metrics meaningful for an arbitrary text→video sample
# (just the generated frames, optionally fps + the source prompt).
# Structured-prompt metrics (vbench.color, vbench.multiple_objects,
# vbench.scene, ...) are excluded — they need prompts built to a
# specific schema.
METRICS = [
    "vbench.aesthetic_quality",       # CLIP + LAION aesthetic head
    "vbench.subject_consistency",     # DINO frame-to-first cosine
    "vbench.background_consistency",  # DINO on background patches
    "vbench.imaging_quality",         # pyiqa MUSIQ
    "vbench.temporal_flickering",     # pixel-wise frame deltas
    "vbench.motion_smoothness",       # AMT frame interpolator residual
    "vbench.dynamic_degree",          # RAFT optical-flow magnitude (needs fps)
    "vbench.overall_consistency",     # ViCLIP video↔prompt similarity
]


def main() -> None:
    # ----- generation (matches examples/inference/basic/basic_ltx2.py) -----
    generator = VideoGenerator.from_pretrained(
        "Davids048/LTX2-Base-Diffusers",
        num_gpus=1,
    )

    output_path = "outputs_video/ltx2_basic/output_ltx2_base_t2v_1088_1920_1.1.mp4"
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=True,
        num_frames=121,
        height=1088,
        width=1920,
    )
    generator.shutdown()
    # Free residual CUDA memory the generator left behind so the
    # evaluator can grab the largest possible workspace for AMT/RAFT.
    torch.cuda.empty_cache()

    # ----- scoring -----
    print(f"\n[eval] building evaluator: {METRICS}")
    evaluator = Evaluator(metrics=METRICS)

    # LTX2 outputs at 24 fps by default.
    sample = build_eval_kwargs({"prompt": PROMPT}, output_path, fps=24.0)
    print(f"[eval] running ({sample['video'].shape[1]} frames @ 24 fps)...")
    results = evaluator.evaluate(**sample)

    print("\n=== VBench scores ===")
    for name in METRICS:
        r = results[name]
        if r.score is None:
            reason = r.details.get("skipped", "no score")
            print(f"  {name}: SKIPPED ({reason})")
        else:
            print(f"  {name}: {r.score:.4f}")


if __name__ == "__main__":
    main()
