"""End-to-end demo: generate one LTX2 video, score it with every
applicable metric via the top-level Evaluator API.

Two phases:

1. **Generate** one LTX2-Distilled video (single GPU).  Mirrors the
   pattern in ``examples/inference/basic/basic_ltx2_distilled.py``,
   forced to single GPU for portability.

2. **Score** the same video duplicated N times into ``gen/`` and ``ref/``
   directories.  Because gen == ref by construction:

   * Per-sample paired metrics (PSNR / SSIM / LPIPS / gt_optical_flow)
     return their perfect-match values (PSNR ≈ 100, SSIM ≈ 1, LPIPS ≈ 0).
   * Set metric ``common.fvd`` returns ≈ 0.
   * VBench per-sample metrics return the same score on every duplicate
     (a stable read on the LTX2 sample).

The scoring block is the point of this script: 4 lines to build the
samples list (``samples_from``) and run the Evaluator
(``ev.evaluate``).  All input-shape ceremony goes away because each
metric reads only the keys it needs from the fat sample dict.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------
# Generation — make one LTX2 video to evaluate.
# ---------------------------------------------------------------------

PROMPT = (
    "A warm sunny backyard. The camera starts in a tight cinematic close-up "
    "of a woman and a man in their 30s, facing each other with serious "
    "expressions. The woman, emotional and dramatic, says softly, \"That's "
    "it... Dad's lost it. And we've lost Dad.\" The man exhales, slightly "
    "annoyed: \"Stop being so dramatic, Jess.\""
)

OUTPUT_PATH = "fastvideo/tests/eval/asset/ltx2.mp4"
N_DUP = 4  # how many times to duplicate the video for the gen/ref corpora


def generate_one_ltx2_video() -> str:
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
    from fastvideo import VideoGenerator

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    # Davids048/LTX2-Base-Diffusers is the audio-capable LTX-2 checkpoint
    # (the Distilled variant ships without the audio VAE, so its mp4
    # audio track is silence/noise — unusable for audio.* metrics).
    generator = VideoGenerator.from_pretrained(
        "Davids048/LTX2-Base-Diffusers",
        num_gpus=1,
    )
    generator.generate_video(
        prompt=PROMPT,
        output_path=OUTPUT_PATH,
        save_video=True,
        num_frames=121,  # ~5s @ 24 fps — long enough for audio.desync (Synchformer ≥14 segments)
        height=480,
        width=832,
        fps=24,
    )
    generator.shutdown()
    torch.cuda.empty_cache()
    return OUTPUT_PATH


# ---------------------------------------------------------------------
# Eval — the point of the script.  4 lines from "two paths" to results.
# ---------------------------------------------------------------------

def _all_registered_metrics() -> list[str]:
    """Every metric in the registry, sorted.  Combined with
    ``skip_missing_deps=True`` this is the "run everything that works in
    this venv" pattern — missing-dep / setup-failed metrics drop with a
    warning, runtime-missing inputs (audio, masks) surface as the metric
    returning a skipped MetricResult."""
    from fastvideo.eval.registry import list_metrics
    return list_metrics()


def score_all_metrics(video_path: str) -> None:
    # Duplicate one video into gen/ ref/ — identical content by design.
    tmp = Path(tempfile.mkdtemp(prefix="all_metrics_demo_"))
    gen_dir = tmp / "gen"
    ref_dir = tmp / "ref"
    gen_dir.mkdir()
    ref_dir.mkdir()
    for i in range(N_DUP):
        shutil.copyfile(video_path, gen_dir / f"clip_{i:03d}.mp4")
        shutil.copyfile(video_path, ref_dir / f"clip_{i:03d}.mp4")

    # === The whole eval, in 4 lines. ===
    from fastvideo.eval import create_evaluator, samples_from
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_init0 = time.perf_counter()
    ev = create_evaluator(metrics=_all_registered_metrics(),
                          device="cuda:0", num_gpus=1, skip_missing_deps=True)
    t_init1 = time.perf_counter()
    samples = samples_from(video=gen_dir, reference=ref_dir, text_prompt=PROMPT, fps=24.0,
                           extract_audio=True)  # auto-extract audio track from videos
    t_eval0 = time.perf_counter()
    results = ev.evaluate(samples=samples)
    torch.cuda.synchronize()
    t_eval1 = time.perf_counter()
    # ===================================

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\n=== Timing & memory ===")
    print(f"  Evaluator construction + model loads: {t_init1 - t_init0:.1f}s")
    print(f"  ev.evaluate(samples):                 {t_eval1 - t_eval0:.1f}s")
    print(f"  total eval block:                      {t_eval1 - t_init0:.1f}s")
    print(f"  peak GPU mem allocated:                {peak_gb:.2f} GB")
    print(f"  peak GPU mem reserved (caches):        {peak_reserved_gb:.2f} GB")

    print("\n=== Per-sample scores (sample 0; identical for every duplicate) ===")
    for name in ev.metric_names:
        r = results[0].get(name)
        if r is None:
            continue  # set metric, will show in corpus section
        if r.score is None:
            print(f"  {name}: SKIPPED — {r.details.get('skipped', 'no score')}")
        else:
            print(f"  {name}: {r.score:.4f}")

    print("\n=== Corpus scores (set metrics) ===")
    if not results.corpus:
        print("  (none)")
    for name, r in results.corpus.items():
        if r.score is None:
            print(f"  {name}: SKIPPED — {r.details.get('skipped', 'no score')}")
        else:
            print(f"  {name}: {r.score:.4f}  details={r.details}")


def main() -> None:
    # Reuse an existing generation if one is already on disk so iteration
    # on the eval block doesn't pay the generation cost every run.
    if not Path(OUTPUT_PATH).exists():
        print(f"[gen] No video at {OUTPUT_PATH}; generating with LTX2-Distilled (1 GPU)...")
        generate_one_ltx2_video()
    else:
        print(f"[gen] Reusing existing {OUTPUT_PATH}")

    score_all_metrics(OUTPUT_PATH)


if __name__ == "__main__":
    main()
