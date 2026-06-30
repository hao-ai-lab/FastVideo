"""Generate one LTX2 video and score its audio track.

LTX2 produces video + audio and muxes both into the output mp4. The
eval suite reads the audio track straight out of that mp4 (librosa
decodes via ffmpeg under the hood), so we just point the audio
metrics at the same path.

Only the reference-free audio metrics run here — they're the ones
that make sense for a single one-shot generation:

  * ``audio.clap_score``         — CLAP text↔audio cosine similarity
  * ``audio.audiobox_aesthetics`` — AudioBox 4-axis quality score

The other audio metrics need extra inputs we don't have for a
one-shot run: ``audio.frechet_distance`` and ``audio.kl_divergence``
need a reference audio, ``audio.wer`` needs a ground-truth transcript.

Install: ``uv pip install -e .[eval-audio]`` covers both metrics here
(and the rest of the audio suite).
"""
import torch

from fastvideo import VideoGenerator
from fastvideo.eval import create_evaluator
from pathlib import Path

PROMPT = (
    "A warm sunny backyard. The camera starts in a tight cinematic close-up "
    "of a woman and a man in their 30s, facing each other with serious "
    "expressions. The woman, emotional and dramatic, says softly, \"That's "
    "it... Dad's lost it. And we've lost Dad.\" The man exhales, slightly "
    "annoyed: \"Stop being so dramatic, Jess.\" A beat. He glances aside, "
    "then mutters defensively, \"He's just having fun.\""
)

METRICS = [
    "audio.clap_score",
    "audio.audiobox_aesthetics",
]


def main() -> None:
    output_path = "outputs_video/ltx2_audio_eval/output.mp4"

    if Path(output_path).exists():
        print(f"[eval] using existing video: {output_path}")
    else:
        print("[eval] generating LTX2 video...")

        generator = VideoGenerator.from_pretrained(
            "Davids048/LTX2-Base-Diffusers",
            num_gpus=1,
        )

        generator.generate_video(
            prompt=PROMPT,
            output_path=output_path,
            save_video=True,
            num_frames=121,
            height=1088,
            width=1920,
        )

        generator.shutdown()
        torch.cuda.empty_cache()

    print(f"\n[eval] building evaluator: {METRICS}")
    evaluator = create_evaluator(metrics=METRICS)
    results = evaluator.evaluate(audio=output_path, text_prompt=PROMPT)

    print("\n=== Audio scores ===")
    for name in METRICS:
        r = None

        # per-sample metric
        if hasattr(results, "__contains__") and name in results:
            r = results[name]

        # corpus-level metric (e.g. audio.frechet_distance)
        elif hasattr(results, "corpus") and name in results.corpus:
            r = results.corpus[name]

        if r is None:
            print(f"  {name}: MISSING")
            continue

        if r.score is None:
            skipped = (
                r.details.get("skipped", "no score")
                if isinstance(r.details, dict)
                else "no score"
            )
            print(f"  {name}: SKIPPED ({skipped})")
        else:
            print(f"  {name}: {r.score:.4f}")

        if r.details:
            for k, v in r.details.items():
                print(f"      {k}: {v}")


if __name__ == "__main__":
    main()
