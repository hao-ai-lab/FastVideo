# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open Small — fast / lightweight T2A example.

User story (interactive UI builder):
    "I'm building a music-prompt UI where the user types a description
    and we want sub-2-second feedback so the experience feels like
    autocomplete, not a render queue. The full Stable Audio Open 1.0
    takes ~12s on a single GPU; the small variant takes a fraction of
    that — quality is lower but completely usable for real-time
    iteration."

User story (mobile / edge deployment):
    "I want to ship a sound-design tool that runs on a single L4 or
    even a beefy CPU. The full model is too heavy for that envelope.
    Same UX, smaller envelope."

User story (overnight batch jobs):
    "I'm generating 10,000 short SFX variants for a procedural game.
    Wall-clock matters more than per-clip polish — give me the small
    model so I can fit the run in one night instead of a week."

How it works:
    The small variant is a separate Stability AI checkpoint
    (`stabilityai/stable-audio-open-small`) that ships the same
    Oobleck VAE as the 1.0 base model but a smaller / faster DiT.
    FastVideo's `StableAudioPipeline` auto-detects from the repo id
    and loads the appropriate weights via the same
    `from_official_state_dict` path.

Status (2026-04-26):
    The repo is gated; request access on
    https://huggingface.co/stabilityai/stable-audio-open-small first.
    If the architecture differs from the base model in dimensions, the
    DiT loader will surface a state-dict mismatch — file an issue with
    the diff and we'll pin sizing parameters per checkpoint. The
    sampling preset (`stable_audio_open_small`) is registered and
    ready.

Prerequisites: same as `basic_stable_audio.py`, plus HF gated access
to `stabilityai/stable-audio-open-small`.
"""
from fastvideo import VideoGenerator

PROMPT = "Lo-fi hip hop instrumental with vinyl crackle and gentle piano."


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "stabilityai/stable-audio-open-small",
        num_gpus=1,
    )
    output_path = "outputs_audio/stable_audio_small/output_stable_audio_small.mp4"
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=True,
        audio_end_in_s=6.0,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
