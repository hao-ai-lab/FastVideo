# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open Small — fast / lightweight T2A example.

User story (interactive UI builder):
    "I'm building a sound-design UI where the user types a prompt and
    we want sub-2-second feedback so the experience feels like
    autocomplete, not a render queue. The full Stable Audio Open 1.0
    takes ~8s on a single GPU; the small variant takes a fraction of
    that — quality is lower but completely usable for real-time
    iteration."

User story (overnight batch jobs):
    "I'm generating 10,000 short SFX variants for a procedural game.
    Wall-clock matters more than per-clip polish — give me the small
    model so I can fit the run in one night instead of a week."

How it works:
    The small variant is a separate Stability AI checkpoint
    (`stabilityai/stable-audio-open-small`) that ships the same Oobleck
    VAE as the 1.0 base model but a smaller / faster DiT (`embed_dim=1024`,
    `depth=16`, `qk_norm="ln"`) and only one duration conditioner
    (`seconds_total`, no `seconds_start`). FastVideo loads from the
    converted Diffusers-format repo `FastVideo/stable-audio-open-small-Diffusers`
    via the standard component loader; per-variant arch fields come
    from `transformer/config.json` and `conditioner/config.json`.

Prerequisites: same as `basic_stable_audio.py`. The converted repo is
public so no gated-access flow is required.
"""
from fastvideo import VideoGenerator

PROMPT = "Lo-fi hip hop instrumental with vinyl crackle and gentle piano."


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "FastVideo/stable-audio-open-small-Diffusers",
        num_gpus=1,
    )
    output_path = "outputs_audio/stable_audio_small/output_stable_audio_small.wav"
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=True,
        # Small variant trains on a ~11.9s window — keep `audio_end_in_s`
        # at or below that.
        audio_end_in_s=6.0,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
