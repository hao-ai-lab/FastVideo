# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 — audio-to-audio variation example.

User story (musician, late at night):
    "I generated this 12-second lo-fi loop earlier and I love the chord
    progression and overall vibe, but the snare hit at 0:08 sounds wrong
    and the rhythm feels stiff. I don't want to start over from scratch
    and lose what's working — I want the model to keep the harmony and
    mood but reroll the percussion + groove."

User story (sound designer, on a deadline):
    "I have one good 'sword clang' SFX. The art director wants 8 sibling
    variations that all feel like the same sword from different angles —
    same metal, same weight, slightly different impact. I'd rather
    refine my one good take than text-prompt my way through 50 misses."

How it works:
    A2A variation feeds the user's reference clip into the pipeline as
    `init_audio`. Internally:
      1. The Oobleck VAE encodes the clip into a latent.
      2. The denoising loop starts from `init_latent + noise * sigma_max`
         where `sigma_max == init_noise_level` (lower = closer to the
         reference, higher = more freedom to drift).
      3. The text prompt + duration conditioning still apply.

    Mirrors upstream `generate_diffusion_cond(init_audio=..., init_noise_level=...)`.

Tunable knobs (the "creative dials"):
    init_noise_level
      0.5  — keep most of the reference (subtle reroll)
      1.0  — moderate variation (default in upstream when init_audio set)
      5.0  — heavy reroll, reference acts as a soft sketch
      500.0 — equivalent to no reference (full T2A)

Prerequisites: same as `basic_stable_audio.py`.
"""
from fastvideo import VideoGenerator

PROMPT = "Lo-fi hip hop instrumental with vinyl crackle and gentle piano."
# Path to any audio-bearing file (wav, mp3, mp4, m4a, flac, ...). Set to
# `None` to skip A2A and run plain T2A.
INIT_AUDIO_PATH: str | None = "/home/william5lin/FastVideo2/outputs_audio/stable_audio_basic/output_stable_audio_2.mp4"
INIT_NOISE_LEVEL = 1.0


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        num_gpus=1,
    )
    generator.generate_video(
        prompt=PROMPT,
        output_path="outputs_audio/stable_audio_a2a/output_a2a.mp4",
        save_video=True,
        audio_end_in_s=6.0,
        init_audio=INIT_AUDIO_PATH,
        init_noise_level=INIT_NOISE_LEVEL,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
