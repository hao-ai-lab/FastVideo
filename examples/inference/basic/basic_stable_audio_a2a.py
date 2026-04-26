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

Picking `init_audio_strength` (the A2A dial):

    `init_audio_strength` is in `[0, 1]` — **higher = closer to the
    reference, lower = more transformation**. Same convention as the
    "Input Audio Strength" slider in Stability's commercial Stable
    Audio web UI, so values transfer directly.

    Internally we map strength to the SDE's starting `sigma_max` via
    log-interpolation between the model's trained range
    (`sigma_min=0.3` ↔ `sigma_max=500`), so equal strength steps map
    to perceptually equal amounts of renoise.

      | strength | ~sigma_max | what you get                       |
      |----------|-----------:|------------------------------------|
      | 1.00     |       0.30 | VAE round-trip. Output ≈ reference.|
      | 0.85     |       0.95 | Texture micro-variation only.      |
      | 0.70     |       2.78 | Light reroll, same instruments.    |
      | 0.65     |       4.03 | Moderate reroll, instrument        |
      |          |            | identity usually survives.         |
      | 0.60     |       5.83 | Default. Instrument identity is    |
      |          |            | replaceable (cello can take over   |
      |          |            | from piano on the same notes).     |
      | 0.50     |      12.25 | Heavy — only melody / chord        |
      |          |            | progression survives, content      |
      |          |            | freely regenerated.                |
      | 0.30     |      54.00 | Reference acts as a loose mood     |
      |          |            | prompt; rhythm + key may carry     |
      |          |            | through but content is new.        |
      | 0.00     |     500.00 | Plain T2A — reference ignored.     |

    Rule of thumb by intent (calibrated against the commercial UI's
    published examples + our cello-for-piano testing):
      * "Fix one part of this clip"            -> 0.75 .. 0.85
      * "Same notes, different instrument"     -> 0.55 .. 0.65
      * "Same chord progression, new content"  -> 0.40 .. 0.55
      * "Use this as a loose mood prompt"      -> 0.20 .. 0.35

    These bands track Stability's own commercial examples
    (https://stableaudio.com/user-guide/audio-to-audio): "synth sample
    to bass guitar" sits at 75%, "piano stem to vibraphone" at 60-65%,
    looser style transfers down at 30-50%. SA Open 1.0 behaves
    similarly per dial; if the reference timbre is bleeding through,
    lower the strength; if structure is gone, raise it.

    Legacy: `init_noise_level` (raw sigma_max, 0.3..500, *higher = more
    freedom*) still works and takes precedence if both are passed.

Prerequisites: same as `basic_stable_audio.py`.
"""
from fastvideo import VideoGenerator

PROMPT = "Change the piano to a cello playing the same notes"
# Path to any audio-bearing file (wav, mp3, mp4, m4a, flac, ...). Set to
# `None` to skip A2A and run plain T2A.
INIT_AUDIO_PATH: str | None = None
# Reference fidelity in [0, 1] (higher = closer to source). See
# "Picking init_audio_strength" in the module docstring above.
INIT_AUDIO_STRENGTH = 0.6  # cross-instrument timbre swap sweet spot


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
        init_audio_strength=INIT_AUDIO_STRENGTH,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
