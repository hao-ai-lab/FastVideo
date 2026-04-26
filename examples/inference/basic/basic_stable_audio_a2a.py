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

Picking `init_noise_level` (the only A2A-specific dial):

    `init_noise_level` is the SDE's starting `sigma_max`. The model was
    trained on `sigma_min=0.3 .. sigma_max=500`. The denoiser walks from
    your chosen `sigma_max` down to `0.3` over `num_inference_steps`,
    starting from `init_latent + noise * sigma_max`. Higher = more
    freedom to drift from the reference; lower = stays closer.

    Useful range: roughly `0.3 .. 50`. SA Open 1.0 needs more renoise
    than visual diffusion models to do cross-instrument timbre swaps —
    don't be afraid to push `sigma_max` higher than your image-diffusion
    intuition suggests.

      | value | what you get                                       |
      |-------|----------------------------------------------------|
      | 0.3   | VAE round-trip (no diffusion). Output ≈ reference. |
      | 1     | Almost unchanged. Texture micro-variation only.    |
      | 2     | Light variation — same instruments, slight reroll  |
      |       | of dynamics + transients.                          |
      | 5     | Moderate reroll — keeps melody + chord progression,|
      |       | starts shifting timbres but instrument identity    |
      |       | usually survives.                                  |
      | 7     | Heavier reroll — instrument identity is replaceable|
      |       | (cello can take over from piano on the same notes).|
      |       | Default for the example prompt above.              |
      | 15    | Only the high-level rhythm / mood survives;        |
      |       | arrangement is freely regenerated.                 |
      | 50+   | Reference is barely visible. Prompt dominates.     |
      | 500   | Equivalent to T2A from scratch (full sigma range). |

    Rule of thumb by intent:
      * "Fix one part of this clip"            -> 1 .. 3
      * "Same notes, different instrument"     -> 5 .. 10
      * "Same chord progression, new content"  -> 10 .. 20
      * "Use this as a loose mood prompt"      -> 30+

    Empirically calibrated against `PROMPT` below ("Change the piano to
    a cello playing the same notes"). Different prompt/reference combos
    can shift the sweet spot by ±2; if the reference timbre is bleeding
    through, raise it; if the structure is gone, lower it.

    Note for users coming from stableaudio.com: the commercial Stable
    Audio web UI exposes a "Reference Audio Strength" slider with
    inverted semantics — higher = closer to reference. This dial is
    upstream's `sigma_max`, so higher = more freedom from the reference.
    The open-source 1.0 community hasn't published recalibrated
    defaults; the gradio default is 1.0 (which we found insufficient
    for instrument swaps).

Prerequisites: same as `basic_stable_audio.py`.
"""
from fastvideo import VideoGenerator

PROMPT = "Change the piano to a cello playing the same notes"
# Path to any audio-bearing file (wav, mp3, mp4, m4a, flac, ...). Set to
# `None` to skip A2A and run plain T2A.
INIT_AUDIO_PATH: str | None = None
# Renoise level for the SDE start. See "Picking init_noise_level" in the
# module docstring. Useful range ~0.3 to ~50; default 7 because the
# example PROMPT is a cross-instrument timbre swap, which empirically
# needs a heavier reroll than image-diffusion intuition suggests.
INIT_NOISE_LEVEL = 7.0


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
