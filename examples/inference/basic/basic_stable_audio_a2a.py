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

Pass `init_audio=path/to/clip` (any wav/mp3/mp4/m4a/flac the standard
deps decode) and the model will use it as a starting point for the
text prompt instead of pure noise.

Picking `init_audio_strength` (0.0 to 1.0):

    Higher = closer to the source clip. Lower = more transformation.
    (Same convention as the "Input Audio Strength" slider in
    Stability's commercial Stable Audio web UI, so values transfer
    directly.)

      | strength | what you get                                       |
      |----------|----------------------------------------------------|
      |   1.00   | Output ≈ reference. No transformation.             |
      |   0.85   | Texture micro-variation only.                      |
      |   0.70   | Light reroll, same instruments.                    |
      |   0.60   | Default. Instrument identity is replaceable        |
      |          | (cello can take over from piano on the same notes).|
      |   0.50   | Heavy — only melody / chord progression survives.  |
      |   0.30   | Reference acts as a loose mood prompt.             |
      |   0.00   | Plain T2A — reference ignored.                     |

    Rule of thumb by intent:
      * "Fix one part of this clip"            -> 0.75 .. 0.85
      * "Same notes, different instrument"     -> 0.55 .. 0.65
      * "Same chord progression, new content"  -> 0.40 .. 0.55
      * "Use this as a loose mood prompt"      -> 0.20 .. 0.35

    If the reference timbre is bleeding through more than you want,
    lower it; if the structure is gone, raise it.

Prerequisites: same as `basic_stable_audio.py`.
"""
from fastvideo import VideoGenerator

PROMPT = "Change the piano to a cello playing the same notes"
# Path to any audio-bearing file (wav, mp3, mp4, m4a, flac, ...).
# Set to `None` to skip A2A and run plain T2A.
INIT_AUDIO_PATH: str | None = None
# Reference fidelity in [0, 1] -- higher = closer to source.
INIT_AUDIO_STRENGTH = 0.6


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        num_gpus=1,
    )
    generator.generate_video(
        prompt=PROMPT,
        output_path="outputs_audio/stable_audio_a2a/output_a2a.wav",
        save_video=True,
        audio_end_in_s=6.0,
        init_audio=INIT_AUDIO_PATH,
        init_audio_strength=INIT_AUDIO_STRENGTH,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
