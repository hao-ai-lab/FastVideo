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
from pathlib import Path

import torch
import torchaudio

from fastvideo import VideoGenerator

PROMPT = "Lo-fi hip hop instrumental with vinyl crackle and gentle piano."
INIT_AUDIO_PATH: str | None = None  # set to a wav/mp3 to use a real reference
INIT_NOISE_LEVEL = 1.0


def _load_reference(path: str, target_sr: int) -> torch.Tensor:
    """Load a wav/mp3 and resample to the model's sample rate.

    Returns shape `[1, channels, samples]` ready for the pipeline.
    """
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # [1, C, S]
    return waveform.float()


def _synthetic_reference(target_sr: int) -> torch.Tensor:
    """Stand-in reference for users who don't have a wav handy: a 6-second
    sine sweep from 220 Hz to 880 Hz, stereo. Replace with a real clip
    in production.
    """
    n = 6 * target_sr
    t = torch.linspace(0, n / target_sr, n)
    freqs = torch.linspace(220.0, 880.0, n)
    mono = torch.sin(2 * torch.pi * freqs * t) * 0.3
    return mono.unsqueeze(0).repeat(2, 1).unsqueeze(0).contiguous()


def main() -> None:
    sample_rate = 44100
    if INIT_AUDIO_PATH is not None and Path(INIT_AUDIO_PATH).exists():
        init_audio = _load_reference(INIT_AUDIO_PATH, sample_rate)
    else:
        print("No INIT_AUDIO_PATH set; using a synthetic sine-sweep reference.")
        init_audio = _synthetic_reference(sample_rate)

    generator = VideoGenerator.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        num_gpus=1,
    )
    output_path = "outputs_audio/stable_audio_a2a/output_a2a.mp4"
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=True,
        audio_end_in_s=6.0,
        init_audio=init_audio,
        init_noise_level=INIT_NOISE_LEVEL,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
