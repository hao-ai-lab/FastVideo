# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 — inpainting / outpainting (loop extension) example.

User story (loop extension — the killer app):
    "I have a 6-second drum loop my client likes. They want it as
    background bed for a 30-second ad. I need it to loop seamlessly,
    but a hard cut every 6s sounds bad. Let me extend it to 30s,
    keeping the first 6s exactly as-is and letting the model continue
    the groove for the remaining 24s."

User story (audio repair):
    "There's a microphone bump at 0:14 in this 30-second field
    recording — really obvious in headphones. Mask out 0:13 to 0:15
    and let the model regenerate plausible ambience that blends in.
    Everything else stays exactly as I recorded it."

User story (transition smoothing):
    "I have two 10-second clips I want to crossfade. Mask out a 1s
    overlap region in the middle and let the model invent a coherent
    transition between the two."

How it works (RePaint-style blending):
    Stable Audio Open 1.0 wasn't trained as an inpainting model
    (`model_type=diffusion_cond`, not `diffusion_cond_inpaint`), so we
    can't use the upstream's mask-conditioned approach directly. We
    use the RePaint trick instead, which works on any v-prediction
    diffusion model:

      1. Encode the reference clip into latent space.
      2. At every denoising step `i`, replace the kept region of the
         in-flight latent (where mask == 1) with the reference
         re-noised to the next timestep's sigma. Only the unkept
         region (mask == 0) is freely denoised.
      3. After the loop, the kept region is exactly the reference;
         the unkept region is freshly generated content.

    This is approximate compared to a properly trained inpainting
    checkpoint — the seam between kept/unkept can have slight EQ
    discontinuity — but it works on the existing public model.

Tunable: the mask is a 1-D tensor in {0, 1} at the model's sample
rate. Conventions:
    1.0 = keep this sample from the reference
    0.0 = regenerate this sample

Prerequisites: same as `basic_stable_audio.py`.
"""
from pathlib import Path

import torch
import torchaudio

from fastvideo import VideoGenerator

PROMPT = "Steady lo-fi hip hop drum loop with vinyl crackle."
REFERENCE_AUDIO_PATH: str | None = None  # set to a wav/mp3 to extend a real loop
KEEP_SECONDS = 6.0       # the first KEEP_SECONDS of the reference are preserved
TOTAL_SECONDS = 12.0     # extend to this duration


def _load_reference(path: str, target_sr: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    return waveform.float()


def _synthetic_loop(target_sr: int, seconds: float) -> torch.Tensor:
    """Stand-in reference: a 6s of stereo amplitude-modulated noise that
    sounds vaguely like a beat. Replace with a real loop in production.
    """
    n = int(seconds * target_sr)
    base = torch.randn(2, n) * 0.1
    # Periodic envelope to fake a beat (~ 4 Hz pulse).
    env = 0.5 + 0.5 * torch.sin(2 * torch.pi * 4 * torch.linspace(0, seconds, n))
    return (base * env).unsqueeze(0).contiguous()


def main() -> None:
    sample_rate = 44100
    if REFERENCE_AUDIO_PATH is not None and Path(REFERENCE_AUDIO_PATH).exists():
        ref = _load_reference(REFERENCE_AUDIO_PATH, sample_rate)
    else:
        print("No REFERENCE_AUDIO_PATH set; using a synthetic AM-noise loop.")
        ref = _synthetic_loop(sample_rate, KEEP_SECONDS)

    # Build the mask in the audio domain at the model's sample rate.
    # `KEEP_SECONDS` of 1.0s, then zeros up to TOTAL_SECONDS.
    keep_samples = int(KEEP_SECONDS * sample_rate)
    total_samples = int(TOTAL_SECONDS * sample_rate)
    mask = torch.zeros(total_samples, dtype=torch.float32)
    mask[:keep_samples] = 1.0

    # Pad the reference up to total_samples so encoding aligns. The kept
    # region is the only thing the mask preserves; the rest is freely
    # regenerated regardless of what's in the padded region.
    padded_ref = torch.zeros((1, ref.shape[1], total_samples), dtype=torch.float32)
    padded_ref[..., :ref.shape[-1]] = ref

    generator = VideoGenerator.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        num_gpus=1,
    )
    output_path = "outputs_audio/stable_audio_inpaint/output_inpaint.mp4"
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=True,
        audio_end_in_s=TOTAL_SECONDS,
        inpaint_audio=padded_ref,
        inpaint_mask=mask,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
