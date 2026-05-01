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
from fastvideo import VideoGenerator

PROMPT = "Steady lo-fi hip hop drum loop with vinyl crackle."
# Required: path to the reference audio file (wav, mp3, mp4, m4a, flac,
# ...) you want to extend or repair. The pipeline raises if a mask is
# passed without a reference, so this must be a real path.
REFERENCE_AUDIO_PATH = "path/to/your/loop.wav"
KEEP_SECONDS = 6.0       # first KEEP_SECONDS preserved exactly
TOTAL_SECONDS = 12.0     # extend the loop to this duration


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        num_gpus=1,
    )
    generator.generate_video(
        prompt=PROMPT,
        output_path="outputs_audio/stable_audio_inpaint/output_inpaint.wav",
        save_video=True,
        audio_end_in_s=TOTAL_SECONDS,
        inpaint_audio=REFERENCE_AUDIO_PATH,
        # Tuple form: keep first KEEP_SECONDS, regenerate the rest.
        inpaint_mask=(KEEP_SECONDS, TOTAL_SECONDS),
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
