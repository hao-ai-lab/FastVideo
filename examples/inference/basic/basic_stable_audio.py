# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 — text-to-audio (baseline) example.

User story (game-audio designer, prototyping):
    "I'm prototyping a level and I need 6 seconds of background
    ambience — gentle wind, distant thunder, a hint of birdsong. I
    don't want to dig through a sound library; I want to type what I
    hear in my head and get a wav back. If it's wrong I'll iterate
    on the prompt. This is the first stop."

User story (musician sketching ideas):
    "I want to bounce a 30s lo-fi drum loop to use as a placeholder
    bed while I build the rest of the track. Type prompt, get audio,
    drop into the DAW. The actual production beat I'll record
    myself, but I need *something* to write the chords against."

User story (researcher exploring the model):
    "First time touching Stable Audio Open — what does it sound
    like at default settings? This is the smallest amount of code
    that goes from prompt to mp4."

How it works:
    Pure text-to-audio (T2A). The pipeline runs:
        T5 + NumberConditioner -> StableAudioDiT -> Oobleck VAE
    via the `dpmpp-3m-sde` k-diffusion sampler. All components are
    FastVideo-native — no diffusers / transformers model imports at
    runtime (see REVIEW item 30). Mirrors upstream
    `stable_audio_tools.inference.generation.generate_diffusion_cond`
    bit-for-bit (~0.2% abs_mean drift on 25 steps).

Tunable knobs (the "creative dials"):
    audio_end_in_s
        1–6   — quick ideation (sub-10s wall clock at 100 steps)
        10–30 — full musical phrase / loop length (the README example
                uses 30s)
        47.5  — model maximum (full sample_size = 2097152 / 44100 Hz)
    num_inference_steps
        25  — fast preview, occasional artifacts
        100 — preset default (matches the HF model card)
        250 — diminishing returns past here
    guidance_scale
        3   — looser, more variation per seed
        7   — preset default; matches README
        12+ — sharper but can sound "fried"

Prerequisites:
  1. Accept the terms on https://huggingface.co/stabilityai/stable-audio-open-1.0
     and export your HF token in the shell:
         export HF_TOKEN=hf_...
  2. Install optional inference deps (one-time):
         pip install k_diffusion einops_exts alias_free_torch torchsde
"""
from fastvideo import VideoGenerator

PROMPT = "Lo-fi hip hop instrumental with vinyl crackle and gentle piano."


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "FastVideo/stable-audio-open-1.0-Diffusers",
        num_gpus=1,
    )
    output_path = "outputs_audio/stable_audio_basic/output_stable_audio.wav"
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=True,
        # 6-second clip; the model max is ~47.5s.
        audio_end_in_s=6.0,
        # The registered preset gives 100 steps + CFG=7.0 by default;
        # override num_inference_steps / guidance_scale here for QA.
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
