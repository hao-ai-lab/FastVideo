# SPDX-License-Identifier: Apache-2.0
"""Minimal user-runnable example for the daVinci-MagiHuman base AV pipeline.

Produces an mp4 with both video (Wan 2.2 TI2V-5B VAE) and audio (Stable
Audio Open 1.0 VAE, first-class FastVideo port in
`fastvideo/models/vaes/oobleck.py`) muxed together via PyAV.

Prerequisites (one-off):

  # Accept terms of use on the gated HF repos with your HF_TOKEN:
  #   - https://huggingface.co/google/t5gemma-9b-9b-ul2
  #   - https://huggingface.co/stabilityai/stable-audio-open-1.0
  # All four cross-variant shared components (Wan 2.2 VAE, T5-Gemma
  # encoder + tokenizer, Stable Audio VAE) are lazy-loaded from their
  # canonical upstream HF repos on first build, so a single ~25 GB
  # cache is shared across every MagiHuman variant.

The umbrella HF repo `FastVideo/MagiHuman-Diffusers` holds all four
variants (base / distill / sr_540p / sr_1080p) under sibling subfolders
and FastVideo will download just the requested subfolder. Local
conversion via `scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py`
is also supported.
"""
from fastvideo import VideoGenerator


PROMPT = (
    "A warm afternoon scene: a person sits on a park bench reading a book, "
    "surrounded by softly swaying trees."
)


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "FastVideo/MagiHuman-Diffusers/base",
        num_gpus=1,
    )
    output_path = "outputs_video/magi_human_basic/output_magi_human.mp4"
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=True,
        # Defaults pulled from the registered preset (magi_human_base):
        # height=256, width=448, fps=25, num_inference_steps=32, seed=42.
        # Override here only if you have a specific QA scenario.
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
