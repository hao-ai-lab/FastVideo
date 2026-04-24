# SPDX-License-Identifier: Apache-2.0
"""Minimal user-runnable example for the daVinci-MagiHuman base AV pipeline.

Produces an mp4 with both video (Wan 2.2 TI2V-5B VAE) and audio (Stable
Audio Open 1.0 VAE, first-class FastVideo port in
`fastvideo/models/vaes/oobleck.py`) muxed together via PyAV.

Prerequisites (one-off):

  # 1) Accept terms of use on the gated HF repos with your HF_TOKEN:
  #    - https://huggingface.co/google/t5gemma-9b-9b-ul2
  #    - https://huggingface.co/stabilityai/stable-audio-open-1.0
  #    Lazy-loaded on first forward; the pipeline fails loudly if either
  #    access is denied.
  # 2) Clone upstream repo (only needed for parity tests, not this example):
  # git clone --depth 1 https://github.com/GAIR-NLP/daVinci-MagiHuman.git
  # 3) Convert the raw MagiHuman base checkpoint to a Diffusers layout
  #    and bundle the Wan 2.2 TI2V-5B VAE:
  python scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py \\
      --source GAIR/daVinci-MagiHuman \\
      --output converted_weights/magi_human_base \\
      --bundle-vae
  # Optional: also bundle the audio VAE (`--bundle-audio-vae`) and/or the
  # T5-Gemma encoder (`--bundle-text-encoder`). Either works lazy too.
"""
from fastvideo import VideoGenerator


PROMPT = (
    "A warm afternoon scene: a person sits on a park bench reading a book, "
    "surrounded by softly swaying trees."
)


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "converted_weights/magi_human_base",
        num_gpus=1,
    )
    output_path = "outputs_video/magi_human_basic/output_magi_human_t2v.mp4"
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=True,
        # Defaults pulled from the registered preset (magi_human_base_t2v):
        # height=256, width=448, fps=25, num_inference_steps=32, seed=42.
        # Override here only if you have a specific QA scenario.
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
