# SPDX-License-Identifier: Apache-2.0
"""Minimal user-runnable example for the daVinci-MagiHuman DMD-2 distilled
text-to-AV pipeline.

Same arch as the base model (`basic_magi_human.py`) but with DMD-2 distilled
weights: 8 denoising steps, no classifier-free guidance. ~4x faster than
base at the same 256x480 resolution. Mirrors upstream
`daVinci-MagiHuman/example/distill/run_T2V.sh`.

Prerequisites (one-off):

  # 1) Accept terms on the gated HF repos with your HF_TOKEN:
  #    - https://huggingface.co/google/t5gemma-9b-9b-ul2
  #    - https://huggingface.co/stabilityai/stable-audio-open-1.0
  # 2) Convert the distill subfolder of GAIR/daVinci-MagiHuman:
  python scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py \\
      --source GAIR/daVinci-MagiHuman \\
      --subfolder distill \\
      --output converted_weights/magi_human_distill \\
      --bundle-vae \\
      --cast-bf16
  # `--cast-bf16` is recommended (61 GB fp32 -> 30 GB bf16); the FV pipeline
  # loads bf16 anyway, and the conversion keeps norms / RoPE bands fp32.
"""
from fastvideo import VideoGenerator


PROMPT = (
    "A warm afternoon scene: a person sits on a park bench reading a book, "
    "surrounded by softly swaying trees."
)


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "converted_weights/magi_human_distill",
        num_gpus=1,
    )
    output_path = "outputs_video/magi_human_basic/output_magi_human_distill.mp4"
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=True,
        # Defaults pulled from the registered preset (magi_human_distill):
        # height=256, width=480, fps=25, num_inference_steps=8, cfg=1, seed=42.
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
