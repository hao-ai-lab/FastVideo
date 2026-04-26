# SPDX-License-Identifier: Apache-2.0
"""Minimal user-runnable example for Stable Audio Open 1.0 (text-to-audio).

Prerequisites:
  1. Accept terms on https://huggingface.co/stabilityai/stable-audio-open-1.0
     and export your HF token:
         export HF_TOKEN=hf_...
  2. (No conversion script needed — the upstream repo is already in
     Diffusers format.)

Note: this pipeline currently reuses three diffusers components
(StableAudioDiTModel, StableAudioProjectionModel,
CosineDPMSolverMultistepScheduler) pending first-class FastVideo
ports. The audio VAE (Oobleck) is already first-class.
"""
from fastvideo import VideoGenerator


PROMPT = "Lo-fi hip hop instrumental with vinyl crackle and gentle piano."


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        num_gpus=1,
    )
    output_path = "outputs_audio/stable_audio_basic/output_stable_audio.mp4"
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=True,
        # 6-second clip; the model max is ~47.5s.
        audio_end_in_s=6.0,
        # The registered preset gives 100 steps + CFG=7.0 by default;
        # override here only for quick QA.
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
