# SPDX-License-Identifier: Apache-2.0
"""Minimal user-runnable example for Stable Audio Open 1.0 (text-to-audio).

Prerequisites:
  1. Accept the terms on https://huggingface.co/stabilityai/stable-audio-open-1.0
     and export your HF token in the shell:
         export HF_TOKEN=hf_...
  2. Have the optional inference deps installed:
         pip install k_diffusion einops_exts alias_free_torch torchsde

The pipeline is fully FastVideo-native — no `from diffusers import` at
runtime. Components used:
  * DiT:         `fastvideo.models.dits.stable_audio.StableAudioDiT`
  * Conditioner: `fastvideo.models.encoders.stable_audio_conditioner.StableAudioMultiConditioner`
  * VAE:         `fastvideo.models.vaes.oobleck.OobleckVAE`
  * Sampler:     `k_diffusion.sampling.sample_dpmpp_3m_sde`
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
        # override num_inference_steps / guidance_scale here for QA.
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
