# SPDX-License-Identifier: Apache-2.0
"""Stub example for daVinci-MagiHuman SR-540p text-to-AV — NOT YET PORTED.

Mirrors upstream `daVinci-MagiHuman/example/sr_540p/run_T2V.sh`.

Upstream pipeline overview (see
`daVinci-MagiHuman/inference/pipeline/video_generate.py:300-360`,
`MagiEvaluator.evaluate`):

  1. Run base model at 256x480 (`magi_human_base`) to get
     `(br_latent_video, br_latent_audio)`.
  2. Trilinear-interpolate the BR latent up to the SR latent shape
     `(latent_length, sr_latent_height, sr_latent_width)` and add
     ZeroSNR noise via `noise_value=220`.
  3. Replace audio with mostly-fresh noise:
     `randn_like(br_latent_audio) * 0.7 + br_latent_audio * 0.3`.
  4. Run a separate SR DiT (`sr_arch_config`, same arch as base, 5 steps)
     with `sr_video_txt_guidance_scale=3.5` and the cfg-trick that drops
     guidance for the first `cfg_trick_start_frame=13` frames to avoid
     overexposure on I2V hand-offs.
  5. Decode with the Wan VAE.

What is missing in FastVideo:

  * `MagiHumanSR540pConfig` (pipeline_config) wiring two DiTs (base + SR)
    plus the `sr_video_txt_guidance_scale`, `cfg_trick_*`,
    `noise_value`, `sr_audio_noise_scale`, `sr_num_inference_steps`
    knobs surfaced in `inference/common/config.py:EvaluationConfig`.
  * A two-stage pipeline: base denoise → SR latent prep
    (`MagiHumanSRLatentPreparationStage`: trilinear up + noise mix) →
    SR denoise → decode. Either a new `ComposedPipelineBase` subclass
    that runs base + SR back-to-back or a meta-pipeline that calls
    both as inner pipelines.
  * `MagiHumanSRDenoisingStage` with the per-frame cfg-trick guidance
    tensor (`evaluate_with_latent` lines 412-418).
  * Conversion script support for the `540p_sr` subfolder (the existing
    `--subfolder` choices already accept it; just need a separate
    `convert_magi_human_to_diffusers ... --subfolder 540p_sr ...`
    invocation that produces a `converted_weights/magi_human_sr_540p`
    layout with `transformer/` AND `sr_transformer/`).
  * A new preset (e.g. `magi_human_sr_540p`) and registry entry.
"""
import sys


def main() -> None:
    raise NotImplementedError(
        "MagiHuman SR-540p T2V is not yet ported. See the docstring at the "
        "top of this file for the missing pipeline-side work, and "
        "`tests/local_tests/magi-human.md` for the port-status journal.",
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
