# SPDX-License-Identifier: Apache-2.0
"""Stub example for daVinci-MagiHuman base text+image-to-AV (TI2V) — NOT YET PORTED.

Mirrors upstream `daVinci-MagiHuman/example/base/run_TI2V.sh`. The base
checkpoint is the same as T2V (`basic_magi_human.py`); only the inference
flow differs: the user passes `--image_path <path>`, the upstream pipeline
encodes it through the Wan VAE, and stitches the latent into
`latent_video[:, :, :1]` at every denoise step
(`daVinci-MagiHuman/inference/pipeline/video_generate.py:424-425`).

What is missing in FastVideo:

  * `MagiHumanBaseConfig.__post_init__` currently sets
    `vae_config.load_encoder = False` (T2V does not need it). A new
    `MagiHumanBaseI2VConfig` (or a flag on the existing config) needs to
    keep the encoder loaded.
  * A new pipeline stage (e.g. `MagiHumanReferenceImageStage`) must encode
    the input image into a `[1, z_dim, 1, H, W]` latent and stash it on
    `batch.image_latents`.
  * `MagiHumanLatentPreparationStage` and `MagiHumanDenoisingStage` need a
    branch that overwrites `latent_video[:, :, :1]` with the encoded
    image latent at every step (matches `evaluate_with_latent` line 425).
  * A new preset (e.g. `magi_human_base_ti2v`) and registry entry.

Until those land, attempting to run this file will raise.
"""
import sys


def main() -> None:
    raise NotImplementedError(
        "MagiHuman base TI2V is not yet ported. See the docstring at the "
        "top of this file for the required pipeline-side work, and "
        "`tests/local_tests/magi-human.md` for the port-status journal.",
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
