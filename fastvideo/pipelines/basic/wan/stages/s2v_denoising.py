# SPDX-License-Identifier: Apache-2.0
"""Wan-S2V conditioning around the shared dense sampling loop."""

from typing import Any

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage


class WanS2VDenoisingStage(DenoisingStage):
    """Audio-driven Wan sampling.

    The reference image, audio and motion history reach the transformer by
    name (``ref_latents``, ``audio_input``, ``motion_latents``) as conditioning
    tokens. They are never concatenated onto the input channels the way the
    I2V models take their image latent -- that would double the channel count
    and break the patch embedding.
    """

    def prepare_model_kwargs(self, batch: ForwardBatch,
                             fastvideo_args: FastVideoArgs) -> tuple[dict[str, Any], dict[str, Any]]:
        """(conditional, unconditional) kwargs for the S2V transformer.

        ``prepare_extra_func_kwargs`` drops every key the forward does not
        name, so a rename on either side fails loudly in the tests rather than
        silently dropping the audio. The unconditional variant zeroes the audio,
        matching the official recipe (speech2video.py passes ``0.0 *
        audio_input`` for the CFG negative pass: guidance contrasts over text
        and audio jointly).
        """
        kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "audio_input": batch.audio_embeds,
                "ref_latents": batch.image_latent,
                "motion_latents": batch.extra.get("motion_latents"),
                "cond_states": batch.extra.get("cond_states"),
                "motion_frames": getattr(fastvideo_args.pipeline_config, "motion_frames", None),
            },
        )
        # A None would override the model's own default, so drop empties.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        uncond_kwargs = dict(kwargs)
        if "audio_input" in uncond_kwargs:
            uncond_kwargs["audio_input"] = torch.zeros_like(uncond_kwargs["audio_input"])
        return kwargs, uncond_kwargs

    def prepare_model_input(self, latents, batch, target_dtype, state) -> torch.Tensor:
        # ``batch.image_latent`` is the reference image and goes in by name above.
        return latents.to(target_dtype)
