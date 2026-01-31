# SPDX-License-Identifier: Apache-2.0
import torch


def _pack_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
) -> torch.Tensor:
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2,
                           width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2),
                              num_channels_latents * 4)
    return latents
