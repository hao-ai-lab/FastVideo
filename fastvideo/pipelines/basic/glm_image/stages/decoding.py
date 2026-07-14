# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.stages.decoding import DecodingStage
from fastvideo.utils import PRECISION_TO_TYPE


class GlmImageDecodingStage(DecodingStage):

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        self.vae.to(get_local_torch_device())
        latents = latents.to(get_local_torch_device())

        vae_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.vae_precision]
        vae_autocast = (vae_dtype != torch.float32 and not fastvideo_args.disable_autocast)

        latents = self._denormalize_latents(latents)
        if latents.dim() == 5:
            latents = latents.squeeze(2)

        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast:
                latents = latents.to(vae_dtype)
            decoded = self.vae.decode(latents)

        image = decoded.sample if hasattr(decoded, "sample") else decoded
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.unsqueeze(2)
