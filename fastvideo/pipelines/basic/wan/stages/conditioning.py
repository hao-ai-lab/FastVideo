# SPDX-License-Identifier: Apache-2.0
"""Prepare normalized first-frame latents before Wan's sampling stages."""

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


class WanFirstFrameEncodingStage(PipelineStage):

    def __init__(self, vae, *, causal: bool = False):
        super().__init__()
        self.vae = vae
        self.causal = causal

    def forward(self, batch, fastvideo_args):
        batch.first_frame_latent = None
        if batch.pil_image is None or not (self.causal or fastvideo_args.pipeline_config.ti2v_task):
            return batch
        assert self.vae is not None, "VAE is required for first-frame conditioning"
        original_device = next(self.vae.parameters()).device
        self.vae = self.vae.to(get_local_torch_device())
        latent = self.vae.encode(batch.pil_image).mean.float()
        # Preserve dense TI2V's restore-before-normalization order and causal
        # DMD's explicit CPU offload after normalization.
        offload = getattr(fastvideo_args, "vae_cpu_offload", False)
        if offload and not self.causal:
            self.vae = self.vae.to(original_device)
        shift = getattr(self.vae, "shift_factor", None)
        if shift is not None:
            latent -= shift.to(latent.device, latent.dtype) if isinstance(shift, torch.Tensor) else shift
        scale = self.vae.scaling_factor
        latent = latent * (scale.to(latent.device, latent.dtype) if isinstance(scale, torch.Tensor) else scale)
        if offload and self.causal:
            self.vae = self.vae.to("cpu")
        batch.first_frame_latent = latent
        return batch

    def verify_input(self, batch, fastvideo_args):
        result = VerificationResult()
        if batch.pil_image is not None and (self.causal or fastvideo_args.pipeline_config.ti2v_task):
            result.add_check("pil_image", batch.pil_image, V.is_tensor)
        return result

    def verify_output(self, batch, fastvideo_args):
        result = VerificationResult()
        if batch.pil_image is not None and (self.causal or fastvideo_args.pipeline_config.ti2v_task):
            result.add_check("first_frame_latent", batch.first_frame_latent, [V.is_tensor, V.with_dims(5)])
        return result
