# SPDX-License-Identifier: Apache-2.0
"""Wan-S2V-specific pipeline stages.

These replicate two steps of the official runner (``wan/speech2video.py``) that
the shared stages do differently:

* The reference image is VAE-encoded **alone** as one frame. The shared
  ``ImageVAEEncodingStage`` instead builds an I2V-style zero-padded video and
  encodes all of it, which is a different conditioning format entirely.
* Decoding prepends the reference latent so the causal VAE has temporal context
  (``decode_latents = cat([ref_latents, latents], dim=2)``), then keeps only
  the generated span and drops the first 3 warm-up frames.
"""
import PIL.Image
import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.decoding import DecodingStage
from fastvideo.pipelines.stages.image_encoding import ImageVAEEncodingStage
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE


class S2VRefImageEncodingStage(ImageVAEEncodingStage):
    """VAE-encode the reference image as a single latent frame.

    Writes ``batch.image_latent`` with shape [B, C, 1, h, w]. Deterministic
    (distribution mode, not a sample): the reference is ground truth to
    preserve, and the official runner's native VAE encode is deterministic too.
    """

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.pil_image is not None and isinstance(batch.pil_image, PIL.Image.Image)
        assert batch.height is not None and batch.width is not None

        self.vae.to(get_local_torch_device())  # Module.to is in-place; no rebind, keeps mypy able to type self.vae
        image = self.preprocess(batch.pil_image,
                                vae_scale_factor=self.vae.spatial_compression_ratio,
                                height=batch.height,
                                width=batch.width)
        image = image.to(get_local_torch_device(), dtype=torch.float32).unsqueeze(2)  # [B, C, 1, H, W]

        vae_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (vae_dtype != torch.float32) and not fastvideo_args.disable_autocast
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
            if not vae_autocast_enabled:
                image = image.to(vae_dtype)
            latent = self.retrieve_latents(self.vae.encode(image), generator=None, sample_mode="argmax")

        # Same normalisation the other latents go through.
        if getattr(self.vae, "shift_factor", None) is not None:
            shift = self.vae.shift_factor
            latent = latent - (shift.to(latent.device, latent.dtype) if isinstance(shift, torch.Tensor) else shift)
        scale = self.vae.scaling_factor
        latent = latent * (scale.to(latent.device, latent.dtype) if isinstance(scale, torch.Tensor) else scale)

        batch.image_latent = latent
        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")
        return batch

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        # One latent frame exactly: the DiT parks these tokens at a dedicated
        # time index and its RoPE grid claims precisely one frame.
        result.add_check("image_latent", batch.image_latent,
                         lambda v: v is not None and v.dim() == 5 and v.shape[2] == 1)
        return result


class S2VDecodingStage(DecodingStage):
    """Decode with the reference latent prepended, official-runner style.

    The Wan VAE is causal in time: the first frames decode with less context
    and come out degraded. The official runner therefore decodes
    ``[ref_latent | generated latents]``, keeps the trailing ``num_frames``
    pixels, and drops 3 more warm-up frames on the first clip.
    """

    _WARMUP_FRAMES = 3

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.image_latent is not None, "S2V decode needs the reference latent for temporal context"
        num_frames = batch.num_frames
        batch.latents = torch.cat([batch.image_latent.to(batch.latents.device, batch.latents.dtype), batch.latents],
                                  dim=2)
        batch = super().forward(batch, fastvideo_args)
        if fastvideo_args.output_type != "latent" and batch.output is not None and num_frames is not None:
            batch.output = batch.output[:, :, -num_frames:][:, :, self._WARMUP_FRAMES:]
        return batch
