# SPDX-License-Identifier: Apache-2.0
"""DreamX-World pipeline stages."""

from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult

from fastvideo.pipelines.basic.dreamx_world.camera_conditioning import (
    build_dreamx_camera_condition, )

DREAMX_Y_CAMERA_KEY = "dreamx_y_camera"

logger = init_logger(__name__)


class DreamXWorldCameraConditioningStage(PipelineStage):
    """Build PRoPE camera conditioning for DreamX-World-5B-Cam."""

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        del fastvideo_args
        if DREAMX_Y_CAMERA_KEY in batch.extra:
            return batch

        action_seq = batch.extra.get("dreamx_action_seq", batch.action_list)
        action_speed_list = batch.extra.get("dreamx_action_speed_list", batch.action_speed_list)
        if action_seq is None:
            action_seq = ["w"]
        if action_speed_list is None:
            action_speed_list = [4]

        if isinstance(action_seq, str):
            action_seq = [action_seq]
        if isinstance(action_speed_list, int | float):
            action_speed_list = [action_speed_list]
        if len(action_speed_list) == 1 and len(action_seq) > 1:
            action_speed_list = list(action_speed_list) * len(action_seq)
        action_speed_list = [float(speed) for speed in action_speed_list]

        height = int(batch.height) if batch.height is not None else 704
        width = int(batch.width) if batch.width is not None else 1280
        num_frames = int(batch.num_frames)
        dtype = batch.latents.dtype if torch.is_tensor(batch.latents) else torch.float32
        device = batch.latents.device if torch.is_tensor(batch.latents) else "cpu"

        y_camera = build_dreamx_camera_condition(
            list(action_seq),
            action_speed_list,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
        )
        batch.extra[DREAMX_Y_CAMERA_KEY] = {key: value.unsqueeze(0) for key, value in y_camera.items()}
        return batch

    def verify_output(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        del fastvideo_args
        result = VerificationResult()
        y_camera = batch.extra.get(DREAMX_Y_CAMERA_KEY)
        result.add_check("dreamx_y_camera", y_camera, lambda value: isinstance(value, dict))
        if isinstance(y_camera, dict):
            result.add_check("dreamx_y_camera.viewmats", y_camera.get("viewmats"), torch.is_tensor)
            result.add_check("dreamx_y_camera.K", y_camera.get("K"), torch.is_tensor)
        return result


class DreamXWorldImageVAEEncodingStage(PipelineStage):
    """Encode the conditioning image into the first-frame latent.

    Official AR-forcing flow (AMAP-ML/DreamX-World inference_ar_forcing.py):
    the input image is resized, normalized to [-1, 1], VAE-encoded
    deterministically, and written into frame 0 of the noise — the causal
    denoiser then treats frame 0 as clean context. This stage produces
    ``batch.image_latent`` ([B, C, 1, H_lat, W_lat]); the injection into
    the latents happens in DreamXWorldARCausalDenoisingStage.
    """

    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        if batch.pil_image is None:
            # No conditioning image: the causal denoiser falls back to
            # running from pure noise (frame 0 uninitialized). Warn loudly —
            # this pipeline is registered I2V and the official flow always
            # forces from a frame.
            logger.warning("DreamXWorldARPipeline called without an input image; "
                           "first-frame context will be noise (T2V-style). Pass an "
                           "image for the official AR-forcing behavior.")
            return batch

        from fastvideo.platforms import get_local_torch_device
        from fastvideo.utils import PRECISION_TO_TYPE

        device = get_local_torch_device()
        image = batch.pil_image
        if not isinstance(image, torch.Tensor):
            import numpy as np
            import PIL.Image
            assert isinstance(image, PIL.Image.Image)
            width = batch.width if isinstance(batch.width, int) else batch.width[0]
            height = batch.height if isinstance(batch.height, int) else batch.height[0]
            image = image.convert("RGB").resize((width, height), PIL.Image.Resampling.LANCZOS)
            arr = torch.from_numpy(np.asarray(image)).float().permute(2, 0, 1) / 255.0
            image = (arr - 0.5) / 0.5  # official Normalize([0.5], [0.5])
            image = image.unsqueeze(0)  # [1, C, H, W]
        if image.dim() == 4:
            image = image.unsqueeze(2)  # [B, C, 1, H, W]
        elif image.dim() == 5:
            image = image[:, :, :1]
        image = image.to(device=device, dtype=torch.float32)

        vae_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (vae_dtype != torch.float32) and not fastvideo_args.disable_autocast
        self.vae = self.vae.to(device)
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
            if not vae_autocast_enabled:
                image = image.to(vae_dtype)
            encoder_output = self.vae.encode(image)

        # Official encode_to_latent is deterministic ((mean - mu) / sigma per
        # channel); the posterior mean + shift/scale is the FastVideo
        # equivalent of that normalization.
        latent = encoder_output.mean
        if getattr(self.vae, "shift_factor", None) is not None:
            shift = self.vae.shift_factor
            latent = latent - (shift.to(latent.device, latent.dtype) if isinstance(shift, torch.Tensor) else shift)
        scale = self.vae.scaling_factor
        latent = latent * (scale.to(latent.device, latent.dtype) if isinstance(scale, torch.Tensor) else scale)

        batch.image_latent = latent
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        return result
