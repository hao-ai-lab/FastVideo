# SPDX-License-Identifier: Apache-2.0
"""
SVI Image VAE Encoding Stage for Stable-Video-Infinity I2V generation.

This stage handles encoding a list of motion frames and reference-padded
slots to latent space with SVI-specific mask construction for I2V
conditioning. Used by the WanSVIImageToVideoPipeline.
"""

import PIL
import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import ExecutionMode, FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.vaes.common import ParallelTiledVAE
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.image_encoding import ImageVAEEncodingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class SVIImageVAEEncodingStage(ImageVAEEncodingStage):
    """
    Stage for encoding motion frames and reference padding for Stable-Video-Infinity.
    """

    vae: ParallelTiledVAE

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if fastvideo_args.mode != ExecutionMode.INFERENCE:
            raise NotImplementedError("SVIImageVAEEncodingStage only supports inference mode")

        assert isinstance(batch.height, int) and isinstance(batch.width, int)
        assert isinstance(batch.num_frames, int)
        height, width, num_frames = batch.height, batch.width, batch.num_frames

        first_frames = batch.svi_first_frames
        if not first_frames:
            assert isinstance(batch.pil_image, PIL.Image.Image)
            first_frames = [batch.pil_image]
        random_ref_frame = batch.svi_random_ref_frame
        if random_ref_frame is None:
            random_ref_frame = first_frames[0]
        ref_pad_num = batch.svi_ref_pad_num if batch.svi_ref_pad_num is not None else 0
        ref_pad_cfg = batch.svi_ref_pad_cfg

        num_condition_frames = len(first_frames)
        remaining_frames = num_frames - num_condition_frames
        if remaining_frames < 0:
            raise ValueError(f"num_frames={num_frames} smaller than len(first_frames)={num_condition_frames}")

        device = get_local_torch_device()
        self.vae = self.vae.to(device)
        vae_scale = self.vae.spatial_compression_ratio
        latent_height = height // vae_scale
        latent_width = width // vae_scale
        temporal_compression = self.vae.temporal_compression_ratio

        # 1) Mask. (1, T, H/8, W/8) -> reshape -> (4, T_lat, H/8, W/8).
        msk = torch.ones(1, num_frames, latent_height, latent_width, device=device, dtype=torch.float32)
        if ref_pad_cfg:
            msk[:, num_condition_frames:] = 0
        else:
            msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=temporal_compression, dim=1), msk[:, 1:]],
            dim=1,
        )
        msk = msk.view(1, msk.shape[1] // temporal_compression, temporal_compression, latent_height, latent_width)
        msk = msk.transpose(1, 2)[0]  # (4, T_lat, H/8, W/8)

        # 2) Condition frames as (1, 3, num_condition_frames, H, W).
        condition_frames: list[torch.Tensor] = []
        for frame in first_frames:
            t = self.preprocess(frame, vae_scale_factor=vae_scale, height=height, width=width).to(device=device,
                                                                                                  dtype=torch.float32)
            # preprocess returns (1, 3, H, W); make it (1, 3, 1, H, W) for the temporal cat.
            condition_frames.append(t.unsqueeze(2))
        vae_input_condition = torch.cat(condition_frames, dim=2)  # (1, 3, num_cond, H, W)

        # 3) Padding frames as (1, 3, remaining_frames, H, W).
        if remaining_frames == 0:
            vae_input_pad = torch.empty(
                vae_input_condition.shape[0],
                vae_input_condition.shape[1],
                0,
                height,
                width,
                device=device,
                dtype=torch.float32,
            )
        elif ref_pad_num == 0:
            vae_input_pad = vae_input_condition.new_zeros(vae_input_condition.shape[0], vae_input_condition.shape[1],
                                                          remaining_frames, height, width)
        elif ref_pad_num == -1:
            ref_t = self.preprocess(random_ref_frame, vae_scale_factor=vae_scale, height=height,
                                    width=width).to(device=device, dtype=torch.float32).unsqueeze(2)
            vae_input_pad = ref_t.repeat(1, 1, remaining_frames, 1, 1)
        elif ref_pad_num > 0:
            ref_t = self.preprocess(random_ref_frame, vae_scale_factor=vae_scale, height=height,
                                    width=width).to(device=device, dtype=torch.float32).unsqueeze(2)
            ref_pad = ref_t.repeat(1, 1, min(ref_pad_num, remaining_frames), 1, 1)
            if remaining_frames > ref_pad_num:
                zero_pad = ref_t.new_zeros(ref_t.shape[0], ref_t.shape[1], remaining_frames - ref_pad_num, height,
                                           width)
                vae_input_pad = torch.cat([ref_pad, zero_pad], dim=2)
            else:
                vae_input_pad = ref_pad
        else:
            raise ValueError(f"Unsupported ref_pad_num={ref_pad_num} (expected -1, 0, or positive int)")

        video_condition = torch.cat([vae_input_condition, vae_input_pad], dim=2)
        assert video_condition.shape[2] == num_frames, video_condition.shape

        # 4) VAE encode (autocast handling mirrors parent stage).
        vae_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (vae_dtype != torch.float32) and not fastvideo_args.disable_autocast
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)
            encoder_output = self.vae.encode(video_condition)

        if batch.generator is None:
            raise ValueError("Generator must be provided")
        latent = self.retrieve_latents(encoder_output, batch.generator)

        if hasattr(self.vae, "shift_factor") and self.vae.shift_factor is not None:
            shift = self.vae.shift_factor
            if isinstance(shift, torch.Tensor):
                shift = shift.to(latent.device, latent.dtype)
            latent = latent - shift

        scaling = self.vae.scaling_factor
        if isinstance(scaling, torch.Tensor):
            scaling = scaling.to(latent.device, latent.dtype)
        latent = latent * scaling

        # 5) y = concat(mask, latent) along channel dim. latent is (1, 16, T_lat, H_lat, W_lat).
        mask_batched = msk.unsqueeze(0).to(latent.device, latent.dtype)
        batch.image_latent = torch.concat([mask_batched, latent], dim=1)  # (1, 20, T_lat, H_lat, W_lat)

        if hasattr(self, "maybe_free_model_hooks"):
            self.maybe_free_model_hooks()
        self.vae.to("cpu")
        return batch

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)])
        return result
