# SPDX-License-Identifier: Apache-2.0
"""
Encoding stage for diffusion pipelines.
"""
from typing import Optional

import PIL.Image
import torch

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.vaes.common import ParallelTiledVAE
from fastvideo.v1.models.vision_utils import (get_default_height_width,
                                              load_image, normalize,
                                              numpy_to_pt, pil_to_numpy, resize)
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage
from fastvideo.v1.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class EncodingStage(PipelineStage):
    """
    Stage for encoding pixel representations into latent space.
    
    This stage handles the encoding of pixel representations into the final
    input format (e.g., latents).
    """

    def __init__(self, vae: ParallelTiledVAE) -> None:
        self.vae: ParallelTiledVAE = vae

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Encode pixel representations into latent space.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with encoded outputs.
        """
        image_path = batch.image_path
        # TODO(will): remove this once we add input/output validation for stages
        if image_path is None:
            raise ValueError("Image Path must be provided")
        assert batch.height is not None
        assert batch.width is not None
        latent_height = batch.height // self.vae.spatial_compression_ratio
        latent_width = batch.width // self.vae.spatial_compression_ratio

        image = load_image(image_path)
        image = self.preprocess(
            image,
            vae_scale_factor=self.vae.spatial_compression_ratio,
            height=batch.height,
            width=batch.width).to(fastvideo_args.device, dtype=torch.float32)
        image = image.unsqueeze(2)
        video_condition = torch.cat([
            image,
            image.new_zeros(image.shape[0], image.shape[1],
                            batch.num_frames - 1, batch.height, batch.width)
        ],
                                    dim=2)
        video_condition = video_condition.to(device=fastvideo_args.device,
                                             dtype=torch.float32)

        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[fastvideo_args.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        # Encode Image
        with torch.autocast(device_type="cuda",
                            dtype=vae_dtype,
                            enabled=vae_autocast_enabled):
            if fastvideo_args.vae_tiling:
                self.vae.enable_tiling()
            # if fastvideo_args.vae_sp:
            #     self.vae.enable_parallel()
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)
            encoder_output = self.vae.encode(video_condition)

        generator = batch.generator
        if generator is None:
            raise ValueError("Generator must be provided")
        latent_condition = self.retrieve_latents(encoder_output, generator[0])

        # Apply shifting if needed
        if (hasattr(self.vae, "shift_factor")
                and self.vae.shift_factor is not None):
            if isinstance(self.vae.shift_factor, torch.Tensor):
                latent_condition -= self.vae.shift_factor.to(
                    latent_condition.device, latent_condition.dtype)
            else:
                latent_condition -= self.vae.shift_factor

        if isinstance(self.vae.scaling_factor, torch.Tensor):
            latent_condition = latent_condition * self.vae.scaling_factor.to(
                latent_condition.device, latent_condition.dtype)
        else:
            latent_condition = latent_condition * self.vae.scaling_factor

        mask_lat_size = torch.ones(1, 1, batch.num_frames, latent_height,
                                   latent_width)
        mask_lat_size[:, :, list(range(1, batch.num_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask,
            dim=2,
            repeats=self.vae.temporal_compression_ratio)
        mask_lat_size = torch.concat(
            [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(1, -1,
                                           self.vae.temporal_compression_ratio,
                                           latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        batch.image_latent = torch.concat([mask_lat_size, latent_condition],
                                          dim=1)

        # Offload models if needed
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        return batch

    def retrieve_latents(self,
                         encoder_output: torch.Tensor,
                         generator: Optional[torch.Generator] = None,
                         sample_mode: str = "sample"):
        if sample_mode == "sample":
            return encoder_output.sample(generator)
        elif sample_mode == "argmax":
            return encoder_output.mode()
        else:
            raise AttributeError(
                "Could not access latents of provided encoder_output")

    def preprocess(
            self,
            image: PIL.Image.Image,
            vae_scale_factor: int,
            height: Optional[int] = None,
            width: Optional[int] = None,
            resize_mode: str = "default",  # "default", "fill", "crop"
    ) -> torch.Tensor:
        image = [image]

        height, width = get_default_height_width(image[0], vae_scale_factor,
                                                 height, width)
        image = [
            resize(i, height, width, resize_mode=resize_mode) for i in image
        ]
        image = pil_to_numpy(image)  # to np
        image = numpy_to_pt(image)  # to pt

        do_normalize = True
        if image.min() < 0:
            do_normalize = False
        if do_normalize:
            image = normalize(image)

        return image
