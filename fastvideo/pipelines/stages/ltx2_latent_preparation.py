# SPDX-License-Identifier: Apache-2.0
"""
Latent preparation stage for LTX-2 pipelines.
"""

import torch
from diffusers.utils.torch_utils import randn_tensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


class LTX2LatentPreparationStage(PipelineStage):
    """Prepare initial LTX-2 latents without relying on a diffusers scheduler."""

    def __init__(self, transformer) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        latent_num_frames = self._adjust_video_length(batch, fastvideo_args)

        if not batch.prompt_embeds:
            batch_size = 1
        elif isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            batch_size = batch.prompt_embeds[0].shape[0]

        batch_size *= batch.num_videos_per_prompt

        if not batch.prompt_embeds:
            transformer_dtype = next(self.transformer.parameters()).dtype
            device = get_local_torch_device()
            dummy_prompt = torch.zeros(
                batch_size,
                0,
                self.transformer.hidden_size,
                device=device,
                dtype=transformer_dtype,
            )
            batch.prompt_embeds = [dummy_prompt]
            batch.negative_prompt_embeds = []
            batch.do_classifier_free_guidance = False

        dtype = batch.prompt_embeds[0].dtype
        device = get_local_torch_device()
        generator = batch.generator
        latents = batch.latents
        num_frames = latent_num_frames if latent_num_frames is not None else batch.num_frames
        height = batch.height
        width = batch.width

        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        spatial_ratio = fastvideo_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        if height % spatial_ratio != 0 or width % spatial_ratio != 0:
            raise ValueError(
                f"Height and width must be divisible by {spatial_ratio} "
                f"but are {height} and {width}.")
        shape = (
            batch_size,
            self.transformer.num_channels_latents,
            num_frames,
            height // spatial_ratio,
            width // spatial_ratio,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, "
                f"but requested an effective batch size of {batch_size}.")

        if latents is None:
            latents = randn_tensor(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            latents = latents.to(device)

        batch.latents = latents
        batch.raw_latent_shape = shape
        return batch

    def _adjust_video_length(self, batch: ForwardBatch,
                             fastvideo_args: FastVideoArgs) -> int | None:
        if not fastvideo_args.pipeline_config.vae_config.use_temporal_scaling_frames:
            return None
        temporal_scale_factor = (fastvideo_args.pipeline_config.vae_config.
                                 arch_config.temporal_compression_ratio)
        video_length = batch.num_frames
        return int((video_length - 1) // temporal_scale_factor + 1)

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "prompt_or_embeds",
            None,
            lambda _: V.string_or_list_strings(batch.prompt) or not batch.
            prompt_embeds or V.list_not_empty(batch.prompt_embeds),
        )
        if batch.prompt_embeds:
            result.add_check("prompt_embeds", batch.prompt_embeds,
                             V.list_of_tensors)
        result.add_check("num_videos_per_prompt", batch.num_videos_per_prompt,
                         V.positive_int)
        result.add_check("generator", batch.generator,
                         V.generator_or_list_generators)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        result.add_check("raw_latent_shape", batch.raw_latent_shape, V.is_tuple)
        return result
