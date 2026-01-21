# SPDX-License-Identifier: Apache-2.0
"""
Latent preparation stage for LTX-2 pipelines.
"""

from pathlib import Path

import torch
from diffusers.utils.torch_utils import randn_tensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


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
        latent_path = fastvideo_args.ltx2_initial_latent_path

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
            if latent_path:
                loaded_latents = self._load_initial_latent(
                    latent_path, device, dtype)
                if loaded_latents is not None:
                    latents = loaded_latents
                else:
                    latents = randn_tensor(
                        shape,
                        generator=generator,
                        device=device,
                        dtype=dtype,
                    )
                    self._save_initial_latent(latent_path, latents)
            else:
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

    def _load_initial_latent(
        self,
        latent_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        path = Path(latent_path)
        if not path.exists():
            return None
        payload = torch.load(path, map_location=device)
        if isinstance(payload, dict):
            if "video_latent" in payload:
                latent = payload["video_latent"]
            elif "latent" in payload:
                latent = payload["latent"]
            else:
                latent = None
        else:
            latent = payload
        if not torch.is_tensor(latent):
            raise TypeError(f"Expected tensor for initial latent in {path}")
        logger.info("[LTX2] Loaded initial latent from %s", path)
        return latent.to(device=device, dtype=dtype)

    def _save_initial_latent(self, latent_path: str,
                             latents: torch.Tensor) -> None:
        path = Path(latent_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        torch.save({"video_latent": latents.detach().cpu()}, path)
        logger.info("[LTX2] Saved initial latent to %s", path)

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
