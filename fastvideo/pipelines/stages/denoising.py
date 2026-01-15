# SPDX-License-Identifier: Apache-2.0
"""
Denoising stage for diffusion pipelines.
"""

import inspect
import weakref
from collections.abc import Iterable
from typing import Any

import torch
from tqdm.auto import tqdm

from fastvideo.attention import get_attn_backend
from fastvideo.configs.pipelines.base import STA_Mode
from fastvideo.distributed import get_world_group
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.platforms import AttentionBackendEnum

logger = init_logger(__name__)


class DenoisingStage(PipelineStage):
    """
    Stage for running the denoising loop in diffusion pipelines.
    
    This stage handles the iterative denoising process that transforms
    the initial noise into the final output.
    """

    def __init__(self,
                 transformer,
                 scheduler,
                 pipeline=None,
                 transformer_2=None,
                 vae=None,
                 strategy_cls=None) -> None:
        super().__init__()
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.scheduler = scheduler
        self.vae = vae
        self.pipeline = weakref.ref(pipeline) if pipeline else None
        self.strategy_cls = strategy_cls
        attn_head_size = self.transformer.hidden_size // self.transformer.num_attention_heads
        self.attn_backend = get_attn_backend(
            head_size=attn_head_size,
            dtype=torch.float16,  # TODO(will): hack
            supported_attention_backends=(
                AttentionBackendEnum.SLIDING_TILE_ATTN,
                AttentionBackendEnum.VIDEO_SPARSE_ATTN,
                AttentionBackendEnum.VMOBA_ATTN,
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
                AttentionBackendEnum.SAGE_ATTN_THREE)  # hack
        )

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Run the denoising loop.
        """
        from fastvideo.pipelines.stages.denoising_engine import DenoisingEngine
        from fastvideo.pipelines.stages.denoising_standard_strategy import (
            StandardStrategy)

        strategy_cls = self.strategy_cls or StandardStrategy
        engine = DenoisingEngine(strategy_cls(self))
        return engine.run(batch, fastvideo_args)

    def prepare_extra_func_kwargs(self, func, kwargs) -> dict[str, Any]:
        """
        Prepare extra kwargs for the scheduler step / denoise step.
        
        Args:
            func: The function to prepare kwargs for.
            kwargs: The kwargs to prepare.
            
        Returns:
            The prepared kwargs.
        """
        extra_step_kwargs = {}
        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs

    def progress_bar(self,
                     iterable: Iterable | None = None,
                     total: int | None = None) -> tqdm:
        """
        Create a progress bar for the denoising process.
        
        Args:
            iterable: The iterable to iterate over.
            total: The total number of items.
            
        Returns:
            A tqdm progress bar.
        """
        local_rank = get_world_group().local_rank
        if local_rank == 0:
            return tqdm(iterable=iterable, total=total)
        else:
            return tqdm(iterable=iterable, total=total, disable=True)

    def rescale_noise_cfg(self,
                          noise_cfg,
                          noise_pred_text,
                          guidance_rescale=0.0) -> torch.Tensor:
        """
        Rescale noise prediction according to guidance_rescale.
        
        Based on findings of "Common Diffusion Noise Schedules and Sample Steps are Flawed"
        (https://arxiv.org/pdf/2305.08891.pdf), Section 3.4.
        
        Args:
            noise_cfg: The noise prediction with guidance.
            noise_pred_text: The text-conditioned noise prediction.
            guidance_rescale: The guidance rescale factor.
            
        Returns:
            The rescaled noise prediction.
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)),
                                       keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)),
                                keepdim=True)
        # Rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # Mix with the original results from guidance by factor guidance_rescale
        noise_cfg = (guidance_rescale * noise_pred_rescaled +
                     (1 - guidance_rescale) * noise_cfg)
        return noise_cfg

    def prepare_sta_param(self, batch: ForwardBatch,
                          fastvideo_args: FastVideoArgs):
        """
        Prepare Sliding Tile Attention (STA) parameters and settings.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
        """
        # TODO(kevin): STA mask search, currently only support Wan2.1 with 69x768x1280
        from fastvideo.attention.backends.STA_configuration import configure_sta
        STA_mode = fastvideo_args.STA_mode
        skip_time_steps = fastvideo_args.skip_time_steps
        if batch.timesteps is None:
            raise ValueError("Timesteps must be provided")
        timesteps_num = batch.timesteps.shape[0]

        logger.info("STA_mode: %s", STA_mode)
        if (batch.num_frames, batch.height,
                batch.width) != (69, 768, 1280) and STA_mode != "STA_inference":
            raise NotImplementedError(
                "STA mask search/tuning is not supported for this resolution")

        if STA_mode == STA_Mode.STA_SEARCHING or STA_mode == STA_Mode.STA_TUNING or STA_mode == STA_Mode.STA_TUNING_CFG:
            size = (batch.width, batch.height)
            if size == (1280, 768):
                # TODO: make it configurable
                sparse_mask_candidates_searching = [
                    "3, 1, 10", "1, 5, 7", "3, 3, 3", "1, 6, 5", "1, 3, 10",
                    "3, 6, 1"
                ]
                sparse_mask_candidates_tuning = [
                    "3, 1, 10", "1, 5, 7", "3, 3, 3", "1, 6, 5", "1, 3, 10",
                    "3, 6, 1"
                ]
                full_mask = ["3,6,10"]
            else:
                raise NotImplementedError(
                    "STA mask search is not supported for this resolution")
        layer_num = self.transformer.config.num_layers
        # specific for HunyuanVideo
        if hasattr(self.transformer.config, "num_single_layers"):
            layer_num += self.transformer.config.num_single_layers
        head_num = self.transformer.config.num_attention_heads

        if STA_mode == STA_Mode.STA_SEARCHING:
            STA_param = configure_sta(
                mode=STA_Mode.STA_SEARCHING,
                layer_num=layer_num,
                head_num=head_num,
                time_step_num=timesteps_num,
                mask_candidates=sparse_mask_candidates_searching +
                full_mask,  # last is full mask; Can add more sparse masks while keep last one as full mask
            )
        elif STA_mode == STA_Mode.STA_TUNING:
            STA_param = configure_sta(
                mode=STA_Mode.STA_TUNING,
                layer_num=layer_num,
                head_num=head_num,
                time_step_num=timesteps_num,
                mask_search_files_path=
                f'output/mask_search_result_pos_{size[0]}x{size[1]}/',
                mask_candidates=sparse_mask_candidates_tuning,
                full_attention_mask=[int(x) for x in full_mask[0].split(',')],
                skip_time_steps=
                skip_time_steps,  # Use full attention for first 12 steps
                save_dir=
                f'output/mask_search_strategy_{size[0]}x{size[1]}/',  # Custom save directory
                timesteps=timesteps_num)
        elif STA_mode == STA_Mode.STA_TUNING_CFG:
            STA_param = configure_sta(
                mode=STA_Mode.STA_TUNING_CFG,
                layer_num=layer_num,
                head_num=head_num,
                time_step_num=timesteps_num,
                mask_search_files_path_pos=
                f'output/mask_search_result_pos_{size[0]}x{size[1]}/',
                mask_search_files_path_neg=
                f'output/mask_search_result_neg_{size[0]}x{size[1]}/',
                mask_candidates=sparse_mask_candidates_tuning,
                full_attention_mask=[int(x) for x in full_mask[0].split(',')],
                skip_time_steps=skip_time_steps,
                save_dir=f'output/mask_search_strategy_{size[0]}x{size[1]}/',
                timesteps=timesteps_num)
        elif STA_mode == STA_Mode.STA_INFERENCE:
            import fastvideo.envs as envs
            config_file = envs.FASTVIDEO_ATTENTION_CONFIG
            if config_file is None:
                raise ValueError("FASTVIDEO_ATTENTION_CONFIG is not set")
            STA_param = configure_sta(mode=STA_Mode.STA_INFERENCE,
                                      layer_num=layer_num,
                                      head_num=head_num,
                                      time_step_num=timesteps_num,
                                      load_path=config_file)

        batch.STA_param = STA_param
        batch.mask_search_final_result_pos = [[] for _ in range(timesteps_num)]
        batch.mask_search_final_result_neg = [[] for _ in range(timesteps_num)]

    def save_sta_search_results(self, batch: ForwardBatch):
        """
        Save the STA mask search results.
        
        Args:
            batch: The current batch information.
        """
        size = (batch.width, batch.height)
        if size == (1280, 768):
            # TODO: make it configurable
            sparse_mask_candidates_searching = [
                "3, 1, 10", "1, 5, 7", "3, 3, 3", "1, 6, 5", "1, 3, 10",
                "3, 6, 1"
            ]
        else:
            raise NotImplementedError(
                "STA mask search is not supported for this resolution")

        from fastvideo.attention.backends.STA_configuration import save_mask_search_results
        if batch.mask_search_final_result_pos is not None and batch.prompt is not None:
            save_mask_search_results(
                [
                    dict(layer_data)
                    for layer_data in batch.mask_search_final_result_pos
                ],
                prompt=str(batch.prompt),
                mask_strategies=sparse_mask_candidates_searching,
                output_dir=f'output/mask_search_result_pos_{size[0]}x{size[1]}/'
            )
        if batch.mask_search_final_result_neg is not None and batch.prompt is not None:
            save_mask_search_results(
                [
                    dict(layer_data)
                    for layer_data in batch.mask_search_final_result_neg
                ],
                prompt=str(batch.prompt),
                mask_strategies=sparse_mask_candidates_searching,
                output_dir=f'output/mask_search_result_neg_{size[0]}x{size[1]}/'
            )

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify denoising stage inputs."""
        result = VerificationResult()
        if batch.timesteps is not None:
            result.add_check("timesteps", batch.timesteps,
                             [V.is_tensor, V.min_dims(1)])
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        result.add_check("image_latent", batch.image_latent,
                         V.none_or_tensor_with_dims(5))
        result.add_check("num_inference_steps", batch.num_inference_steps,
                         V.positive_int)
        result.add_check("guidance_scale", batch.guidance_scale,
                         V.positive_float)
        result.add_check("eta", batch.eta, V.non_negative_float)
        result.add_check("generator", batch.generator,
                         V.generator_or_list_generators)
        result.add_check("do_classifier_free_guidance",
                         batch.do_classifier_free_guidance, V.bool_value)
        result.add_check(
            "negative_prompt_embeds", batch.negative_prompt_embeds, lambda x:
            not batch.do_classifier_free_guidance or V.list_not_empty(x))
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify denoising stage outputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        return result
