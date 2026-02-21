# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import gc
from typing import Any

import torch

from fastvideo.configs.sample import SamplingParam
from fastvideo.distributed import get_local_torch_device, get_world_group
from fastvideo.pipelines import TrainingBatch
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.basic.wan.wan_dmd_pipeline import WanDMDPipeline
from fastvideo.training.distillation_pipeline import DistillationPipeline

from fastvideo.distillation.adapters.base import DistillAdapter


class WanPipelineAdapter(DistillAdapter):

    def __init__(self, pipeline: DistillationPipeline) -> None:
        self.pipeline = pipeline

    def _get_training_dtype(self) -> torch.dtype:
        # Phase 0: match existing training pipelines (autocast bf16).
        return torch.bfloat16

    def ensure_negative_conditioning(self) -> None:
        if getattr(self.pipeline, "negative_prompt_embeds", None) is not None:
            return

        training_args = self.pipeline.training_args
        world_group = get_world_group()
        device = get_local_torch_device()
        dtype = self._get_training_dtype()

        neg_embeds: torch.Tensor | None = None
        neg_mask: torch.Tensor | None = None

        if world_group.rank_in_group == 0:
            sampling_param = SamplingParam.from_pretrained(
                training_args.model_path)
            negative_prompt = sampling_param.negative_prompt

            prompt_pipeline = getattr(self.pipeline, "validation_pipeline",
                                      None)
            created_pipeline = False
            if prompt_pipeline is None:
                args_copy = copy.deepcopy(training_args)
                args_copy.inference_mode = True
                prompt_pipeline = WanDMDPipeline.from_pretrained(
                    training_args.model_path,
                    args=args_copy,
                    inference_mode=True,
                    loaded_modules={
                        "transformer": self.pipeline.get_module("transformer")
                    },
                    tp_size=training_args.tp_size,
                    sp_size=training_args.sp_size,
                    num_gpus=training_args.num_gpus,
                    pin_cpu_memory=training_args.pin_cpu_memory,
                    dit_cpu_offload=True,
                )
                created_pipeline = True

            batch_negative = ForwardBatch(
                data_type="video",
                prompt=negative_prompt,
                prompt_embeds=[],
                prompt_attention_mask=[],
            )
            result_batch = prompt_pipeline.prompt_encoding_stage(  # type: ignore[attr-defined]
                batch_negative,
                training_args,
            )

            neg_embeds = result_batch.prompt_embeds[0].to(device=device,
                                                          dtype=dtype)
            neg_mask = result_batch.prompt_attention_mask[0].to(device=device,
                                                                dtype=dtype)

            if created_pipeline:
                del prompt_pipeline
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        meta = torch.zeros((2, ), device=device, dtype=torch.int64)
        if world_group.rank_in_group == 0:
            assert neg_embeds is not None
            assert neg_mask is not None
            meta[0] = neg_embeds.ndim
            meta[1] = neg_mask.ndim
        world_group.broadcast(meta, src=0)
        embed_ndim, mask_ndim = (int(meta[0].item()), int(meta[1].item()))

        max_ndim = 8
        embed_shape = torch.full((max_ndim, ),
                                 -1,
                                 device=device,
                                 dtype=torch.int64)
        mask_shape = torch.full((max_ndim, ),
                                -1,
                                device=device,
                                dtype=torch.int64)
        if world_group.rank_in_group == 0:
            assert neg_embeds is not None
            assert neg_mask is not None
            embed_shape[:embed_ndim] = torch.tensor(list(neg_embeds.shape),
                                                    device=device,
                                                    dtype=torch.int64)
            mask_shape[:mask_ndim] = torch.tensor(list(neg_mask.shape),
                                                  device=device,
                                                  dtype=torch.int64)
        world_group.broadcast(embed_shape, src=0)
        world_group.broadcast(mask_shape, src=0)

        embed_sizes = tuple(int(x) for x in embed_shape[:embed_ndim].tolist())
        mask_sizes = tuple(int(x) for x in mask_shape[:mask_ndim].tolist())

        if world_group.rank_in_group != 0:
            neg_embeds = torch.empty(embed_sizes, device=device, dtype=dtype)
            neg_mask = torch.empty(mask_sizes, device=device, dtype=dtype)
        assert neg_embeds is not None
        assert neg_mask is not None

        world_group.broadcast(neg_embeds, src=0)
        world_group.broadcast(neg_mask, src=0)

        self.pipeline.negative_prompt_embeds = neg_embeds
        self.pipeline.negative_prompt_attention_mask = neg_mask

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        current_vsa_sparsity: float = 0.0,
    ) -> TrainingBatch:
        self.ensure_negative_conditioning()

        device = get_local_torch_device()
        dtype = self._get_training_dtype()

        training_batch = TrainingBatch(
            current_vsa_sparsity=current_vsa_sparsity)
        encoder_hidden_states = raw_batch["text_embedding"]
        encoder_attention_mask = raw_batch["text_attention_mask"]
        infos = raw_batch.get("info_list")

        if self.pipeline.training_args.simulate_generator_forward:
            batch_size = encoder_hidden_states.shape[0]
            vae_config = self.pipeline.training_args.pipeline_config.vae_config.arch_config
            num_channels = vae_config.z_dim
            spatial_compression_ratio = vae_config.spatial_compression_ratio
            latent_height = self.pipeline.training_args.num_height // spatial_compression_ratio
            latent_width = self.pipeline.training_args.num_width // spatial_compression_ratio
            latents = torch.zeros(
                batch_size,
                num_channels,
                self.pipeline.training_args.num_latent_t,
                latent_height,
                latent_width,
                device=device,
                dtype=dtype,
            )
        else:
            if "vae_latent" not in raw_batch:
                raise ValueError(
                    "vae_latent not found in batch and simulate_generator_forward is False"
                )
            latents = raw_batch["vae_latent"]
            latents = latents[:, :, :self.pipeline.training_args.num_latent_t]
            latents = latents.to(device, dtype=dtype)

        training_batch.latents = latents
        training_batch.encoder_hidden_states = encoder_hidden_states.to(
            device, dtype=dtype)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(
            device, dtype=dtype)
        training_batch.infos = infos

        training_batch = self.pipeline._normalize_dit_input(training_batch)
        training_batch = self.pipeline._prepare_dit_inputs(training_batch)
        training_batch = self.pipeline._build_attention_metadata(training_batch)

        training_batch.attn_metadata_vsa = copy.deepcopy(
            training_batch.attn_metadata)
        if training_batch.attn_metadata is not None:
            training_batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore[attr-defined]

        return training_batch
