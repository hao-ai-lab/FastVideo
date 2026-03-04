# SPDX-License-Identifier: Apache-2.0
"""Minimal FastVideoArgs subclass for PipelineComponentLoader."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.fastvideo_args import ExecutionMode, FastVideoArgs

if TYPE_CHECKING:
    from fastvideo.distillation.utils.distill_config import (
        DistillTrainingConfig, )


@dataclasses.dataclass
class DistillLoaderArgs(FastVideoArgs):
    """Minimal FastVideoArgs for PipelineComponentLoader."""

    @classmethod
    def from_training_config(
        cls,
        tc: DistillTrainingConfig,
        *,
        model_path: str,
    ) -> DistillLoaderArgs:
        return cls(
            model_path=model_path,
            mode=ExecutionMode.DISTILLATION,
            inference_mode=False,
            pipeline_config=(tc.pipeline_config or PipelineConfig()),
            num_gpus=tc.distributed.num_gpus,
            tp_size=tc.distributed.tp_size,
            sp_size=tc.distributed.sp_size,
            hsdp_replicate_dim=(tc.distributed.hsdp_replicate_dim),
            hsdp_shard_dim=tc.distributed.hsdp_shard_dim,
            pin_cpu_memory=tc.distributed.pin_cpu_memory,
            dit_cpu_offload=False,
            dit_layerwise_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=False,
            image_encoder_cpu_offload=False,
            use_fsdp_inference=False,
            enable_torch_compile=False,
        )

    @classmethod
    def for_inference(
        cls,
        tc: DistillTrainingConfig,
        *,
        model_path: str,
    ) -> DistillLoaderArgs:
        args = cls.from_training_config(tc, model_path=model_path)
        args.inference_mode = True
        args.mode = ExecutionMode.INFERENCE
        args.dit_cpu_offload = True
        return args
