# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, TYPE_CHECKING

import torch

from fastvideo.attention.selector import (
    coerce_attn_backend,
    global_force_attn_backend_context_manager,
)
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.fastvideo_args import ExecutionMode, TrainingArgs
from fastvideo.models.loader.component_loader import (
    PipelineComponentLoader, )
from fastvideo.utils import (
    maybe_download_model,
    verify_model_config_and_directory,
)
from fastvideo.platforms import AttentionBackendEnum

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )

# ------------------------------------------------------------------
# TrainingArgs builders (only place that creates FastVideoArgs)
# ------------------------------------------------------------------


def _make_training_args(
    tc: TrainingConfig,
    *,
    model_path: str,
) -> TrainingArgs:
    """Build a TrainingArgs for PipelineComponentLoader."""
    pipeline_config = tc.pipeline_config or PipelineConfig()
    # Propagate dit_precision from TrainingConfig to PipelineConfig
    # so that TransformerLoader.load() picks up the correct
    # default_dtype (e.g. fp32 master weights for training).
    if tc.dit_precision and tc.dit_precision != pipeline_config.dit_precision:
        pipeline_config.dit_precision = tc.dit_precision
    return TrainingArgs(
        model_path=model_path,
        mode=ExecutionMode.DISTILLATION,
        inference_mode=False,
        pipeline_config=pipeline_config,
        num_gpus=tc.distributed.num_gpus,
        tp_size=tc.distributed.tp_size,
        sp_size=tc.distributed.sp_size,
        hsdp_replicate_dim=tc.distributed.hsdp_replicate_dim,
        hsdp_shard_dim=tc.distributed.hsdp_shard_dim,
        pin_cpu_memory=tc.distributed.pin_cpu_memory,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        use_fsdp_inference=False,
        enable_torch_compile=tc.model.enable_torch_compile,
    )


def make_inference_args(
    tc: TrainingConfig,
    *,
    model_path: str,
) -> TrainingArgs:
    """Build a TrainingArgs for inference (validation / pipelines)."""
    args = _make_training_args(tc, model_path=model_path)
    args.inference_mode = True
    args.mode = ExecutionMode.INFERENCE
    args.dit_cpu_offload = True
    args.VSA_sparsity = tc.vsa_sparsity
    return args


# ------------------------------------------------------------------
# Module loading
# ------------------------------------------------------------------


def load_module_from_path(
    *,
    model_path: str,
    module_type: str,
    training_config: TrainingConfig,
    disable_custom_init_weights: bool = False,
    override_transformer_cls_name: str | None = None,
    transformer_override_safetensor: str | None = None,
    attention_backend: AttentionBackendEnum | str | None = None,
) -> torch.nn.Module:
    """Load a single pipeline component module.

    Accepts a ``TrainingConfig`` and internally builds the
    ``TrainingArgs`` needed by ``PipelineComponentLoader``.
    """
    fastvideo_args: Any = _make_training_args(training_config, model_path=model_path)

    local_model_path = maybe_download_model(model_path)
    config = verify_model_config_and_directory(local_model_path)

    if module_type not in config:
        raise ValueError(f"Module {module_type!r} not found in "
                         f"config at {local_model_path}")

    module_info = config[module_type]
    if module_info is None:
        raise ValueError(f"Module {module_type!r} has null value in "
                         f"config at {local_model_path}")

    transformers_or_diffusers, _architecture = module_info
    component_path = os.path.join(local_model_path, module_type)

    old_override: str | None = None
    if override_transformer_cls_name is not None:
        old_override = getattr(
            fastvideo_args,
            "override_transformer_cls_name",
            None,
        )
        fastvideo_args.override_transformer_cls_name = str(override_transformer_cls_name)

    if transformer_override_safetensor:
        fastvideo_args.init_weights_from_safetensors = str(transformer_override_safetensor)

    if attention_backend is not None and module_type != "transformer":
        raise ValueError("attention_backend can only be set when loading "
                         f"a transformer, got module_type={module_type!r}")
    resolved_attention_backend = coerce_attn_backend(attention_backend)
    attention_context = (nullcontext() if resolved_attention_backend is None else
                         global_force_attn_backend_context_manager(resolved_attention_backend))

    if disable_custom_init_weights:
        fastvideo_args._loading_teacher_critic_model = True
    try:
        # Attention implementations are bound while transformer layers are
        # constructed. Scope the override to this one role so student,
        # teacher, and critic can use independent backends in one process.
        with attention_context:
            module = PipelineComponentLoader.load_module(
                module_name=module_type,
                component_model_path=component_path,
                transformers_or_diffusers=(transformers_or_diffusers),
                fastvideo_args=fastvideo_args,
            )
    finally:
        if disable_custom_init_weights and hasattr(fastvideo_args, "_loading_teacher_critic_model"):
            del fastvideo_args._loading_teacher_critic_model
        if override_transformer_cls_name is not None:
            if old_override is None:
                if hasattr(
                        fastvideo_args,
                        "override_transformer_cls_name",
                ):
                    fastvideo_args.override_transformer_cls_name = (None)
            else:
                fastvideo_args.override_transformer_cls_name = (old_override)

    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Loaded {module_type!r} is not a "
                        f"torch.nn.Module: {type(module)}")
    if module_type == "transformer":
        reshard_after_forward = training_config.distributed.reshard_after_forward
        set_reshard_after_forward = getattr(
            module,
            "set_reshard_after_forward",
            None,
        )
        if callable(set_reshard_after_forward):
            set_reshard_after_forward(
                reshard_after_forward,
                recurse=True,
            )
        elif not reshard_after_forward:
            raise RuntimeError("training.distributed.reshard_after_forward=false requires "
                               "an FSDP-wrapped transformer")
    return module
