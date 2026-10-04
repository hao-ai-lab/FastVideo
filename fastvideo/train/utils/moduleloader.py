# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, TYPE_CHECKING

import torch

from fastvideo.api.inference_resolution import (
    generator_resolution_steps,
    resolve_config,
    resolve_inference_config,
)
from fastvideo.api.resolution import ResolutionStep, ResolvedGeneratorConfig
from fastvideo.api.schema import ExecutionMode, GeneratorConfig
from fastvideo.api.training_schema import validate_training_parallel_sizes
from fastvideo.attention.selector import (
    _component_attention_backend_scope,
    coerce_attn_backend,
)
from fastvideo.configs.pipelines.base import PipelineConfig
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
# Resolved configs (the only place that builds them from a TrainingConfig)
# ------------------------------------------------------------------


def _generator_config_mapping(
    tc: TrainingConfig,
    *,
    model_path: str,
    mode: ExecutionMode,
    pipeline_config: PipelineConfig,
) -> dict[str, Any]:
    """``GeneratorConfig`` mapping with the training parallel layout and every offload and compile switch off.

    ``pipeline_config`` is the model definition that resolution copies and
    materializes. ``tc.dit_precision`` sets the DiT precision, so
    ``TransformerLoader`` builds the master weights in it (fp32 for training).
    """
    return {
        "model_path": model_path,
        "mode": mode,
        "engine": {
            "num_gpus": tc.distributed.num_gpus,
            "parallelism": {
                "tp_size": tc.distributed.tp_size,
                "sp_size": tc.distributed.sp_size,
                "hsdp_replicate_dim": tc.distributed.hsdp_replicate_dim,
                "hsdp_shard_dim": tc.distributed.hsdp_shard_dim,
            },
            "offload": {
                "dit": False,
                "dit_layerwise": False,
                "text_encoder": False,
                "image_encoder": False,
                "vae": False,
                "pin_cpu_memory": tc.distributed.pin_cpu_memory,
            },
            "use_fsdp_inference": False,
            "compile": {
                "enabled": False
            },
            "precision": {
                "dit": tc.dit_precision
            },
        },
        "pipeline": {
            "experimental": {
                "pipeline_config": pipeline_config
            }
        },
    }


def _training_resolution_steps(config: GeneratorConfig, defaults: Any) -> tuple[ResolutionStep, ...]:
    """Generator resolution steps plus the training parallel-size check."""
    return generator_resolution_steps(
        config,
        defaults,
        before_placeholders=(validate_training_parallel_sizes, ),
    )


def _build_training_resolved_config(
    tc: TrainingConfig,
    *,
    model_path: str,
    override_transformer_cls_name: str | None = None,
    transformer_weights: str | None = None,
) -> ResolvedGeneratorConfig:
    """Build the distillation-mode resolved config that ``PipelineComponentLoader`` reads.

    ``override_transformer_cls_name`` and ``transformer_weights`` set
    ``pipeline.components.override_transformer_cls_name`` and
    ``pipeline.components.transformer_weights`` when they are given.
    """
    raw = _generator_config_mapping(
        tc,
        model_path=model_path,
        mode=ExecutionMode.DISTILLATION,
        pipeline_config=tc.pipeline_config if tc.pipeline_config is not None else PipelineConfig(),
    )
    components: dict[str, str] = {}
    if override_transformer_cls_name is not None:
        components["override_transformer_cls_name"] = str(override_transformer_cls_name)
    if transformer_weights:
        components["transformer_weights"] = str(transformer_weights)
    if components:
        raw["pipeline"]["components"] = components
    return resolve_config(raw, GeneratorConfig, _training_resolution_steps)


def build_inference_resolved_config(
    tc: TrainingConfig,
    *,
    model_path: str,
    pipeline_config: PipelineConfig | None = None,
    dmd_denoising_steps: list[int] | None = None,
) -> ResolvedGeneratorConfig:
    """Build the inference-mode resolved config for validation forwards and standalone encoder loads.

    It has the training parallel layout, the DiT offloaded to the CPU, every
    other offload off, and ``tc.vsa_sparsity`` as the VSA sparsity.
    ``pipeline_config`` replaces ``tc.pipeline_config`` as the model
    definition. ``dmd_denoising_steps`` sets ``pipeline.dmd_denoising_steps``,
    the DMD schedule that the causal and DMD denoising stages read.
    """
    if pipeline_config is None:
        pipeline_config = tc.pipeline_config if tc.pipeline_config is not None else PipelineConfig()
    raw = _generator_config_mapping(
        tc,
        model_path=model_path,
        mode=ExecutionMode.INFERENCE,
        pipeline_config=pipeline_config,
    )
    raw["engine"]["offload"]["dit"] = True
    raw["engine"]["attention"] = {"vsa_sparsity": tc.vsa_sparsity}
    if dmd_denoising_steps is not None:
        raw["pipeline"]["dmd_denoising_steps"] = list(dmd_denoising_steps)
    return resolve_inference_config(raw)


def keep_checkpoint_component_config(
    tc: TrainingConfig,
    resolved_config: ResolvedGeneratorConfig,
    component_config_name: str,
) -> None:
    """Point ``tc.pipeline_config.<component_config_name>`` at the component config that a loader filled.

    A resolved config holds a copy of ``tc.pipeline_config``, and the VAE and
    encoder loaders fill that copy's arch configs from the checkpoint's
    ``config.json``. Training code reads component shapes, such as the VAE
    latent channels and compression ratios, from ``tc.pipeline_config``.
    """
    if tc.pipeline_config is not None:
        setattr(tc.pipeline_config, component_config_name,
                getattr(resolved_config.pipeline_config, component_config_name))


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
    """Load one pipeline component with its role-scoped attention policy.

    Accepts a ``TrainingConfig`` and internally builds the distillation-mode
    resolved config that ``PipelineComponentLoader`` reads.

    Diffusers component entries retain provider and architecture as their
    first two fields and can append modular loading metadata. Attention layers
    bind their backend during construction, so the requested backend remains
    scoped to this load call.
    """
    resolved_config = _build_training_resolved_config(
        training_config,
        model_path=model_path,
        override_transformer_cls_name=override_transformer_cls_name,
        transformer_weights=transformer_override_safetensor,
    )

    local_model_path = maybe_download_model(model_path)
    config = verify_model_config_and_directory(local_model_path)

    if module_type not in config:
        raise ValueError(f"Module {module_type!r} not found in "
                         f"config at {local_model_path}")

    module_info = config[module_type]
    if module_info is None:
        raise ValueError(f"Module {module_type!r} has null value in "
                         f"config at {local_model_path}")

    # Trailing modular-manifest metadata does not change component dispatch;
    # the provider and architecture remain the first two fields.
    transformers_or_diffusers, _architecture = module_info[:2]
    component_path = os.path.join(local_model_path, module_type)

    if attention_backend is not None and module_type != "transformer":
        raise ValueError("attention_backend can only be set when loading "
                         f"a transformer, got module_type={module_type!r}")
    resolved_attention_backend = coerce_attn_backend(attention_backend)
    # Per-role request delivered as a construction scope: process-local,
    # exception-safe, and part of the selector's cache key (no global
    # mutation, no cache flushes between roles).
    attention_context = (nullcontext() if resolved_attention_backend is None else _component_attention_backend_scope(
        resolved_attention_backend, component=module_type))

    # Attention implementations are bound while transformer layers are
    # constructed. Scope the override to this one role so student,
    # teacher, and critic can use independent backends in one process.
    with attention_context:
        module = PipelineComponentLoader.load_module(
            module_name=module_type,
            component_model_path=component_path,
            transformers_or_diffusers=(transformers_or_diffusers),
            resolved_config=resolved_config,
            loading_teacher_critic_model=disable_custom_init_weights,
        )
    if module_type == "vae":
        keep_checkpoint_component_config(training_config, resolved_config, "vae_config")

    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Loaded {module_type!r} is not a "
                        f"torch.nn.Module: {type(module)}")
    return module
