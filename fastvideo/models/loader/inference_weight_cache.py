# SPDX-License-Identifier: Apache-2.0
"""Opt-in native Wan loader adapters for persistent CUDA IPC weights."""
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fastvideo.models.loader.weight_cache import WeightCacheClient, WeightCacheKey

COMPONENTS = ("transformer", "text_encoder", "vae")
_CLASSES = {"transformer": "WanTransformer3DModel", "text_encoder": "UMT5EncoderModel",
            "vae": "AutoencoderKLWan"}


def validate_weight_cache_args(args) -> None:
    """Reject modes that can mutate, relocate or reinterpret shared storage."""
    if not args.inference_mode or str(args.mode) not in ("inference", "ExecutionMode.INFERENCE"):
        raise ValueError("Weight reuse supports inference only")
    if type(args.pipeline_config).__name__ != "WanT2V480PConfig":
        raise ValueError("Weight reuse currently supports the native Wan 2.1 T2V 1.3B pipeline only")
    if any(getattr(args, name) != 1 for name in ("num_gpus", "tp_size", "sp_size",
                                                "hsdp_replicate_dim", "hsdp_shard_dim")):
        raise ValueError("Weight reuse requires one GPU and single-rank parallelism")
    unsupported = ("use_fsdp_inference", "dit_cpu_offload", "dit_layerwise_offload", "vae_cpu_offload",
                   "text_encoder_cpu_offload", "lazy_module_load", "enable_torch_compile",
                   "enable_torch_compile_text_encoder", "enable_torch_compile_vae", "inference_torch_compile",
                   "lora_path", "init_weights_from_safetensors", "init_weights_from_safetensors_2",
                   "override_text_encoder_safetensors", "override_text_encoder_quant",
                   "override_transformer_cls_name", "override_transformer_safetensors")
    enabled = [name for name in unsupported if getattr(args, name, None)]
    if enabled:
        raise ValueError(f"Weight reuse does not support: {', '.join(enabled)}")
    configs = [args.pipeline_config.dit_config, *args.pipeline_config.text_encoder_configs]
    if any(getattr(config, "quant_config", None) is not None for config in configs):
        raise ValueError("Weight reuse does not support quantized components")
    if args.pipeline_config.vae_config.use_light_vae:
        raise ValueError("Weight reuse does not support light-VAE memory-format conversion")
    if not args.weight_cache_checkpoint:
        raise ValueError("Weight reuse requires an immutable weight_cache_checkpoint identity")


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, torch.Tensor):
        if value.is_meta:
            raise ValueError("Configuration constants must be initialized outside meta construction")
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.cpu().tolist()}
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    if callable(value):
        return value.__module__ + "." + value.__qualname__
    raise TypeError(f"Unsupported weight configuration identity: {type(value)}")


@dataclass
class _LoadContext:
    component: str
    args: Any
    leases: ExitStack | None
    key: WeightCacheKey | None = None
    witness: dict | None = None


_CURRENT: ContextVar[_LoadContext | None] = ContextVar("inference_weight_cache_load", default=None)


@contextmanager
def component_weight_cache(component, args, leases):
    """The normal loader owns dispatch; the pipeline owns attachment lifetimes.

    A checkpoint identity without a cache directory records donor keys while
    retaining normal disk hydration. With a directory, attachment is mandatory.
    """
    if not (getattr(args, "weight_cache_dir", None) or getattr(args, "weight_cache_checkpoint", None)):
        yield None
        return
    validate_weight_cache_args(args)
    if component not in (*COMPONENTS, "scheduler", "tokenizer"):
        raise ValueError(f"Unsupported weight-cache pipeline component: {component}")
    if component not in COMPONENTS:
        yield None
        return
    if args.weight_cache_dir and leases is None:
        raise ValueError("IPC loading requires a caller-owned weight_cache_leases ExitStack")
    context = _LoadContext(component, args, leases)
    token = _CURRENT.set(context)
    try:
        yield context
        if context.key is None:
            raise ValueError(f"No native weight-cache adapter for {component}")
    finally:
        _CURRENT.reset(token)


def validate_cached_class(class_name):
    context = _CURRENT.get()
    if context is not None and class_name != _CLASSES[context.component]:
        raise ValueError(f"Unsupported cached {context.component} class: {class_name}")


def attach_cached_model(model_cls, config, dtype, factory: Callable[[], nn.Module], *, extra_config=None):
    """Called after config resolution and before any checkpoint hydration."""
    context = _CURRENT.get()
    if context is None:
        return None
    validate_cached_class(model_cls.__name__)
    if getattr(config, "quant_config", None) is not None:
        raise ValueError("Weight reuse does not support checkpoint-selected quantization")
    identity = {"class": model_cls.__module__ + "." + model_cls.__qualname__,
                "config": config, "extra": extra_config}
    encoded = json.dumps(identity, default=_json_default, sort_keys=True).encode()
    context.key = WeightCacheKey(context.args.weight_cache_checkpoint, context.component,
                                hashlib.sha256(encoded).hexdigest(), str(dtype), "native-wan-loader-v1")
    if not context.args.weight_cache_dir:
        return None
    client = WeightCacheClient(Path(context.args.weight_cache_dir) / f"{context.component}.sock", context.key)
    attached = context.leases.enter_context(client.load_model(factory))
    context.witness = {"donor_pid": attached.donor_pid, "handles": attached.handles}
    return attached.model
