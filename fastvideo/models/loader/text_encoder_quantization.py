# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-serialized quantization lifecycle for native text encoders."""

import json
import os
from itertools import chain
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import safe_open

from fastvideo.configs.models import EncoderConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.layers.linear import LinearBase, UnquantizedLinearMethod
from fastvideo.layers.quantization.base_config import QuantizationConfig
from fastvideo.models.encoders.base import TextEncoder


def _resolve_text_encoder_checkpoint_path(
    model_path: str,
    fastvideo_args: FastVideoArgs,
    use_text_encoder_override: bool,
) -> str:
    override = fastvideo_args.override_text_encoder_safetensors if use_text_encoder_override else None
    checkpoint_path = override or model_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Text-encoder checkpoint does not exist: {checkpoint_path}")
    if not os.path.isdir(checkpoint_path) and not os.path.isfile(checkpoint_path):
        raise ValueError(f"Text-encoder checkpoint must be a file or directory: {checkpoint_path}")
    return checkpoint_path


def _read_text_encoder_checkpoint_quantization_config(checkpoint_path: str) -> dict[str, Any] | None:
    checkpoint_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as config_file:
                checkpoint_config = json.load(config_file)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid text-encoder checkpoint config: {config_path}") from error
        quantization_config = checkpoint_config.get("quantization_config")
        if quantization_config is not None:
            if not isinstance(quantization_config, dict):
                raise ValueError(f"quantization_config in {config_path} must be an object")
            return quantization_config

    if not os.path.isfile(checkpoint_path) or not checkpoint_path.endswith(".safetensors"):
        return None
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint_file:
        metadata = checkpoint_file.metadata() or {}
    for key in ("quantization_config", "_quantization_metadata"):
        serialized = metadata.get(key)
        if serialized is None:
            continue
        try:
            quantization_config = json.loads(serialized)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid {key} metadata in {checkpoint_path}") from error
        if not isinstance(quantization_config, dict):
            raise ValueError(f"{key} metadata in {checkpoint_path} must decode to an object")
        return quantization_config
    return None


def _configure_text_encoder_quantization(
    model_config: EncoderConfig,
    model_cls: type[nn.Module],
    checkpoint_path: str,
) -> QuantizationConfig | None:
    if not issubclass(model_cls, TextEncoder):
        return None
    checkpoint_quantization = _read_text_encoder_checkpoint_quantization_config(checkpoint_path)
    if checkpoint_quantization is None:
        return None

    quant_method = str(checkpoint_quantization.get("quant_method", "")).lower()
    if not quant_method:
        raise ValueError(f"Quantized text-encoder checkpoint {checkpoint_path} does not declare quant_method")
    supported_methods = getattr(model_cls, "supported_checkpoint_quantization_methods", frozenset())
    if quant_method not in supported_methods:
        supported = ", ".join(sorted(supported_methods)) or "none"
        raise ValueError(f"Text encoder {model_cls.__name__} does not support serialized {quant_method!r} "
                         f"checkpoints (supported: {supported})")

    factory = getattr(model_cls, "checkpoint_quantization_config_from_metadata", None)
    if not callable(factory):
        raise ValueError(f"Text encoder {model_cls.__name__} advertises serialized {quant_method!r} support "
                         "without a checkpoint quantization factory")
    quant_config = factory(checkpoint_quantization)
    model_config.quant_config = quant_config
    return quant_config


def _module_tensor_device(module: nn.Module) -> torch.device | None:
    devices = {
        tensor.device
        for tensor in chain(
            module.parameters(recurse=False),
            module.buffers(recurse=False),
        )
    }
    if len(devices) > 1:
        raise ValueError(f"Quantized text-encoder module {type(module).__name__} spans multiple devices: {devices}")
    return next(iter(devices), None)


def _process_quantized_text_encoder_weights(model: nn.Module, process_device: torch.device) -> int:
    """Run quantized post-load hooks one linear at a time on ``process_device``."""
    processed = 0
    for module in model.modules():
        if not isinstance(module, LinearBase) or isinstance(module.quant_method, UnquantizedLinearMethod):
            continue
        if module.quant_method is None:
            continue
        original_device = _module_tensor_device(module)
        try:
            module.to(process_device)
            module.quant_method.process_weights_after_loading(module)
        finally:
            if original_device is not None:
                module.to(original_device)
        processed += 1
    if processed == 0:
        raise ValueError("Serialized quantized text-encoder checkpoint selected, but no quantized linear layers exist")
    return processed
