# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Sequence
from logging import Logger
from typing import TypeVar

import torch

DEVICE_REFERENCE_FOLDER_SUFFIX = "_reference_videos"
FULL_QUALITY_ENV_VAR = "FASTVIDEO_SSIM_FULL_QUALITY"
DEFAULT_OUTPUT_QUALITY_TIER = "default"
FULL_OUTPUT_QUALITY_TIER = "full_quality"
REFERENCE_VIDEOS_DIRNAME = "reference_videos"
DeviceMapping = tuple[str, str]
T = TypeVar("T")


def build_reference_folder_name(
    device_prefix: str,
    suffix: str = DEVICE_REFERENCE_FOLDER_SUFFIX,
) -> str:
    return f"{device_prefix}{suffix}"


def get_cuda_device_name(index: int = 0) -> str:
    if not torch.cuda.is_available():
        return "CPU"
    return torch.cuda.get_device_name(index)


def resolve_device_reference_folder(
    device_mappings: Sequence[DeviceMapping],
    *,
    device_name: str | None = None,
    fallback_device_prefix: str | None = None,
    unknown_device_prefix: str | None = None,
    suffix: str = DEVICE_REFERENCE_FOLDER_SUFFIX,
    logger: Logger | None = None,
) -> str | None:
    resolved_device_name = device_name or get_cuda_device_name()
    for device_pattern, device_prefix in device_mappings:
        if device_pattern in resolved_device_name:
            return build_reference_folder_name(device_prefix, suffix)

    if logger is not None:
        if fallback_device_prefix is None:
            logger.warning(
                "Unsupported device for ssim tests: %s",
                resolved_device_name,
            )
        else:
            logger.warning(
                "Unsupported device for ssim tests: %s; using %s references",
                resolved_device_name,
                fallback_device_prefix,
            )

    if unknown_device_prefix is not None:
        return build_reference_folder_name(unknown_device_prefix, suffix)
    if fallback_device_prefix is not None:
        return build_reference_folder_name(fallback_device_prefix, suffix)
    return None


def build_reference_folder_path(
    script_dir: str,
    device_reference_folder: str,
    model_id: str,
    attention_backend: str,
) -> str:
    quality_tier = get_output_quality_tier()
    tiered_reference_folder = os.path.join(
        script_dir,
        REFERENCE_VIDEOS_DIRNAME,
        quality_tier,
        device_reference_folder,
        model_id,
        attention_backend,
    )
    if os.path.exists(tiered_reference_folder):
        return tiered_reference_folder

    legacy_reference_folder = os.path.join(
        script_dir,
        device_reference_folder,
        model_id,
        attention_backend,
    )
    if quality_tier == DEFAULT_OUTPUT_QUALITY_TIER and os.path.exists(
        legacy_reference_folder
    ):
        return legacy_reference_folder

    return tiered_reference_folder


def use_full_quality_configs() -> bool:
    value = os.environ.get(FULL_QUALITY_ENV_VAR, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_output_quality_tier() -> str:
    if use_full_quality_configs():
        return FULL_OUTPUT_QUALITY_TIER
    return DEFAULT_OUTPUT_QUALITY_TIER


def build_generated_output_dir(
    script_dir: str,
    device_reference_folder: str,
    model_id: str,
    attention_backend: str,
) -> str:
    return os.path.join(
        script_dir,
        "generated_videos",
        get_output_quality_tier(),
        device_reference_folder,
        model_id,
        attention_backend,
    )


def select_ssim_params(default_params: T, full_quality_params: T) -> T:
    if use_full_quality_configs():
        return full_quality_params
    return default_params
