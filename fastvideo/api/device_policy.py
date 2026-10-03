# SPDX-License-Identifier: Apache-2.0
"""Offload decisions that depend on the device a worker binds, as overrides of a resolved config.

Resolution runs before any device is chosen, so it cannot tell whether host and device memory are one physical pool.
A worker calls :func:`finalize_device_offload_policy` once after it binds its device and keeps the returned config.
Each decision is a recorded override (``device_policy:<name>``) on a new config object; the input is unchanged.

:func:`offload_disabled_on_unified_memory` answers the same question without overriding, for a loader that chooses a
target device for one component.
"""
from __future__ import annotations

from typing import Any

from fastvideo.api.resolution import ResolvedGeneratorConfig
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Offload settings that trade device memory for host memory, by flat name. All of them are a loss on a device whose
# host and device memory are the same physical pool.
UNIFIED_MEMORY_OFFLOAD_PATHS = {
    "dit_layerwise_offload": "engine.offload.dit_layerwise",
    "dit_cpu_offload": "engine.offload.dit",
    "text_encoder_cpu_offload": "engine.offload.text_encoder",
    "image_encoder_cpu_offload": "engine.offload.image_encoder",
    "vae_cpu_offload": "engine.offload.vae",
}


def _device_name(device_id: int) -> str:
    """Device name for log messages; the platform name when the device cannot be named."""
    from fastvideo.platforms import current_platform

    try:
        return current_platform.get_device_name(device_id)
    except Exception:
        # Device naming is diagnostic only. NVML can be unavailable on an integrated GPU (for example Jetson), and
        # its physical-ordinal lookup cannot interpret CUDA_VISIBLE_DEVICES UUID/MIG selectors.
        return current_platform.device_name


def has_unified_memory(device_id: int) -> bool:
    """Whether the bound device shares one memory pool with the host.

    CUDA's probe reads runtime device properties and may initialize a CUDA context, so call it only in a process
    that owns ``device_id``.
    """
    from fastvideo.platforms import current_platform

    return bool(current_platform.has_unified_memory(device_id))


def offload_disabled_on_unified_memory(device_id: int, offload_flag: str | None = None) -> bool:
    """Whether the unified-memory policy turns off ``offload_flag`` (any offload when ``None``) on ``device_id``."""
    if not has_unified_memory(device_id):
        return False
    return offload_flag is None or offload_flag in UNIFIED_MEMORY_OFFLOAD_PATHS


def disable_offload_on_unified_memory(resolved_config: ResolvedGeneratorConfig,
                                      device_id: int,
                                      *,
                                      unified: bool | None = None) -> ResolvedGeneratorConfig:
    """Turn off every enabled host offload when ``device_id`` has unified memory.

    ``unified`` is the probe result when the caller already has it. The decision is the override
    ``device_policy:unified_memory``.
    """
    if unified is None:
        unified = has_unified_memory(device_id)
    if not unified:
        return resolved_config
    enabled = [flag for flag, path in UNIFIED_MEMORY_OFFLOAD_PATHS.items() if _typed_value(resolved_config, path)]
    if not enabled:
        return resolved_config
    device_name = _device_name(device_id)
    for flag in enabled:
        logger.info(
            "Disabling %s: %s has unified memory, so moving weights to the host duplicates them rather than "
            "freeing device memory.", flag, device_name)
    return resolved_config.with_override("device_policy:unified_memory",
                                         {UNIFIED_MEMORY_OFFLOAD_PATHS[flag]: False
                                          for flag in enabled})


def resolve_device_offload_conflicts(resolved_config: ResolvedGeneratorConfig) -> ResolvedGeneratorConfig:
    """Turn off offload modes that cannot run together on this platform.

    On MPS, FSDP inference and layerwise offload turn off. With layerwise offload on, FSDP inference and DiT CPU
    offload turn off.
    """
    from fastvideo.platforms import current_platform

    if current_platform.is_mps():
        resolved_config = resolved_config.with_override("device_policy:mps", {
            "engine.use_fsdp_inference": False,
            "engine.offload.dit_layerwise": False
        })
    if _typed_value(resolved_config, "engine.offload.dit_layerwise"):
        if _typed_value(resolved_config, "engine.use_fsdp_inference"):
            logger.warning("dit_layerwise_offload is enabled, automatically disabling use_fsdp_inference.")
            resolved_config = resolved_config.with_override("device_policy:layerwise_offload",
                                                            {"engine.use_fsdp_inference": False})
        if _typed_value(resolved_config, "engine.offload.dit"):
            logger.warning("dit_layerwise_offload is enabled, automatically disabling dit_cpu_offload.")
            resolved_config = resolved_config.with_override("device_policy:layerwise_offload",
                                                            {"engine.offload.dit": False})
    return resolved_config


def finalize_device_offload_policy(resolved_config: ResolvedGeneratorConfig,
                                   device_id: int = 0) -> ResolvedGeneratorConfig:
    """Apply the device-local memory policy of ``device_id``, then resolve incompatible offload modes.

    On unified memory every host offload turns off. An unset ``engine.offload.lazy_module_load`` becomes ``True`` on
    unified memory for inference and ``False`` otherwise. Applying the policy to its own result changes nothing.
    """
    unified = has_unified_memory(device_id)
    resolved_config = disable_offload_on_unified_memory(resolved_config, device_id, unified=unified)
    if _typed_value(resolved_config, "engine.offload.lazy_module_load") is None:
        lazy_module_load = unified and not resolved_config.training_mode
        resolved_config = resolved_config.with_override("device_policy:lazy_module_load",
                                                        {"engine.offload.lazy_module_load": lazy_module_load})
        if lazy_module_load:
            logger.info(
                "Enabling lazy_module_load: %s has unified memory, so encoder, DiT, and VAEs cannot stay "
                "resident together. Pass --no-lazy-module-load to keep every component loaded.",
                _device_name(device_id),
            )
    return resolve_device_offload_conflicts(resolved_config)


def _typed_value(resolved_config: ResolvedGeneratorConfig, path: str) -> Any:
    """The resolved value of a dotted path."""
    return resolved_config.provenance(path).value


__all__ = [
    "UNIFIED_MEMORY_OFFLOAD_PATHS",
    "disable_offload_on_unified_memory",
    "finalize_device_offload_policy",
    "has_unified_memory",
    "offload_disabled_on_unified_memory",
    "resolve_device_offload_conflicts",
]
