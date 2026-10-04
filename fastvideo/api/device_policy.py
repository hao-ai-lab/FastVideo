# SPDX-License-Identifier: Apache-2.0
"""Offload decisions that depend on the device class of the local process, as resolution steps.

The steps run in the main process during resolution, so every worker starts from a config whose offload settings are
final. They query the platform for ``LOCAL_DEVICE_ID`` (device 0 of the local process) and assume that every device of
a node has the same memory class: a node mixes no unified-memory and discrete devices. Importing ``fastvideo``
initializes the CUDA driver already, so the query adds no new initialization; when the platform cannot answer (for
example ``torch.cuda`` is unavailable), the steps decide "not unified".

``UNIFIED_MEMORY_OFFLOAD_PATHS`` names the offload settings that the unified-memory policy turns off, by the flag name
that loaders and log messages use.

``APPLY_DEVICE_POLICY`` is the switch of every step here. The test isolation
(``fastvideo/tests/api/config_snapshot.py::isolated_environment``) sets it to ``False`` the same way it blocks
downloads, so the golden snapshots do not depend on the machine that produced them; the steps then decide nothing and
the offload settings keep their input values.
"""
from __future__ import annotations

from typing import Any

from fastvideo.api.resolution import ResolutionStep, ResolutionView
from fastvideo.api.schema import ExecutionMode
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# The device whose memory class decides the policy: device 0 of the local process (see the module docstring).
LOCAL_DEVICE_ID = 0
# Whether the device-policy steps decide anything; the test isolation turns it off (see the module docstring).
APPLY_DEVICE_POLICY = True

# Offload settings that trade device memory for host memory, by the flag name that loaders and log messages use. All
# of them are a loss on a device whose host and device memory are the same physical pool.
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


def has_unified_memory(device_id: int = LOCAL_DEVICE_ID) -> bool:
    """Whether ``device_id`` shares one memory pool with the host; ``False`` when the platform cannot answer."""
    from fastvideo.platforms import current_platform

    try:
        return bool(current_platform.has_unified_memory(device_id))
    except Exception as error:
        logger.debug("Treating device %d as discrete: the unified-memory query failed (%s)", device_id, error)
        return False


def apply_unified_memory_offload_policy(view: ResolutionView) -> dict[str, Any]:
    """Every enabled host offload of ``UNIFIED_MEMORY_OFFLOAD_PATHS`` turns off when the local device has unified
    memory: moving weights to the host duplicates them instead of freeing device memory."""
    if not APPLY_DEVICE_POLICY or not has_unified_memory(LOCAL_DEVICE_ID):
        return {}
    enabled = [flag for flag, path in UNIFIED_MEMORY_OFFLOAD_PATHS.items() if view.get(path)]
    if not enabled:
        return {}
    device_name = _device_name(LOCAL_DEVICE_ID)
    for flag in enabled:
        logger.info(
            "Disabling %s: %s has unified memory, so moving weights to the host duplicates them rather than "
            "freeing device memory.", flag, device_name)
    return {UNIFIED_MEMORY_OFFLOAD_PATHS[flag]: False for flag in enabled}


def fill_lazy_module_load(view: ResolutionView) -> dict[str, Any]:
    """An unset ``engine.offload.lazy_module_load`` becomes ``True`` on unified memory for inference and ``False``
    otherwise."""
    if not APPLY_DEVICE_POLICY or view.get("engine.offload.lazy_module_load") is not None:
        return {}
    training = view.get("mode") in (ExecutionMode.FINETUNING, ExecutionMode.DISTILLATION)
    lazy_module_load = not training and has_unified_memory(LOCAL_DEVICE_ID)
    if lazy_module_load:
        logger.info(
            "Enabling lazy_module_load: %s has unified memory, so encoder, DiT, and VAEs cannot stay resident "
            "together. Set engine.offload.lazy_module_load to false to keep every component loaded.",
            _device_name(LOCAL_DEVICE_ID),
        )
    return {"engine.offload.lazy_module_load": lazy_module_load}


def apply_mps_offload_policy(view: ResolutionView) -> dict[str, Any]:
    """On MPS, FSDP inference and layerwise offload turn off."""
    if not APPLY_DEVICE_POLICY:
        return {}
    from fastvideo.platforms import current_platform

    if not current_platform.is_mps():
        return {}
    return {"engine.use_fsdp_inference": False, "engine.offload.dit_layerwise": False}


def apply_layerwise_offload_conflicts(view: ResolutionView) -> dict[str, Any]:
    """With layerwise offload on, FSDP inference and DiT CPU offload turn off."""
    if not APPLY_DEVICE_POLICY or not view.get("engine.offload.dit_layerwise"):
        return {}
    values: dict[str, Any] = {}
    if view.get("engine.use_fsdp_inference"):
        logger.warning("dit_layerwise_offload is enabled, automatically disabling use_fsdp_inference.")
        values["engine.use_fsdp_inference"] = False
    if view.get("engine.offload.dit"):
        logger.warning("dit_layerwise_offload is enabled, automatically disabling dit_cpu_offload.")
        values["engine.offload.dit"] = False
    return values


# The device-policy steps in the order that they run: the unified-memory policy first, so the conflict steps see the
# offload settings that survive it.
DEVICE_POLICY_STEPS: tuple[ResolutionStep, ...] = (
    apply_unified_memory_offload_policy,
    fill_lazy_module_load,
    apply_mps_offload_policy,
    apply_layerwise_offload_conflicts,
)

__all__ = [
    "APPLY_DEVICE_POLICY",
    "DEVICE_POLICY_STEPS",
    "LOCAL_DEVICE_ID",
    "UNIFIED_MEMORY_OFFLOAD_PATHS",
    "apply_layerwise_offload_conflicts",
    "apply_mps_offload_policy",
    "apply_unified_memory_offload_policy",
    "fill_lazy_module_load",
    "has_unified_memory",
]
