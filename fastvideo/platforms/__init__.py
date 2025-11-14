# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/__init__.py

import traceback
from typing import TYPE_CHECKING

from fastvideo.logger import init_logger
# imported by other files, do not remove
from fastvideo.platforms.interface import AttentionBackendEnum  # noqa: F401
from fastvideo.platforms.interface import Platform, PlatformEnum
from fastvideo.utils import resolve_obj_by_qualname

logger = init_logger(__name__)


def cuda_platform_plugin() -> str | None:
    is_cuda = False

    try:
        from fastvideo.utils import import_pynvml
        pynvml = import_pynvml()  # type: ignore[no-untyped-call]
        pynvml.nvmlInit()
        try:
            # NOTE: Edge case: fastvideo cpu build on a GPU machine.
            # Third-party pynvml can be imported in cpu build,
            # we need to check if fastvideo is built with cpu too.
            # Otherwise, fastvideo will always activate cuda plugin
            # on a GPU machine, even if in a cpu build.
            is_cuda = (pynvml.nvmlDeviceGetCount() > 0)
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        if "nvml" not in e.__class__.__name__.lower():
            # If the error is not related to NVML, re-raise it.
            raise e

        # CUDA is supported on Jetson, but NVML may not be.
        import os

        def cuda_is_jetson() -> bool:
            return os.path.isfile("/etc/nv_tegra_release") \
                or os.path.exists("/sys/class/tegra-firmware")

        if cuda_is_jetson():
            is_cuda = True
    if is_cuda:
        logger.info("CUDA is available")

    return "fastvideo.platforms.cuda.CudaPlatform" if is_cuda else None


def mps_platform_plugin() -> str | None:
    """Detect if MPS (Metal Performance Shaders) is available on macOS."""
    is_mps = False

    try:
        import torch
        if torch.backends.mps.is_available():
            is_mps = True
            logger.info("MPS (Metal Performance Shaders) is available")
    except Exception as e:
        logger.info("MPS detection failed: %s", e)

    return "fastvideo.platforms.mps.MpsPlatform" if is_mps else None


def npu_platform_plugin() -> str | None:
    is_npu = False

    try:
        import torch
        # 导入 torch_npu 以初始化 NPU 后端
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            is_npu = True
            logger.info("NPU is available")
    except ImportError:
        logger.error(
            "NPU detection failed: PyTorch or PyTorch_NPU is not installed")
    except AttributeError:
        logger.error(
            "NPU detection failed: PyTorch has no 'npu' attribute (use Ascend-adapted PyTorch)"
        )
    except Exception as e:
        logger.error("NPU detection failed: unknown error - %s", str(e))

    return "fastvideo.platforms.npu.NPUPlatform" if is_npu else None


def cpu_platform_plugin() -> str | None:
    """Detect if CPU platform should be used."""
    # CPU is always available as a fallback
    return "fastvideo.platforms.cpu.CpuPlatform"


def rocm_platform_plugin() -> str | None:
    is_rocm = False

    try:
        import amdsmi
        amdsmi.amdsmi_init()
        try:
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                is_rocm = True
                logger.info("ROCm platform is available")
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.info("ROCm platform is unavailable: %s", e)

    return "fastvideo.platforms.rocm.RocmPlatform" if is_rocm else None


builtin_platform_plugins = {
    'cuda': cuda_platform_plugin,
    'rocm': rocm_platform_plugin,
    'mps': mps_platform_plugin,
    'cpu': cpu_platform_plugin,
    'npu': npu_platform_plugin,
}


def resolve_current_platform_cls_qualname() -> str:
    # TODO(will): if we need to support other platforms, we should consider if
    # vLLM's plugin architecture is suitable for our needs.

    # Try MPS first on macOS
    platform_cls_qualname = mps_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    # Fall back to ROCm
    platform_cls_qualname = rocm_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    # Fall back to CUDA
    platform_cls_qualname = cuda_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    # Fall back to NPU
    platform_cls_qualname = npu_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    # Fall back to CPU as last resort
    platform_cls_qualname = cpu_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    raise RuntimeError("No platform plugin found. Please check your "
                       "installation.")


_current_platform = None
_init_trace: str = ''

if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name: str):
    if name == 'current_platform':
        # lazy init current_platform.
        # 1. out-of-tree platform plugins need `from fastvideo.platforms import
        #    Platform` so that they can inherit `Platform` class. Therefore,
        #    we cannot resolve `current_platform` during the import of
        #    `fastvideo.platforms`.
        # 2. when users use out-of-tree platform plugins, they might run
        #    `import fastvideo`, some fastvideo internal code might access
        #    `current_platform` during the import, and we need to make sure
        #    `current_platform` is only resolved after the plugins are loaded
        #    (we have tests for this, if any developer violate this, they will
        #    see the test failures).
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(
            f"No attribute named '{name}' exists in {__name__}.")


__all__ = ['Platform', 'PlatformEnum', 'current_platform', "_init_trace"]
