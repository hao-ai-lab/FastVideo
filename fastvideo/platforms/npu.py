import gc
import os
from datetime import timedelta
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import PrefixStore

import os
from collections.abc import Callable
from functools import lru_cache, wraps
from typing import TypeVar

import torch
from typing_extensions import ParamSpec

import fastvideo.envs as envs
from fastvideo.logger import init_logger
from fastvideo.platforms.interface import (AttentionBackendEnum,
                                           DeviceCapability, Platform,
                                           PlatformEnum)
from fastvideo.utils import import_pynvml

logger = init_logger(__name__)

class NPUPlatform(Platform):

    _enum = PlatformEnum.NPU
    device_name: str = "npu"
    device_type: str = "npu"
    simple_compile_backend: str = "eager"  # Disable torch.compile()
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    dispatch_key: str = "PrivateUse1"

    # supported_quantization: list[str] = [ASCEND_QUATIZATION_METHOD]

    def is_sleep_mode_available(self) -> bool:
        return True

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.npu.get_device_name(device_id)

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True

    @classmethod
    def inference_mode(cls):
        return torch.inference_mode()

    @classmethod
    def set_device(cls, device: torch.device):
        torch.npu.set_device(device)

    @classmethod
    def empty_cache(cls):
        torch.npu.empty_cache()

    @classmethod
    def synchronize(cls):
        torch.npu.synchronize()

    @classmethod
    def mem_get_info(cls) -> Tuple[int, int]:
        return torch.npu.mem_get_info()

    @classmethod
    def clear_npu_memory(cls):
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: AttentionBackendEnum | None,
                             head_size: int, dtype: torch.dtype) -> str:
        # the NPU only supports Flash Attention
        # TODO(will): Other tasks will be synchronized in subsequent updates.

        logger.info("Trying FASTVIDEO_ATTENTION_BACKEND=%s",
                    envs.FASTVIDEO_ATTENTION_BACKEND)
        if selected_backend == AttentionBackendEnum.SLIDING_TILE_ATTN:
            try:
                from st_attn import sliding_tile_attention  # noqa: F401

                from fastvideo.attention.backends.sliding_tile_attn import (  # noqa: F401
                    SlidingTileAttentionBackend)
                logger.info("Using Sliding Tile Attention backend.")

                return "fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend"
            except ImportError as e:
                logger.error(
                    "Failed to import Sliding Tile Attention backend: %s",
                    str(e))
                raise ImportError(
                    "Sliding Tile Attention backend is not installed. ") from e
        elif selected_backend == AttentionBackendEnum.SAGE_ATTN:
            try:
                from sageattention import sageattn  # noqa: F401

                from fastvideo.attention.backends.sage_attn import (  # noqa: F401
                    SageAttentionBackend)
                logger.info("Using Sage Attention backend.")

                return "fastvideo.attention.backends.sage_attn.SageAttentionBackend"
            except ImportError as e:
                logger.info(e)
                logger.info(
                    "Sage Attention backend is not installed. Fall back to Flash Attention."
                )
        elif selected_backend == AttentionBackendEnum.VIDEO_SPARSE_ATTN:
            try:
                from vsa import block_sparse_attn  # noqa: F401

                from fastvideo.attention.backends.video_sparse_attn import (  # noqa: F401
                    VideoSparseAttentionBackend)
                logger.info("Using Video Sparse Attention backend.")

                return "fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend"
            except ImportError as e:
                logger.error(
                    "Failed to import Video Sparse Attention backend: %s",
                    str(e))
                raise ImportError(
                    "Video Sparse Attention backend is not installed. ") from e
        elif selected_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend.")
            return "fastvideo.attention.backends.sdpa.SDPABackend"
        elif selected_backend == AttentionBackendEnum.FLASH_ATTN or selected_backend is None:
            pass
        elif selected_backend:
            raise ValueError(f"Invalid attention backend for {cls.device_name}")

        target_backend = AttentionBackendEnum.FLASH_ATTN
        if not cls.has_device_capability(80):
            logger.info(
                "Cannot use FlashAttention-2 backend for Volta and Turing "
                "GPUs.")
            target_backend = AttentionBackendEnum.TORCH_SDPA
        elif dtype not in (torch.float16, torch.bfloat16):
            logger.info(
                "Cannot use FlashAttention-2 backend for dtype other than "
                "torch.float16 or torch.bfloat16.")
            target_backend = AttentionBackendEnum.TORCH_SDPA

        # FlashAttn is valid for the model, checking if the package is
        # installed.
        if target_backend == AttentionBackendEnum.FLASH_ATTN:
            try:
                import flash_attn  # noqa: F401

                from fastvideo.attention.backends.flash_attn import (  # noqa: F401
                    FlashAttentionBackend)

                supported_sizes = \
                    FlashAttentionBackend.get_supported_head_sizes()
                if head_size not in supported_sizes:
                    logger.info(
                        "Cannot use FlashAttention-2 backend for head size %d.",
                        head_size)
                    target_backend = AttentionBackendEnum.TORCH_SDPA
            except ImportError:
                logger.info("Cannot use FlashAttention-2 backend because the "
                            "flash_attn package is not found. "
                            "Make sure that flash_attn was built and installed "
                            "(on by default).")
                target_backend = AttentionBackendEnum.TORCH_SDPA

        if target_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend.")

            return "fastvideo.attention.backends.sdpa.SDPABackend"

        logger.info("Using Flash Attention backend.")

        return "fastvideo.attention.backends.flash_attn.FlashAttentionBackend"


    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        torch.npu.reset_peak_memory_stats(device)
        return torch.npu.max_memory_allocated(device)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "fastvideo.distributed.device_communicators.npu_communicator.NpuCommunicator"

    @classmethod
    def is_pin_memory_available(cls):
        return True


    @classmethod
    def stateless_init_device_torch_dist_pg(
        cls,
        backend: str,
        prefix_store: PrefixStore,
        group_rank: int,
        group_size: int,
        timeout: timedelta,
    ) -> ProcessGroup:
        from torch.distributed import is_hccl_available
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL

        assert is_hccl_available()
        options = ProcessGroup.Options(backend=backend)
        pg: ProcessGroup = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
            options,
        )

        backend_options = ProcessGroupHCCL.Options()
        backend_options._timeout = timeout

        backend_class = ProcessGroupHCCL(prefix_store, group_rank, group_size,
                                         backend_options)
        device = torch.device("npu")
        backend_class._set_sequence_number_for_group()
        backend_type = ProcessGroup.BackendType.CUSTOM

        pg._register_backend(device, backend_type, backend_class)
        return pg
