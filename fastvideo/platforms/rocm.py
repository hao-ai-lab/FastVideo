# SPDX-License-Identifier: Apache-2.0
# Adapted from rocm/vllm: https://github.com/ROCm/vllm/blob/v0.7.3%2Brocm/vllm/platforms/rocm.py
"""
This file is a platform abstraction for ROCm GPUs,
adjusted to match the structure and interface of `cuda.py`.
"""

import torch

import fastvideo.envs as envs
from fastvideo.logger import init_logger
from fastvideo.platforms.interface import (AttentionBackendEnum, DeviceCapability, Platform, PlatformEnum)

logger = init_logger(__name__)


# ROCm uses the same torch.cuda interface
class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM
    device_name: str = "rocm"
    device_type: str = "cuda"  # torch uses 'cuda' backend string
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.cuda.get_device_name(device_id))

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(device_id).total_memory

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        if enforce_eager:
            logger.warning("To see benefits of async output processing, enable CUDA graph. "
                           "Since enforce-eager is enabled, async output processor cannot be used")
            return False
        return True

    @classmethod
    def log_warnings(cls) -> None:
        pass  # ROCm-specific warnings can be added here

    @classmethod
    def get_current_memory_usage(cls, device: torch.device | None = None) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        return float(torch.cuda.max_memory_allocated(device))

    @classmethod
    def get_torch_device(cls):
        """
        Return torch.cuda
        """
        return torch.cuda

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: AttentionBackendEnum | None, head_size: int,
                             dtype: torch.dtype) -> str:
        logger.info("Trying FASTVIDEO_ATTENTION_BACKEND=%s", envs.FASTVIDEO_ATTENTION_BACKEND)

        if selected_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend.")
            return "fastvideo.attention.backends.sdpa.SDPABackend"

        elif selected_backend in (AttentionBackendEnum.FLASH_ATTN, None):
            pass

        elif selected_backend == AttentionBackendEnum.VIDEO_SPARSE_ATTN:
            # fastvideo_kernel's block-sparse attention dispatcher takes its
            # Triton route here: the ThunderKittens kernels are CUDA-only and
            # the CuTe fastpath (FASTVIDEO_VSA_CUTEDSL) is not available on ROCm.
            try:
                from fastvideo_kernel import video_sparse_attn  # noqa: F401
            except ImportError as e:
                raise ImportError("VIDEO_SPARSE_ATTN selected but fastvideo_kernel is not importable. On ROCm it "
                                  "runs through its Triton kernels; build it with fastvideo-kernel/build.sh --rocm "
                                  "or pick a different FASTVIDEO_ATTENTION_BACKEND.") from e
            logger.info("Using Video Sparse Attention backend (Triton kernels).")
            return "fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend"

        elif selected_backend == AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3:
            try:
                from fastvideo_kernel.block_sparse_attn_256 import (  # noqa: F401
                    block_sparse_attn_256_bshd)
            except ImportError as e:
                raise ImportError("VIDEO_SPARSE_ATTN_H3 selected but fastvideo_kernel is not importable. On ROCm "
                                  "its block-sparse kernels run through Triton; build it with "
                                  "fastvideo-kernel/build.sh --rocm or pick a different "
                                  "FASTVIDEO_ATTENTION_BACKEND.") from e
            logger.info("Using MiniMax-H3 Video Sparse Attention backend (Triton kernels).")
            return "fastvideo.attention.backends.video_sparse_attn_h3.MiniMaxH3VSABackend"

        elif selected_backend == AttentionBackendEnum.SAGE_ATTN:
            raise ValueError(f"{selected_backend.name} is not supported on {cls.device_name}.")
        elif selected_backend:
            raise ValueError(f"Invalid attention backend for {cls.device_name}: {selected_backend}")

        target_backend = AttentionBackendEnum.FLASH_ATTN
        if dtype not in (torch.float16, torch.bfloat16):
            logger.info("Cannot use FlashAttention backend for dtype other than "
                        "torch.float16 or torch.bfloat16.")
            target_backend = AttentionBackendEnum.TORCH_SDPA

        if target_backend == AttentionBackendEnum.FLASH_ATTN:
            try:
                import flash_attn  # noqa: F401

                from fastvideo.attention.backends.flash_attn import (  # noqa: F401
                    FlashAttentionBackend)

                supported_sizes = \
                    FlashAttentionBackend.get_supported_head_sizes()
                if head_size not in supported_sizes:
                    logger.info("Cannot use FlashAttention-2 backend for head size %d.", head_size)
                    target_backend = AttentionBackendEnum.TORCH_SDPA
            except ImportError:
                logger.info("Cannot use FlashAttention backend because the "
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
    def get_device_communicator_cls(cls) -> str:
        return "fastvideo.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # works for ROCm too
