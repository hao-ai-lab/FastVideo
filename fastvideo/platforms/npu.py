import gc
from datetime import timedelta

import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import PrefixStore

import fastvideo.envs as envs
from fastvideo.logger import init_logger
from fastvideo.platforms.interface import (AttentionBackendEnum, Platform,
                                           PlatformEnum)

logger = init_logger(__name__)


class NPUPlatform(Platform):

    _enum = PlatformEnum.NPU
    device_name: str = "npu"
    device_type: str = "npu"
    simple_compile_backend: str = "eager"  # Disable torch.compile()
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    dispatch_key: str = "PrivateUse1"

    def is_sleep_mode_available(self) -> bool:
        return True

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.npu.get_device_name(device_id)

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
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
    def mem_get_info(cls) -> tuple[int, int]:
        return torch.npu.mem_get_info()

    @classmethod
    def clear_npu_memory(cls):
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: AttentionBackendEnum | None,
                             head_size: int, dtype: torch.dtype) -> str:
        logger.info("Trying FASTVIDEO_ATTENTION_BACKEND=%s",
                    envs.FASTVIDEO_ATTENTION_BACKEND)
        if envs.FASTVIDEO_ATTENTION_BACKEND != "TORCH_SDPA":
            logger.info("Ascend NPU only supports the Torch SDPA backend.")
        else:
            logger.info("Using Torch SDPA backend.")
        return "fastvideo.attention.backends.sdpa.SDPABackend"

    @classmethod
    def get_current_memory_usage(cls,
                                 device: torch.types.Device | None = None
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
    def get_torch_device(cls):
        """
        Return torch.npu
        """
        return torch.npu

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
