import torch
import os


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class ProcessGroupSingleton(Singleton):

    def __init__(self):
        self.ULYSSES_PG = None
        self.RING_PG = None


PROCESS_GROUP = ProcessGroupSingleton()


def set_seq_parallel_pg(sp_ulysses_degree,
                        sp_ring_degree,
                        rank,
                        world_size,
                        use_ulysses_low=True):
    """
    sp_ulysses_degree x sp_ring_degree = seq_parallel_degree
    (ulysses_degree, dp_degree)
    """
    sp_degree = sp_ring_degree * sp_ulysses_degree
    dp_degree = world_size // sp_degree

    assert (world_size % sp_degree == 0
            ), f"world_size {world_size} % sp_degree {sp_ulysses_degree} == 0"

    num_ulysses_pgs = sp_ring_degree  # world_size // sp_ulysses_degree
    num_ring_pgs = sp_ulysses_degree  # world_size // sp_ring_degree

    if use_ulysses_low:
        for dp_rank in range(dp_degree):
            offset = dp_rank * sp_degree
            for i in range(num_ulysses_pgs):
                ulysses_ranks = list(
                    range(
                        i * sp_ulysses_degree + offset,
                        (i + 1) * sp_ulysses_degree + offset,
                    ))
                group = torch.distributed.new_group(ulysses_ranks)
                if rank in ulysses_ranks:
                    ulyssess_pg = group

            for i in range(num_ring_pgs):
                ring_ranks = list(
                    range(i + offset, sp_degree + offset, num_ring_pgs))
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    ring_pg = group

    else:
        for dp_rank in range(dp_degree):
            offset = dp_rank * sp_degree
            for i in range(num_ring_pgs):
                ring_ranks = list(
                    range(i * sp_ring_degree + offset,
                          (i + 1) * sp_ring_degree + offset))
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    ring_pg = group

            for i in range(num_ulysses_pgs):
                ulysses_ranks = list(
                    range(i + offset, sp_degree + offset, num_ulysses_pgs))
                group = torch.distributed.new_group(ulysses_ranks)
                if rank in ulysses_ranks:
                    ulyssess_pg = group

    PROCESS_GROUP.ULYSSES_PG = ulyssess_pg
    PROCESS_GROUP.RING_PG = ring_pg


# test if flash_attn is available
try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from flash_attn_interface import _flash_attn_forward as flash_attn_forward_hopper
    from flash_attn_interface import _flash_attn_backward as flash_attn_func_hopper_backward
    from flash_attn_interface import flash_attn_func as flash3_attn_func
    HAS_FLASH_ATTN_HOPPER = True
except ImportError:
    HAS_FLASH_ATTN_HOPPER = False

try:
    from flashinfer.prefill import single_prefill_with_kv_cache
    HAS_FLASHINFER = True

    def get_cuda_arch():
        major, minor = torch.cuda.get_device_capability()
        return f"{major}.{minor}"

    cuda_arch = get_cuda_arch()
    os.environ['TORCH_CUDA_ARCH_LIST'] = cuda_arch
    print(f"Set TORCH_CUDA_ARCH_LIST to {cuda_arch}")
except ImportError:
    HAS_FLASHINFER = False

try:
    import aiter
    from aiter import flash_attn_func as flash_attn_func_aiter
    HAS_AITER = True
except ImportError:
    HAS_AITER = False

try:
    import sageattention
    HAS_SAGE_ATTENTION = True
except ImportError:
    HAS_SAGE_ATTENTION = False

try:
    import spas_sage_attn
    HAS_SPARSE_SAGE_ATTENTION = True
except ImportError:
    HAS_SPARSE_SAGE_ATTENTION = False

try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False
