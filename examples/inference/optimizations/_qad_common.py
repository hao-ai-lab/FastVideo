"""Shared GPU-arch helpers for the QAD 5090 example scripts.

The default (non ``--bf16``) path of the NVFP4 scripts runs DiT linears through
flashinfer's cutlass FP4 gemm, which ships sm_120a cubins only (RTX 5090-class
consumer Blackwell). These helpers pick the right FLASHINFER_CUDA_ARCH_LIST and
fail fast with a readable error instead of an opaque flashinfer ValueError.
"""

import torch

# Capabilities the flashinfer FP4 gemm path has cubins for.
FP4_CAPABILITIES = ((12, 0), (12, 1))


def flashinfer_arch_list() -> str:
    """FLASHINFER_CUDA_ARCH_LIST derived from the local GPU, defaulting to 12.0a."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) in FP4_CAPABILITIES:
            return f"{major}.{minor}a"
    return "12.0a"


def require_fp4_capable_gpu() -> None:
    """Exit with a one-line error if the GPU cannot run the NVFP4 gemm path."""
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    if cap not in FP4_CAPABILITIES:
        raise SystemExit(
            f"The default NVFP4 path requires an sm_120a GPU (RTX 5090-class Blackwell, "
            f"compute capability 12.0/12.1); found {cap}. Re-run with --bf16 on other GPUs.")
