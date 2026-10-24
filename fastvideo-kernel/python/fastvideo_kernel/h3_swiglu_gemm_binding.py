# SPDX-License-Identifier: Apache-2.0
"""Private eager binding for the SM121 H3 SwiGLU/FP4 GEMM epilogue.

The Spark prebuild installs this module beside its fingerprinted native library.
Inputs are prequantized, with value/gate weight columns paired in memory.
"""
import ctypes
from pathlib import Path

import torch

_library = ctypes.CDLL(str(Path(__file__).with_name('lib_fastvideo_h3_swiglu_gemm.so')))
_launch = _library.launch_gemm_swiglu
_launch.argtypes = ([ctypes.c_void_p] * 10 + [ctypes.c_int] * 3
                    + [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
_launch.restype = ctypes.c_int
_workspaces: dict[torch.device, torch.Tensor] = {}


def run(a, b, a_scale, b_scale, alpha, global_scale, q, scales, amax=None, values=None,
        raster=0, swizzle=1):
    """Write packed SwiGLU outputs; b and b_scale have contiguous transposes."""
    workspace = _workspaces.get(a.device)
    if workspace is None:
        workspace = torch.empty(16 * 1024 * 1024, device=a.device, dtype=torch.uint8)
        _workspaces[a.device] = workspace
    rc = _launch(
        a.data_ptr(), b.data_ptr(), a_scale.data_ptr(), b_scale.data_ptr(), alpha.data_ptr(),
        global_scale.data_ptr(), q.data_ptr(), scales.data_ptr(),
        0 if amax is None else amax.data_ptr(), 0 if values is None else values.data_ptr(),
        a.shape[0], b.shape[1], a.shape[1] * 2, workspace.data_ptr(), workspace.numel(),
        raster, swizzle, torch.cuda.current_stream(a.device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f'H3 SwiGLU GEMM returned CUDA/CUTLASS error {rc}')
