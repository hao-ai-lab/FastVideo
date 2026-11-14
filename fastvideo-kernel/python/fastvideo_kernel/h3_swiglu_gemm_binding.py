# SPDX-License-Identifier: Apache-2.0
"""Private eager binding for the SM121 H3 SwiGLU/FP4 GEMM epilogue.

``fastvideo-kernel`` builds ``lib_fastvideo_h3_swiglu_gemm.so`` beside this module when
``TORCH_CUDA_ARCH_LIST`` includes 12.1a. Importing raises ImportError when it is absent.
Inputs are prequantized, with value/gate weight columns paired in memory.
"""
import ctypes
from pathlib import Path

import torch

try:
    _library = ctypes.CDLL(str(Path(__file__).with_name('lib_fastvideo_h3_swiglu_gemm.so')))
except OSError as exc:
    raise ImportError(f'H3 SwiGLU GEMM library is not built: {exc}') from exc
_launch = _library.launch_gemm_swiglu
_launch.argtypes = ([ctypes.c_void_p] * 10 + [ctypes.c_int] * 3
                    + [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
_launch.restype = ctypes.c_int
# One workspace per (device, stream): the CUTLASS scheduler writes it during the launch.
_workspaces: dict[tuple[torch.device, int], torch.Tensor] = {}


def _require(name, tensor, device, dtypes, numel=0, contiguous=True):
    if not isinstance(tensor, torch.Tensor) or tensor.device != device:
        raise ValueError(f'H3 SwiGLU GEMM {name} must be a tensor on {device}')
    if tensor.dtype not in dtypes:
        raise ValueError(f'H3 SwiGLU GEMM {name} has dtype {tensor.dtype}; expected one of {dtypes}')
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f'H3 SwiGLU GEMM {name} must be contiguous')
    if tensor.numel() < numel:
        raise ValueError(f'H3 SwiGLU GEMM {name} holds {tensor.numel()} elements; the launch writes {numel}')


def _validate(a, b, a_scale, b_scale, alpha, global_scale, q, scales, amax, values):
    if a.dim() != 2 or b.dim() != 2 or b.shape[0] != a.shape[1] or not b.T.is_contiguous():
        raise ValueError(f'H3 SwiGLU GEMM needs a [M, K/2] and b [K/2, N] with a contiguous b.T; got '
                         f'{tuple(a.shape)}, {tuple(b.shape)}')
    device, byte = a.device, (torch.uint8, torch.float8_e4m3fn)
    rows, n = a.shape[0], b.shape[1]
    padded_rows = (rows + 127) // 128 * 128
    _require('a', a, device, (torch.uint8, ))
    _require('b', b, device, (torch.uint8, ), contiguous=False)
    _require('a_scale', a_scale, device, byte)
    _require('b_scale', b_scale, device, byte, contiguous=False)
    _require('alpha', alpha, device, (torch.float32, ), 1)
    _require('global_scale', global_scale, device, (torch.float32, ), 1)
    _require('q', q, device, (torch.uint8, ), rows * (n // 4))
    _require('scales', scales, device, byte, padded_rows * (n // 32))
    if amax is not None:
        _require('amax', amax, device, (torch.float32, ), padded_rows * (n // 32))
    if values is not None:
        _require('values', values, device, (torch.bfloat16, ), rows * (n // 2))


def run(a, b, a_scale, b_scale, alpha, global_scale, q, scales, amax=None, values=None,
        raster=0, swizzle=1):
    """Write packed SwiGLU outputs; b and b_scale have contiguous transposes.

    ``q`` holds ``M * N/4`` packed codes. ``scales`` and ``amax`` cover rows padded
    to 128, because the 128x4 scale layout spans whole row tiles. ``values`` holds
    ``M * N/2`` BF16 activations. Undersized or misplaced outputs raise ValueError.
    """
    _validate(a, b, a_scale, b_scale, alpha, global_scale, q, scales, amax, values)
    stream = torch.cuda.current_stream(a.device).cuda_stream
    workspace = _workspaces.get((a.device, stream))
    if workspace is None:
        workspace = torch.empty(16 * 1024 * 1024, device=a.device, dtype=torch.uint8)
        _workspaces[(a.device, stream)] = workspace
    rc = _launch(
        a.data_ptr(), b.data_ptr(), a_scale.data_ptr(), b_scale.data_ptr(), alpha.data_ptr(),
        global_scale.data_ptr(), q.data_ptr(), scales.data_ptr(),
        0 if amax is None else amax.data_ptr(), 0 if values is None else values.data_ptr(),
        a.shape[0], b.shape[1], a.shape[1] * 2, workspace.data_ptr(), workspace.numel(),
        raster, swizzle, stream,
    )
    if rc:
        if rc == -2:
            raise ValueError(f'H3 SwiGLU GEMM needs M > 0, N % 128 == 0 and K % 64 == 0; got {tuple(a.shape)}, '
                             f'{tuple(b.shape)}')
        raise RuntimeError(f'H3 SwiGLU GEMM returned CUDA/CUTLASS error {rc}')
