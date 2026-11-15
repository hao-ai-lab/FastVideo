# SPDX-License-Identifier: Apache-2.0
"""Fused Q/K preconditioning + NVFP4 quantization for the FlashAttention-4 NVFP4 path.

One persistent Triton kernel per tensor replaces ``precondition_qk`` followed by ``_nvfp4_quantize_for_fa4`` (fp32
casts, mean subtraction, an fp32 Hadamard matmul, the BF16 round, ``F.pad``, flashinfer's ``nvfp4_quantize`` and the
scale-factor permute). Each tile of BLOCK_M tokens x one head:

1. (K, ``smooth_k``) subtracts the per-(batch, head, channel) token mean, computed by
   :func:`reference_token_mean`: the reference's own ``k.float().mean(dim=1)`` reduction, bit for bit, without
   materializing the fp32 copy.
2. (``hadamard``) multiplies by ``hadamard_matrix(D)`` the way the reference's cuBLAS GEMM does. When torch lets
   cuBLAS use TF32 (``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`` in NVIDIA's PyTorch containers, so in training and
   deployment on the cluster), that is a TF32 tensor-core GEMM: operands rounded to nearest-even TF32, exact
   products, fp32 accumulation. The kernel issues the same TF32 MMA, which reproduces cuBLAS bit for bit. With TF32
   off, the fp32 operand is split exactly into three BF16 terms multiplied by the +/-1 matrix and scaled by fp32
   ``1/sqrt(D)``; only fp32 summation order then differs from cuBLAS's fp32 GEMM.
3. Rounds to BF16, as the reference does before quantizing.
4. Quantizes per 16 channels exactly like flashinfer's default (fast-math) ``nvfp4_quantize`` with a unit global
   scale: ``sf = e4m3(amax * rcp.approx(6))``, ``code = cvt.rn.satfinite.e2m1(x * rcp.approx(sf))``.
5. Writes the packed FP4 data as ``(batch, seqlen_padded, nheads, headdim // 2)`` and the scale factors straight into
   FlashAttention-4's MMA layout (a permuted view of the contiguous ``(batch, nheads, rest_m, rest_k, 32, 4, 4)``
   buffer, as the reference returns), with zero codes and zero scales in the rows padding seqlen to a multiple of
   128.

The ``fastvideo::nvfp4_fused_qk_quantize_fa4`` custom op (with a stride-exact fake) keeps
``torch.compile(fullgraph=True)`` tracing.
"""
from __future__ import annotations

import os
from functools import lru_cache

import torch

from fastvideo.attention.utils.nvfp4_qk_precondition import hadamard_matrix

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on hosts without Triton
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

TILE_M = 128  # seqlen padding and scale-factor tile rows (flashinfer layout_128x4)
SF_VEC = 16  # NVFP4 channels per scale factor
SUPPORTED_HEAD_DIMS = (64, 128)


def _flag_set(value: str | None) -> bool:
    return (value or "").strip().lower() not in ("", "0", "false", "no", "off")


def fused_qk_quant_available() -> bool:
    """Process-level gate: Triton is importable and flashinfer's NVFP4 quantization runs its default fast-math
    recipe (the one reproduced here). Read once at attention construction, outside compiled code."""
    non_default_recipe = (_flag_set(os.environ.get("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"))
                          or _flag_set(os.environ.get("TRTLLM_DISABLE_FP4_QUANT_FAST_MATH"))
                          or _flag_set(os.environ.get("FLASHINFER_NVFP4_4OVER6")))
    return _TRITON_AVAILABLE and not non_default_recipe


def fused_qk_quant_supported(query: torch.Tensor, key: torch.Tensor, *, hadamard: bool = False) -> bool:
    """Tensor-level gate: BF16 CUDA ``[B, L, H, D]`` Q/K with a supported, channel-contiguous head dim.

    The Hadamard rotation is bit-exact only against cuBLAS's TF32 GEMM, so with TF32 matmuls off it takes the
    reference path.
    """
    if hadamard and not torch.backends.cuda.matmul.allow_tf32:
        return False
    return (query.is_cuda and key.is_cuda and query.dtype == torch.bfloat16 and key.dtype == torch.bfloat16
            and query.dim() == 4 and key.dim() == 4 and query.shape[0] == key.shape[0]
            and query.shape[-1] == key.shape[-1] and query.shape[-1] in SUPPORTED_HEAD_DIMS and query.stride(-1) == 1
            and key.stride(-1) == 1 and _token_mean_reproducible(key))


def _padded(seqlen: int) -> int:
    return (seqlen + TILE_M - 1) // TILE_M * TILE_M


def _output_shapes(tensor_4d: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
    batch, seqlen, nheads, headdim = tensor_4d.shape
    seqlen_padded = _padded(seqlen)
    fp4_shape = (batch, seqlen_padded, nheads, headdim // 2)
    sf_canonical = (batch, nheads, seqlen_padded // TILE_M, headdim // SF_VEC // 4, 32, 4, 4)
    return fp4_shape, sf_canonical


# FA4's MMA view of the contiguous canonical scale buffer: (32, 4, rest_m, 4, rest_k, nheads, batch), stride[3] = 1.
_SF_MMA_PERMUTE = (4, 5, 2, 6, 3, 1, 0)

if _TRITON_AVAILABLE:

    @triton.jit
    def _rcp_approx_ftz(x):
        return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;",
                                         "=f,f", [x],
                                         dtype=tl.float32,
                                         is_pure=True,
                                         pack=1)

    @triton.jit
    def _e2m1x2(lo, hi):
        # cvt.rn.satfinite.e2m1x2.f32 d, a, b puts a in the high nibble: element 2j -> low nibble (flashinfer order).
        return tl.inline_asm_elementwise(
            """
            {
            .reg .b8 b0, b1, b2, b3;
            cvt.rn.satfinite.e2m1x2.f32 b0, $5, $1;
            cvt.rn.satfinite.e2m1x2.f32 b1, $6, $2;
            cvt.rn.satfinite.e2m1x2.f32 b2, $7, $3;
            cvt.rn.satfinite.e2m1x2.f32 b3, $8, $4;
            mov.b32 $0, {b0, b1, b2, b3};
            }
            """,
            "=r,f,f,f,f,f,f,f,f",
            [lo, hi],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )

    @triton.jit
    def _precondition_tile(x, xf_mean, rotation, hadamard_scale, SUBTRACT_MEAN: tl.constexpr, HADAMARD: tl.constexpr,
                           TF32: tl.constexpr):
        """Steps 1-2 on one [BLOCK_M, D] BF16 tile; returns fp32."""
        if HADAMARD and TF32:
            # Reproduce cuBLAS's TF32 GEMM bit for bit: operands rounded to nearest-even TF32, exact products with
            # the TF32-rounded +/-1/sqrt(D) entries (rotation, fp32), fp32 tensor-core accumulation.
            xf = x.to(tl.float32)
            if SUBTRACT_MEAN:
                xf = xf - xf_mean
                bits = xf.to(tl.int32, bitcast=True)
                bits = (bits + 0xFFF + ((bits >> 13) & 1)) & -8192
                xf = bits.to(tl.float32, bitcast=True)
            xf = tl.dot(xf, rotation, input_precision="tf32", out_dtype=tl.float32)
        elif HADAMARD:
            # fp32 GEMM: split the fp32 operand exactly into three BF16 terms, multiply each by the exact +/-1
            # matrix (rotation, BF16) on tensor cores, then scale by fp32 1/sqrt(D).
            if SUBTRACT_MEAN:
                xf = x.to(tl.float32) - xf_mean
                hi = xf.to(tl.bfloat16)
                rest = xf - hi.to(tl.float32)
                mid = rest.to(tl.bfloat16)
                lo = (rest - mid.to(tl.float32)).to(tl.bfloat16)
                acc = tl.dot(hi, rotation, out_dtype=tl.float32)
                acc = tl.dot(mid, rotation, acc, out_dtype=tl.float32)
                acc = tl.dot(lo, rotation, acc, out_dtype=tl.float32)
            else:
                acc = tl.dot(x, rotation, out_dtype=tl.float32)
            xf = acc * hadamard_scale
        elif SUBTRACT_MEAN:
            xf = x.to(tl.float32) - xf_mean
        else:
            xf = x.to(tl.float32)
        return xf

    @triton.jit
    def _fused_precondition_nvfp4_kernel(
        x_ptr,
        mean_ptr,
        rotation_ptr,
        fp4_ptr,
        sf_ptr,
        seqlen,
        seqlen_padded,
        nheads,
        num_tiles,
        stride_b,
        stride_l,
        stride_h,
        hadamard_scale,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        SUBTRACT_MEAN: tl.constexpr,
        HADAMARD: tl.constexpr,
        TF32: tl.constexpr,
        NUM_STAGES: tl.constexpr,
    ):
        cols = tl.arange(0, D)
        if HADAMARD:  # noqa: SIM108 (Triton resolves constexpr branches; keep the statement form)
            rotation = tl.load(rotation_ptr + cols[:, None] * D + cols[None, :])
        else:
            rotation = None
        tiles_m = seqlen_padded // BLOCK_M
        r = tl.arange(0, BLOCK_M)
        kk = tl.arange(0, D // 16)
        half_cols = tl.arange(0, D // 2)
        # Persistent: each program walks tiles (BLOCK_M tokens x one head) so the rotation is loaded once.
        for tile in tl.range(tl.program_id(0), num_tiles, tl.num_programs(0), num_stages=NUM_STAGES):
            bh = tile // tiles_m
            pid_m = tile % tiles_m
            b = bh // nheads
            h = bh % nheads
            rows = pid_m * BLOCK_M + r
            valid = (rows < seqlen)[:, None]
            src = (x_ptr + b.to(tl.int64) * stride_b + h.to(tl.int64) * stride_h +
                   rows.to(tl.int64)[:, None] * stride_l + cols[None, :])
            x = tl.load(src, mask=valid, other=0.0)
            if SUBTRACT_MEAN:  # noqa: SIM108
                xf_mean = tl.load(mean_ptr + bh * D + cols)[None, :]
            else:
                xf_mean = None
            xf = _precondition_tile(x, xf_mean, rotation, hadamard_scale, SUBTRACT_MEAN, HADAMARD, TF32)

            # Step 3: round to BF16. Rows padding seqlen to a multiple of 128 quantize as zeros.
            y = tl.where(valid, xf, 0.0).to(tl.bfloat16).to(tl.float32)

            # Step 4: flashinfer's fast-math NVFP4 recipe with a unit global scale.
            blocks = tl.reshape(y, [BLOCK_M, D // 16, 16])
            vec_max = tl.max(tl.abs(blocks), axis=2)
            sf_value = vec_max * _rcp_approx_ftz(tl.full([BLOCK_M, D // 16], 6.0, tl.float32))
            sf_e4m3 = sf_value.to(tl.float8e4nv)
            out_scale = tl.where(vec_max != 0.0, _rcp_approx_ftz(sf_e4m3.to(tl.float32)), 0.0)
            scaled = tl.reshape(blocks * out_scale[:, :, None], [BLOCK_M, D // 2, 2])
            even, odd = tl.split(scaled)
            codes = _e2m1x2(even, odd)

            out_rows = (b * seqlen_padded + rows).to(tl.int64)
            tl.store(fp4_ptr + (out_rows[:, None] * nheads + h) * (D // 2) + half_cols[None, :], codes)

            # Scale bytes go straight to the canonical (batch, nheads, rest_m, rest_k, 32, 4, 4) buffer. A 128-row
            # tile owns D // 64 contiguous 512-byte atoms; inside an atom, row r and scale column c land at
            # (r % 32) * 16 + (r // 32) * 4 + c (flashinfer layout_128x4).
            row128 = rows % 128
            atom_base = ((b * nheads + h) * (seqlen_padded // 128) + rows // 128).to(tl.int64) * (D // 16 * 128)
            sf_offsets = (atom_base[:, None] + (kk[None, :] // 4) * 512 + (row128[:, None] % 32) * 16 +
                          (row128[:, None] // 32) * 4 + kk[None, :] % 4)
            tl.store(sf_ptr + sf_offsets, sf_e4m3.to(tl.uint8, bitcast=True))


def _round_tf32(t: torch.Tensor) -> torch.Tensor:
    """fp32 -> nearest-even TF32 (kept in fp32), the operand rounding cuBLAS TF32 GEMMs apply."""
    bits = t.contiguous().view(torch.int32)
    return ((bits + 0xFFF + ((bits >> 13) & 1)) & -8192).view(torch.float32)


@lru_cache(maxsize=8)
def _hadamard_operand(n: int, device: torch.device, tf32: bool) -> torch.Tensor:
    """Kernel operand for the reference ``x @ hadamard_matrix(n)``: TF32-rounded entries in fp32 under TF32, else
    the exact +/-1 signs in BF16 (the kernel scales by fp32 ``1/sqrt(n)`` afterwards)."""
    rotation = hadamard_matrix(n, "cpu")
    if tf32:
        return _round_tf32(rotation).to(device)
    return torch.sign(rotation).to(device=device, dtype=torch.bfloat16).contiguous()


@lru_cache(maxsize=8)
def _hadamard_scale(n: int) -> float:
    """The reference matrix entry ``1/sqrt(n)`` in fp32 (applied after the exact +/-1 product when TF32 is off)."""
    return float(hadamard_matrix(n, "cpu")[0, 0].item())


# (block_m, num_warps, num_stages, ctas_per_sm): persistent-grid launch per mode, tuned on GB200 at L = 58k, H = 56.
LAUNCH_CONFIGS = {"plain": (64, 4, 1, 8), "tf32": (64, 4, 3, 2), "fp32": (64, 4, 3, 2)}


def _launch_config(hadamard: bool, tf32: bool) -> tuple[int, int, int, int]:
    return LAUNCH_CONFIGS["plain" if not hadamard else ("tf32" if tf32 else "fp32")]


@lru_cache(maxsize=8)
def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


_INT32_MAX = 2**31 - 1


def _fits_32bit_indexing(batch: int, seqlen: int, rows: int, row_bytes: int, row_numel: int) -> bool:
    max_offset = (batch - 1) * seqlen * row_bytes + (rows - 1) * row_bytes + (row_bytes - 4)
    return max_offset <= _INT32_MAX and batch * rows * row_numel <= _INT32_MAX


def _token_mean_reproducible(tensor_4d: torch.Tensor) -> bool:
    """Batched inputs whose fp32 copy needs 64-bit indexing split differently in torch; those take the unfused path."""
    batch, seqlen, nheads, headdim = tensor_4d.shape
    return batch == 1 or _fits_32bit_indexing(batch, seqlen, seqlen, nheads * headdim * 4, nheads * headdim)


def _fp32_reduction_chunks(tensor_4d: torch.Tensor) -> list[tuple[int, int]]:
    """Token ranges torch reduces separately in ``k.float().mean(dim=1)``.

    A CUDA reduction whose operand needs 64-bit indexing is split (TensorIterator ``with_32bit_indexing``) by halving
    the largest dimension, here the tokens of the contiguous fp32 copy, until every piece's byte offsets fit in int32;
    the pieces' partial sums are added left to right. 768p/15 s H3 (113k tokens x 56 heads) splits in two.
    """
    batch, seqlen, nheads, headdim = tensor_4d.shape
    row_bytes = nheads * headdim * 4
    if not _token_mean_reproducible(tensor_4d):
        raise ValueError(f"the split fp32 token mean of a batched {tuple(tensor_4d.shape)} input is not reproduced")

    def split(start: int, rows: int) -> list[tuple[int, int]]:
        if _fits_32bit_indexing(batch, seqlen, rows, row_bytes, nheads * headdim):
            return [(start, rows)]
        first = rows // 2
        return split(start, first) + split(start + first, rows - first)

    return split(0, seqlen)


def reference_token_mean(tensor_4d: torch.Tensor) -> torch.Tensor:
    """``tensor_4d.float().mean(dim=1)`` bit for bit (``[B, H, D]`` fp32) without materializing the fp32 copy.

    Reading BF16 with an fp32 accumulator runs the same reduction as the fp32 copy; when that copy needs 64-bit
    indexing, torch reduces token ranges separately, adds the partial sums and scales by ``fp32(H * D) / numel``,
    which is reproduced here. Exact for contiguous (or non-dense) inputs, whose fp32 copy is contiguous.
    """
    chunks = _fp32_reduction_chunks(tensor_4d)
    if len(chunks) == 1:
        return tensor_4d.mean(dim=1, dtype=torch.float32).contiguous()
    total = None
    for start, rows in chunks:
        partial = tensor_4d[:, start:start + rows].sum(dim=1, dtype=torch.float32)
        total = partial if total is None else total + partial
    batch, seqlen, nheads, headdim = tensor_4d.shape
    factor = (torch.tensor(batch * nheads * headdim, dtype=torch.float32) /
              torch.tensor(batch * seqlen * nheads * headdim, dtype=torch.float32)).item()
    return (total * factor).contiguous()


def _precondition_quantize(tensor_4d: torch.Tensor, *, subtract_mean: bool, hadamard: bool,
                           tf32: bool) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seqlen, nheads, headdim = tensor_4d.shape
    fp4_shape, sf_canonical = _output_shapes(tensor_4d)
    seqlen_padded = fp4_shape[1]
    fp4 = torch.empty(fp4_shape, device=tensor_4d.device, dtype=torch.uint8)
    sf = torch.empty(sf_canonical, device=tensor_4d.device, dtype=torch.uint8)
    # The reference's own reduction (``k.float().mean(dim=1)``): reading BF16 with an fp32 accumulator gives the
    # same bits without materializing the fp32 copy.
    mean = rotation = fp4  # placeholder pointers for the operands a mode does not read
    if subtract_mean:
        mean = reference_token_mean(tensor_4d).view(batch * nheads, headdim)
    if hadamard:
        rotation = _hadamard_operand(headdim, tensor_4d.device, tf32)
    block_m, num_warps, num_stages, ctas_per_sm = _launch_config(hadamard, tf32)
    num_tiles = seqlen_padded // block_m * batch * nheads
    grid = (min(num_tiles, _num_sms(tensor_4d.device) * ctas_per_sm), )
    _fused_precondition_nvfp4_kernel[grid](
        tensor_4d,
        mean,
        rotation,
        fp4,
        sf,
        seqlen,
        seqlen_padded,
        nheads,
        num_tiles,
        tensor_4d.stride(0),
        tensor_4d.stride(1),
        tensor_4d.stride(2),
        _hadamard_scale(headdim) if hadamard else 1.0,
        D=headdim,
        BLOCK_M=block_m,
        SUBTRACT_MEAN=subtract_mean,
        HADAMARD=hadamard,
        TF32=tf32 and hadamard,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
    )
    return fp4.view(torch.float4_e2m1fn_x2), sf.permute(*_SF_MMA_PERMUTE)


@torch.library.custom_op(
    "fastvideo::nvfp4_fused_qk_quantize_fa4",
    mutates_args=(),
    device_types="cuda",
)
def _nvfp4_fused_qk_quantize_fa4_op(query: torch.Tensor, key: torch.Tensor, smooth_k: bool,
                                    hadamard: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Mirror the reference ``x_fp32 @ H`` exactly: cuBLAS runs it in TF32 whenever torch allows TF32 matmuls.
    tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    q_fp4, q_sf = _precondition_quantize(query, subtract_mean=False, hadamard=hadamard, tf32=tf32)
    k_fp4, k_sf = _precondition_quantize(key, subtract_mean=smooth_k, hadamard=hadamard, tf32=tf32)
    return q_fp4, q_sf, k_fp4, k_sf


def _fake_outputs(tensor_4d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp4_shape, sf_canonical = _output_shapes(tensor_4d)
    fp4 = tensor_4d.new_empty(fp4_shape, dtype=torch.float4_e2m1fn_x2)
    # Reproduce the real strides: sf is a permuted view of a contiguous canonical buffer.
    sf = tensor_4d.new_empty(sf_canonical, dtype=torch.uint8).permute(*_SF_MMA_PERMUTE)
    return fp4, sf


@torch.library.register_fake("fastvideo::nvfp4_fused_qk_quantize_fa4")
def _nvfp4_fused_qk_quantize_fa4_fake(query: torch.Tensor, key: torch.Tensor, smooth_k: bool,
                                      hadamard: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del smooth_k, hadamard
    return (*_fake_outputs(query), *_fake_outputs(key))


def nvfp4_fused_qk_quantize_fa4(query: torch.Tensor, key: torch.Tensor, *, smooth_k: bool,
                                hadamard: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``precondition_qk`` + ``_nvfp4_quantize_for_fa4`` on Q and K in fused kernels.

    Returns ``(q_fp4, q_sf, k_fp4, k_sf)`` with the reference layout contract (seqlen padded to 128; slice
    ``[:, :seqlen]`` before FA4). Check :func:`fused_qk_quant_supported` first.
    """
    return torch.ops.fastvideo.nvfp4_fused_qk_quantize_fa4(query, key, smooth_k, hadamard)


__all__ = [
    "SUPPORTED_HEAD_DIMS", "fused_qk_quant_available", "fused_qk_quant_supported", "nvfp4_fused_qk_quantize_fa4",
    "reference_token_mean"
]
