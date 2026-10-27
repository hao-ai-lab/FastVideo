# SPDX-License-Identifier: Apache-2.0
"""SIMD-group block-sparse attention for MiniMax H3 VSA.

This backend supports tile sizes 64 and 128 and head dimension 128, evaluates the tile map selected by the reference router, and leaves
unsupported shapes to the reference implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_SIMD_KERNEL = None
_SIMD_KERNEL_ERROR: str | None = None

SIMD_TILE_SIZES = (64, 128)
SIMD_HEAD_DIM = 128

_SIMD_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;
"""

# Flash-style block-sparse attention. One threadgroup = (head, video query tile,
# 32-row slice of the tile); each of its 4 SIMD-groups owns 8 query rows. Per
# selected key tile, K/V chunks of BK rows are staged once in threadgroup memory
# (read in place from the [S, H, D] inputs, no gathered copy) and consumed with
# 8x8 simdgroup MMAs and an online softmax in fp32. Only the valid prefix of a
# ragged key tile is visited, and rows of a partially filled query tile are
# written as zero. Packed-tile padding rows are zero in Q/K/V by construction.
_SIMD_SOURCE = """
    constexpr int D = 128;
    constexpr int NSG = 4;
    constexpr int NT = NSG * 32;
    constexpr int BK = 16;
    constexpr int NKB = BK / 8;
    constexpr int RB = 1;
    constexpr int LD = D + 8;
    constexpr float NEG = -3.0e38f;
    threadgroup T ksm[BK * LD];
    threadgroup T vsm[BK * LD];

    uint tid = thread_position_in_threadgroup.x;
    uint sid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    int qsub = (int)threadgroup_position_in_grid.x;
    int q_tile = (int)threadgroup_position_in_grid.y;
    int head = (int)threadgroup_position_in_grid.z;
    int k_max = meta_i[4];
    int n_prefix = meta_i[5];
    int n_video = meta_i[6];
    int H = meta_i[7];
    float scale_log2 = meta_f[0] * 1.44269504088896341f;
    int qt = n_prefix + q_tile;
    int q_valid = vbs[qt];
    int row_base = qsub * NSG * 8 * RB;
    if (row_base >= q_valid) {
        return;
    }
    int row_stride = H * D;
    int srow0 = row_base + (int)sid * 8 * RB;
    int fm = (int)(((lane / 4) & 4) + ((lane / 2) % 4));
    int fn = (int)(((lane / 4) & 2) * 2 + (lane % 2) * 2);

    simdgroup_matrix<T, 8, 8> qf[RB][16];
    simdgroup_matrix<float, 8, 8> acc[RB][16];
    for (int r = 0; r < RB; r++) {
        device const T *qp = q + ((long)(qt * TILE + srow0 + r * 8) * H + head) * D;
        for (int kk = 0; kk < 16; kk++) {
            simdgroup_load(qf[r][kk], qp + kk * 8, row_stride);
            acc[r][kk] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    float row_m[RB];
    float row_l[RB];
    for (int r = 0; r < RB; r++) {
        row_m[r] = NEG;
        row_l[r] = 0.0f;
    }

    int nsel = block_num[head * n_video + q_tile];
    for (int s = 0; s < nsel; s++) {
        int kt = block_idx[(head * n_video + q_tile) * k_max + s];
        int k_valid = vbs[kt];
        long kv0 = (long)kt * TILE * H + head;
        for (int j0 = 0; j0 < k_valid; j0 += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = (int)tid; i < BK * (D / 4); i += NT) {
                int row = i / (D / 4);
                int c4 = i - row * (D / 4);
                long src = (kv0 + (long)(j0 + row) * H) * D + c4 * 4;
                *(threadgroup vec<T, 4> *)(ksm + row * LD + c4 * 4) = *(device const vec<T, 4> *)(k + src);
                *(threadgroup vec<T, 4> *)(vsm + row * LD + c4 * 4) = *(device const vec<T, 4> *)(v + src);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<float, 8, 8> sf[RB][NKB];
            for (int r = 0; r < RB; r++) {
                for (int c = 0; c < NKB; c++) {
                    sf[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
            }
            for (int kk = 0; kk < 16; kk++) {
                for (int c = 0; c < NKB; c++) {
                    simdgroup_matrix<T, 8, 8> km;
                    simdgroup_load(km, (const threadgroup T *)(ksm + c * 8 * LD + kk * 8), LD, ulong2(0, 0), true);
                    for (int r = 0; r < RB; r++) {
                        simdgroup_multiply_accumulate(sf[r][c], qf[r][kk], km, sf[r][c]);
                    }
                }
            }

            simdgroup_matrix<T, 8, 8> pf[RB][NKB];
            for (int r = 0; r < RB; r++) {
                float cmax = NEG;
                for (int c = 0; c < NKB; c++) {
                    for (int e = 0; e < 2; e++) {
                        int col = j0 + c * 8 + fn + e;
                        float sc = (col < k_valid) ? sf[r][c].thread_elements()[e] * scale_log2 : NEG;
                        sf[r][c].thread_elements()[e] = sc;
                        cmax = metal::max(cmax, sc);
                    }
                }
                cmax = metal::max(cmax, simd_shuffle_xor(cmax, ushort(1)));
                cmax = metal::max(cmax, simd_shuffle_xor(cmax, ushort(8)));
                float m_new = metal::max(row_m[r], cmax);
                float alpha = metal::fast::exp2(row_m[r] - m_new);
                row_m[r] = m_new;
                row_l[r] *= alpha;
                for (int c = 0; c < NKB; c++) {
                    for (int e = 0; e < 2; e++) {
                        float p = metal::fast::exp2(sf[r][c].thread_elements()[e] - m_new);
                        row_l[r] += p;
                        pf[r][c].thread_elements()[e] = static_cast<T>(p);
                    }
                }
                for (int kk = 0; kk < 16; kk++) {
                    acc[r][kk].thread_elements()[0] *= alpha;
                    acc[r][kk].thread_elements()[1] *= alpha;
                }
            }
            for (int kk = 0; kk < 16; kk++) {
                for (int c = 0; c < NKB; c++) {
                    simdgroup_matrix<T, 8, 8> vm;
                    simdgroup_load(vm, (const threadgroup T *)(vsm + c * 8 * LD + kk * 8), LD);
                    for (int r = 0; r < RB; r++) {
                        simdgroup_multiply_accumulate(acc[r][kk], pf[r][c], vm, acc[r][kk]);
                    }
                }
            }
        }
    }

    for (int r = 0; r < RB; r++) {
        float l = row_l[r];
        l += simd_shuffle_xor(l, ushort(1));
        l += simd_shuffle_xor(l, ushort(8));
        float inv = l > 0.0f ? 1.0f / l : 0.0f;
        int orow = srow0 + r * 8 + fm;
        if (orow < q_valid) {
            device T *op = out + ((long)(qt * TILE + orow) * H + head) * D + fn;
            for (int kk = 0; kk < 16; kk++) {
                op[kk * 8] = static_cast<T>(acc[r][kk].thread_elements()[0] * inv);
                op[kk * 8 + 1] = static_cast<T>(acc[r][kk].thread_elements()[1] * inv);
            }
        }
    }
"""

# Tuned on M4 Max (tile 128, bf16): 4 SIMD-groups x 8 rows per threadgroup and 16
# staged keys per chunk (BK in the source). 16 rows per SIMD-group spills
# registers; 8 or 32 keys per chunk, and 8 or 16 SIMD-groups, were slower.
_SIMDGROUPS = 4
_THREADS = 32 * _SIMDGROUPS
_ROWS_PER_GROUP = 8 * _SIMDGROUPS


def simd_kernel_error() -> str | None:
    _simd_kernel()
    return _SIMD_KERNEL_ERROR


def disable_simd_kernel(error: Exception) -> None:
    """Remember a failed compile or execution so subsequent blocks use reference."""
    global _SIMD_KERNEL, _SIMD_KERNEL_ERROR
    _SIMD_KERNEL_ERROR = str(error)
    _SIMD_KERNEL = None


def simd_kernel_available() -> bool:
    return _simd_kernel() is not None


def _simd_kernel() -> Any | None:
    global _SIMD_KERNEL, _SIMD_KERNEL_ERROR
    if _SIMD_KERNEL is not None:
        return _SIMD_KERNEL
    if _SIMD_KERNEL_ERROR is not None:
        return None
    try:
        import mlx.core as mx

        if not mx.metal.is_available():
            _SIMD_KERNEL_ERROR = "Metal is not available in this MLX build"
            return None
        # The SIMD kernel compiles but fails BF16 parity on the 7 GiB virtual
        # Metal device in GitHub's M1 runner. Physical M1 devices also use the
        # reference path until this kernel is validated on that GPU family.
        device_info = mx.metal.device_info()
        device_name = str(device_info.get("device_name", ""))
        memory_size = int(device_info.get("memory_size", 0))
        if device_name.startswith("Apple M1") or 0 < memory_size <= 8 * 2**30:
            _SIMD_KERNEL_ERROR = "SIMD VSA BF16 parity is unvalidated on Apple M1 or 8 GiB Metal devices"
            return None
        if not hasattr(mx.fast, "metal_kernel"):
            _SIMD_KERNEL_ERROR = "mx.fast.metal_kernel is not available in this MLX build"
            return None
        kwargs: dict[str, Any] = {
            "name": "h3_vsa_simdgroup_mma",
            "input_names": ["q", "k", "v", "block_idx", "block_num", "vbs", "meta_i", "meta_f"],
            "output_names": ["out"],
            "source": _SIMD_SOURCE,
            "header": _SIMD_HEADER,
        }
        try:
            kernel = mx.fast.metal_kernel(**kwargs, compile_options={"math_mode": "safe"})
        except TypeError:
            kernel = mx.fast.metal_kernel(**kwargs)
        # Constructing a CustomKernel does not compile it. Execute every
        # supported tile and dtype once before advertising this backend.
        for tile in SIMD_TILE_SIZES:
            for dtype in (mx.float32, mx.float16, mx.bfloat16):
                q, indices, counts, geometry = _probe_inputs(tile, dtype)
                probe = _launch_simd_block_sparse(kernel, q, q, q, indices, counts, geometry, SIMD_HEAD_DIM**-0.5)
                mx.eval(probe)
        _SIMD_KERNEL = kernel
    except Exception as error:  # noqa: BLE001 - keep reference usable
        _SIMD_KERNEL_ERROR = str(error)
        _SIMD_KERNEL = None
    return _SIMD_KERNEL


def _probe_inputs(tile: int, dtype):
    """Tiled Q and routing for the capability probe, sized exactly like a real launch."""
    import mlx.core as mx

    from fastvideo.mlx_runtime.minimax_h3_vsa import build_h3_tile_geometry

    geometry = build_h3_tile_geometry((1, ), (4, 4, 8), tile)
    selected = [0, geometry.num_tiles - 1]
    indices = mx.array([[selected] * geometry.num_video_tiles], dtype=mx.int32)
    counts = mx.full((1, geometry.num_video_tiles), len(selected), dtype=mx.int32)
    q = mx.zeros((geometry.padded_length, 1, SIMD_HEAD_DIM), dtype=dtype)
    return q, indices, counts, geometry


def simd_block_sparse(
    q_tiled,
    k_tiled,
    v_tiled,
    block_idx,
    block_num,
    geometry,
    scale: float,
):
    """Block-sparse attention over tiled ``[S, H, D]`` using the reference tile map."""

    kernel = _simd_kernel()
    if kernel is None:
        raise RuntimeError(_SIMD_KERNEL_ERROR or "SIMD-group VSA kernel is unavailable")
    return _launch_simd_block_sparse(kernel, q_tiled, k_tiled, v_tiled, block_idx, block_num, geometry, scale)


def _launch_simd_block_sparse(kernel: Any, q_tiled: Any, k_tiled: Any, v_tiled: Any, block_idx: Any, block_num: Any,
                              geometry: Any, scale: float) -> Any:
    """Run the kernel on tiled ``[S_pad, H, D]`` inputs; returns the same layout."""
    import mlx.core as mx

    tile = geometry.tile_elems
    if tile not in SIMD_TILE_SIZES:
        raise ValueError(f"SIMD-group VSA kernel supports tiles {SIMD_TILE_SIZES}, got {tile}")
    heads, dim = q_tiled.shape[1], q_tiled.shape[2]
    if dim != SIMD_HEAD_DIM:
        raise ValueError(f"SIMD-group VSA kernel supports head dim {SIMD_HEAD_DIM}, got {dim}")
    q = mx.contiguous(q_tiled)
    k = mx.contiguous(k_tiled)
    v = mx.contiguous(v_tiled)
    meta_i = mx.array(
        [
            dim,
            tile,
            geometry.padded_length,
            geometry.num_tiles,
            int(block_idx.shape[-1]),
            geometry.num_prefix_tiles,
            geometry.num_video_tiles,
            heads,
        ],
        dtype=mx.int32,
    )
    meta_f = mx.array([scale], dtype=mx.float32)
    call_kwargs = {
        "inputs": [
            q,
            k,
            v,
            mx.contiguous(block_idx.astype(mx.int32)),
            mx.contiguous(block_num.astype(mx.int32)),
            mx.array(geometry.variable_block_sizes.astype(np.int32)),
            meta_i,
            meta_f,
        ],
        "template": [("T", q.dtype), ("TILE", tile)],
        "grid": (_THREADS * (tile // _ROWS_PER_GROUP), geometry.num_video_tiles, heads),
        "threadgroup": (_THREADS, 1, 1),
        "output_shapes": [q.shape],
        "output_dtypes": [q.dtype],
    }
    try:
        outputs = kernel(**call_kwargs, init_value=0)
    except TypeError:
        outputs = kernel(**call_kwargs)
    return outputs[0]
