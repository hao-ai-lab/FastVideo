# SPDX-License-Identifier: Apache-2.0
"""SIMD-group block-sparse attention for MiniMax H3 VSA.

This backend is an explicit opt-in. It supports tile size 64 and head dimension
128, evaluates the tile map selected by the reference router, and leaves
unsupported shapes to the reference implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_SIMD_KERNEL = None
_SIMD_KERNEL_ERROR: str | None = None

_SIMD_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

METAL_FUNC void scale_rows_simd8x8(thread simdgroup_float8x8 &mat, const threadgroup float *row_scale,
                                   threadgroup float *tmp, uint lane) {
    simdgroup_store(mat, tmp, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8) {
        float s = row_scale[lane];
        for (int c = 0; c < 8; c++) {
            tmp[lane * 8 + c] *= s;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_load(mat, tmp, 8);
}
"""

# One threadgroup = one (head, video query tile). 8 SIMD-groups x 32 = 256
# threads cover 64 query rows. Q is half smem (16 KiB). K/V stage 8 keys.
_SIMD_SOURCE = """
    const int TILE = 64;
    const int D = 128;
    const int SG = 32;
    const int N_SG = 8;
    const int ROWS = 8;
    const int KCHUNK = 8;
    threadgroup float kvsmem[8 * 128];
    threadgroup float score_smem[8 * 64];
    threadgroup float scale_tmp[8 * 64];
    threadgroup float qtile[8 * 64];
    threadgroup float row_alpha[8 * 8];

    uint tid = thread_position_in_threadgroup.x;
    uint sid = tid / SG;
    uint lane = tid % SG;
    uint q_tile = thread_position_in_grid.y;
    uint head = thread_position_in_grid.z;
    int k_max = meta_i[4];
    int n_prefix = meta_i[5];
    int n_video = meta_i[6];
    int S = meta_i[2];
    float scale = meta_f[0];
    bool active = ((int)q_tile < n_video && (int)head < (int)q_shape[0]);
    int qt = n_prefix + (int)q_tile;
    int q_valid = active ? vbs[qt] : 0;
    int q_base_tile = (((int)head * S) + qt * TILE) * D;
    threadgroup float *sg_scores = score_smem + sid * 64;
    threadgroup float *sg_tmp = scale_tmp + sid * 64;
    threadgroup float *sg_qtile = qtile + sid * 64;
    threadgroup float *sg_alpha = row_alpha + sid * 8;
    int qrow0 = (int)sid * ROWS;

    thread simdgroup_float8x8 qfrag[16];
    for (int kk = 0; kk < 16; kk++) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        int r0 = (int)lane / 8;
        int c0 = (int)lane % 8;
        int g0 = qrow0 + r0;
        int g1 = qrow0 + r0 + 4;
        float v0 = 0.0f;
        float v1 = 0.0f;
        if (active && g0 < q_valid) {
            v0 = float(q[q_base_tile + g0 * D + kk * 8 + c0]);
        }
        if (active && g1 < q_valid) {
            v1 = float(q[q_base_tile + g1 * D + kk * 8 + c0]);
        }
        sg_qtile[lane] = v0;
        sg_qtile[lane + 32] = v1;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(qfrag[kk], sg_qtile, 8);
    }

    thread simdgroup_float8x8 acc[16];
    for (int kk = 0; kk < 16; kk++) {
        acc[kk] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_m = -3.402823466e+38f;
    float row_lse = 0.0f;
    int nsel = active ? block_num[(int)head * n_video + (int)q_tile] : 0;

    for (int s = 0; s < nsel; s++) {
        int kt = block_idx[(((int)head * n_video) + (int)q_tile) * k_max + s];
        int k_valid = vbs[kt];
        int kv_base = (((int)head * S) + kt * TILE) * D;
        for (int j0 = 0; j0 < TILE; j0 += KCHUNK) {
            for (uint i = tid; i < (uint)(KCHUNK * D); i += N_SG * SG) {
                int row = (int)i / D;
                int col = (int)i - row * D;
                int gtok = j0 + row;
                float val = 0.0f;
                if (gtok < k_valid && gtok < TILE && col < D) {
                    val = float(k[kv_base + gtok * D + col]);
                }
                kvsmem[i] = val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            thread simdgroup_float8x8 smat = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int kk = 0; kk < 16; kk++) {
                simdgroup_float8x8 kmat;
                simdgroup_load(kmat, (const threadgroup float*)(kvsmem + kk * 8), D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(smat, qfrag[kk], kmat, smat);
            }
            simdgroup_store(smat, sg_scores, KCHUNK);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            float scores[8];
            float cmax = -3.402823466e+38f;
            if (lane < (uint)ROWS) {
                int grow = qrow0 + (int)lane;
                for (int t = 0; t < KCHUNK; t++) {
                    int gtok = j0 + t;
                    float sc = -3.402823466e+38f;
                    if (grow < q_valid && gtok < k_valid && gtok < TILE) {
                        sc = sg_scores[(int)lane * KCHUNK + t] * scale;
                    }
                    scores[t] = sc;
                    cmax = metal::max(cmax, sc);
                }
                float m_new = metal::max(row_m, cmax);
                float alpha = metal::exp(row_m - m_new);
                row_lse *= alpha;
                row_m = m_new;
                sg_alpha[(int)lane] = alpha;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int kk = 0; kk < 16; kk++) {
                scale_rows_simd8x8(acc[kk], sg_alpha, sg_tmp, lane);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = tid; i < (uint)(KCHUNK * D); i += N_SG * SG) {
                int row = (int)i / D;
                int col = (int)i - row * D;
                int gtok = j0 + row;
                float val = 0.0f;
                if (gtok < k_valid && gtok < TILE && col < D) {
                    val = float(v[kv_base + gtok * D + col]);
                }
                kvsmem[i] = val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float local = 0.0f;
            if (lane < (uint)ROWS) {
                int grow = qrow0 + (int)lane;
                for (int t = 0; t < KCHUNK; t++) {
                    float w = 0.0f;
                    if (grow < q_valid) {
                        w = metal::exp(scores[t] - row_m);
                    }
                    sg_scores[(int)lane * KCHUNK + t] = w;
                    local += w;
                }
                row_lse += local;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            thread simdgroup_float8x8 pmat;
            simdgroup_load(pmat, sg_scores, KCHUNK);
            for (int kk = 0; kk < 16; kk++) {
                simdgroup_float8x8 vmat;
                simdgroup_load(vmat, (const threadgroup float*)(kvsmem + kk * 8), D);
                simdgroup_multiply_accumulate(acc[kk], pmat, vmat, acc[kk]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (lane < (uint)ROWS) {
        sg_alpha[(int)lane] = row_lse > 0.0f ? 1.0f / row_lse : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < 16; kk++) {
        scale_rows_simd8x8(acc[kk], sg_alpha, sg_tmp, lane);
        simdgroup_store(acc[kk], sg_tmp, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            int r0 = (int)lane / 8;
            int c0 = (int)lane % 8;
            int g0 = qrow0 + r0;
            int g1 = qrow0 + r0 + 4;
            if (g0 < TILE) {
                out[q_base_tile + g0 * D + kk * 8 + c0] = (g0 < q_valid) ? T(sg_tmp[lane]) : T(0);
            }
            if (g1 < TILE) {
                out[q_base_tile + g1 * D + kk * 8 + c0] = (g1 < q_valid) ? T(sg_tmp[lane + 32]) : T(0);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


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
        from fastvideo.mlx_runtime.minimax_h3_vsa import build_h3_tile_geometry

        geometry = build_h3_tile_geometry((1, ), (4, 4, 4), 64)
        indices = mx.array([[[0, 1]]], dtype=mx.int32)
        counts = mx.array([[2]], dtype=mx.int32)
        # Constructing a CustomKernel does not compile it. Execute every
        # supported dtype once before advertising this backend as available.
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            q = mx.zeros((geometry.padded_length, 1, 128), dtype=dtype)
            probe = _launch_simd_block_sparse(kernel, q, q, q, indices, counts, geometry, 128**-0.5)
            mx.eval(probe)
        _SIMD_KERNEL = kernel
    except Exception as error:  # noqa: BLE001 - keep reference usable
        _SIMD_KERNEL_ERROR = str(error)
        _SIMD_KERNEL = None
    return _SIMD_KERNEL


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
    import mlx.core as mx

    if geometry.tile_elems != 64:
        raise ValueError(f"SIMD-group VSA kernel supports tile 64 only, got {geometry.tile_elems}")
    heads, dim = q_tiled.shape[1], q_tiled.shape[2]
    if dim != 128:
        raise ValueError(f"SIMD-group VSA kernel supports head dim 128, got {dim}")
    q = mx.contiguous(q_tiled.transpose(1, 0, 2))
    k = mx.contiguous(k_tiled.transpose(1, 0, 2))
    v = mx.contiguous(v_tiled.transpose(1, 0, 2))
    meta_i = mx.array(
        [
            dim,
            geometry.tile_elems,
            geometry.padded_length,
            geometry.num_tiles,
            int(block_idx.shape[-1]),
            geometry.num_prefix_tiles,
            geometry.num_video_tiles,
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
        "template": [("T", q.dtype)],
        "grid": (256, geometry.num_video_tiles, heads),
        "threadgroup": (256, 1, 1),
        "output_shapes": [q.shape],
        "output_dtypes": [q.dtype],
    }
    try:
        outputs = kernel(**call_kwargs, init_value=0)
    except TypeError:
        outputs = kernel(**call_kwargs)
    return outputs[0].transpose(1, 0, 2)
