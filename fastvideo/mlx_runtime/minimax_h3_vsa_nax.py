# SPDX-License-Identifier: Apache-2.0 AND MIT
# mypy: disable-error-code=no-untyped-call
#
# This file contains a kernel adapted from MLX. Its MIT notice follows.
# MIT License
#
# Copyright © 2023 Apple Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Opt-in M5 block-sparse TensorOps attention using installed MLX NAX helpers.

Only selected blocks and their valid rows are visited, without K/V duplication
or a dense mask. Prefix and padded query outputs remain zero. Online softmax
and matrix accumulation can differ numerically from SIMD and fused SDPA.

This backend depends on private headers from the installed MLX package, tested
with MLX 0.32.4 and the Metal 4 SDK. Only the validated header hashes are accepted.
Missing headers, incompatible APIs and
unsupported hardware raise explicitly; no fallback changes the implementation.
Importing this module performs no MLX operations. Header copyrights are retained
and get_nax_provenance() provides hashes for reproducible benchmark receipts.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from pathlib import Path

NAX_TILE_SIZES = (64, 128)
NAX_HEAD_DIM = 128
DEFAULT_QUERY_ROWS = 32
_NAX_HEADER = "mlx/backend/metal/kernels/steel/attn/nax.h"
# Private fragment layouts can change without renaming their APIs. Accept only
# the headers used by the numerical and full-generation validation on MLX 0.32.4.
_VALIDATED_HEADER_SHA256 = {
    _NAX_HEADER:
    "fb10bcae44095e9fed5c81213d03527dd5fe5ac11be11ec7784bd357c6f317ec",
    "mlx/backend/metal/kernels/steel/defines.h":
    "b03cea6a7d5cfe814e6838d8536e2aacdcafd2744cb408e954fda336ce21759a",
    "mlx/backend/metal/kernels/steel/utils/integral_constant.h":
    "886eccaac4f8f00eab5a10551fdb5315df7ab7462aba13d69bb95765c1a4735e",
    "mlx/backend/metal/kernels/steel/utils/type_traits.h":
    "4766c5a3809d3cf7740e63acc5703ef76b6b4e85d8baf16602c9d517a2fd426c",
}

_OPERATORS = r"""
namespace h3_sparse_nax {
struct MaxOp {
  template <typename V>
  METAL_FUNC static constexpr V apply(V x, V y) { return metal::max(x, y); }
};
struct SumOp {
  template <typename V>
  METAL_FUNC static constexpr V apply(V x, V y) { return x + y; }
};
struct MulOp {
  template <typename V>
  METAL_FUNC static constexpr V apply(V x, V y) { return x * y; }
};
struct ExpSubOp {
  template <typename V>
  METAL_FUNC static constexpr V apply(V x, V y) { return metal::fast::exp2(x - y); }
};
}
"""

# Adapted from MLX steel_attention_nax.h (Copyright 2024-26 Apple Inc.).
# The fragment math/online softmax follows that implementation; this launch
# addresses the existing H3 tiled layout and walks selected key blocks directly.
_SOURCE = r"""
    using namespace mlx::steel;
    using namespace h3_sparse_nax;
    constexpr int D = 128;
    constexpr int BK = 32;
    constexpr int TK = BK / 16;
    constexpr int TD = D / 16;
    constexpr float NEG = -3.402823466e38f;

    const int qsub = int(threadgroup_position_in_grid.x);
    const int query_tile = int(threadgroup_position_in_grid.y);
    const int head = int(threadgroup_position_in_grid.z);
    const int sid = int(simdgroup_index_in_threadgroup);
    const int k_max = meta_i[0];
    const int n_prefix = meta_i[1];
    const int n_video = meta_i[2];
    const int heads = meta_i[3];
    const int qt = n_prefix + query_tile;
    const int q_valid = vbs[qt];
    const int row_base = qsub * BQ;
    const int nsel = block_num[head * n_video + query_tile];

    // Uniform across the threadgroup: later barriers cannot be skipped by only
    // some SIMD groups. Inactive SIMD groups use guarded loads and stores.
    if (row_base >= q_valid || nsel <= 0) { return; }

    const int row_stride = heads * D;
    const int qrow = row_base + sid * 16;
    const short q_remaining = short(q_valid - qrow);
    const device T* Q = q + (long(qt * TILE + qrow) * heads + head) * D;
    device T* O = out + (long(qt * TILE + qrow) * heads + head) * D;
    const float scale2 = meta_f[0] * 1.44269504088896341f;

    using otile_t = NAXTile<float, 1, TD>;
    using stile_t = NAXTile<float, 1, TK>;
    otile_t Otile;
    Otile.clear();
    constexpr short kRowsPT = otile_t::kRowsPerThread;
    metal::vec<float, kRowsPT> max_score;
    metal::vec<float, kRowsPT> sum_score{0};
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) { max_score[i] = NEG; }
    const short sn = otile_t::NAXFrag_t::get_coord().x;

    for (int selected = 0; selected < nsel; ++selected) {
      const int kt = block_idx[(head * n_video + query_tile) * k_max + selected];
      const int k_valid = vbs[kt];
      const long kv_offset = (long(kt) * TILE * heads + head) * D;
      for (int krow = 0; krow < k_valid; krow += BK) {
        const short k_remaining = short(metal::min(BK, k_valid - krow));
        const device T* K = k + kv_offset + long(krow) * row_stride;
        const device T* V = v + kv_offset + long(krow) * row_stride;
        stile_t Stile;
        Stile.clear();

        // QK: retain MLX's four-way partial unroll for the 128-wide head.
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik += 2) {
#pragma clang loop unroll_count(4)
          for (short id = 0; id < TD; ++id) {
            NAXTile<T, 1, 1> Qtile;
            NAXTile<T, 2, 1> Ktile;
            if (q_remaining < 16) {
              Qtile.load_rows(Q + id * 16, row_stride, q_remaining);
            } else {
              Qtile.load(Q + id * 16, row_stride);
            }
            if (k_remaining < BK) {
              Ktile.load_rows(K + ik * 16 * row_stride + id * 16,
                              row_stride, k_remaining - ik * 16);
            } else {
              Ktile.load(K + ik * 16 * row_stride + id * 16, row_stride);
            }
            stile_t::NAXFrag_t::mma(
                Stile.frag_at(0, ik), Stile.frag_at(0, ik + 1),
                Qtile.frag_at(0, 0), metal::false_type{},
                Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < stile_t::kElemsPerTile; ++i) {
          Stile.elems()[i] *= scale2;
        }
        if (k_remaining < BK) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ++ik) {
            thread auto& frag = Stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < stile_t::kFragThrRows; ++i) {
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < stile_t::kFragThrCols; ++j) {
                if (ik * 16 + sn + j >= k_remaining) {
                  frag[i * stile_t::kFragThrCols + j] = NEG;
                }
              }
            }
          }
        }

        metal::vec<float, kRowsPT> new_max = max_score;
        metal::vec<float, kRowsPT> factor;
        Stile.template row_reduce<MaxOp>(new_max);
        Stile.template row_bin_op<ExpSubOp>(new_max);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
          factor[i] = metal::fast::exp2(max_score[i] - new_max[i]);
          max_score[i] = new_max[i];
          sum_score[i] *= factor[i];
        }
        Stile.template row_reduce<SumOp>(sum_score);
        Otile.template row_bin_op<MulOp>(factor);
        simdgroup_barrier(mem_flags::mem_none);

        // PV retains float32 probabilities, matching MLX's NAX SDPA kernel.
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id += 2) {
          if (id == 4) { threadgroup_barrier(mem_flags::mem_none); }
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ++ik) {
            NAXTile<T, 1, 2> Vtile;
            if (k_remaining < BK) {
              Vtile.load_rows(V + ik * 16 * row_stride + id * 16,
                              row_stride, k_remaining - ik * 16);
            } else {
              Vtile.load(V + ik * 16 * row_stride + id * 16, row_stride);
            }
            otile_t::NAXFrag_t::mma(
                Otile.frag_at(0, id), Otile.frag_at(0, id + 1),
                Stile.frag_at(0, ik), metal::false_type{},
                Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_none);
    metal::vec<float, kRowsPT> rcp;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      rcp[i] = sum_score[i] > 0.f ? 1.f / sum_score[i] : 0.f;
    }
    Otile.template row_bin_op<MulOp>(rcp);
    if (q_remaining > 0) {
      if (q_remaining < 16) { Otile.store_rows(O, row_stride, q_remaining); }
      else { Otile.store(O, row_stride); }
    }
"""


def _mlx_include_root() -> Path:
    """Resolve headers beside the imported package, including namespace installs."""
    import mlx

    for package_path in mlx.__path__:
        include_root = Path(package_path) / "include"
        if (include_root / _NAX_HEADER).is_file():
            return include_root
    raise RuntimeError("NAX VSA requires installed MLX NAX headers (tested with MLX 0.32.4); "
                       "the mlx package has no include/mlx/backend/metal/kernels/steel/attn/nax.h")


@lru_cache(maxsize=1)
def _header():
    """Inline quoted MLX includes; system Metal/MPP includes stay intact."""
    include_root = _mlx_include_root()
    visited = {}

    def expand(relative):
        if relative in visited:
            return f"// Already included: {relative}\n"
        path = include_root / relative
        try:
            original_bytes = path.read_bytes()
        except OSError:
            raise RuntimeError(f"NAX VSA cannot read required installed MLX header: {relative}") from None
        digest = hashlib.sha256(original_bytes).hexdigest()
        if digest != _VALIDATED_HEADER_SHA256.get(relative):
            raise RuntimeError(f"Unvalidated MLX private header: {relative}; NAX VSA requires the headers "
                               "validated with MLX 0.32.4. Revalidate this backend before accepting new headers.")
        original = original_bytes.decode("utf-8")
        visited[relative] = digest
        content = re.sub(r"^\s*#pragma once\s*$", "", original, flags=re.MULTILINE)
        content = re.sub(r'^\s*#include "(mlx/[^"\n]+)"\s*$',
                         lambda match: expand(match.group(1)),
                         content,
                         flags=re.MULTILINE)
        if relative == _NAX_HEADER:
            old = "transpose_b,\n        true,\n        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate"
            expected_api = ("struct BaseNAXFrag", "struct NAXTile", "get_coord()", "row_reduce(", "row_bin_op(",
                            "load_rows(", "store_rows(")
            if content.count(old) != 2 or any(marker not in content for marker in expected_api):
                raise RuntimeError("Installed MLX NAX private API changed; this backend was tested with MLX 0.32.4 "
                                   "and needs compatibility review")
        return f"\n// Begin installed MLX header: {relative}\n{content}\n// End: {relative}\n"

    header = "#ifndef METAL_FUNC\n#define METAL_FUNC inline __attribute__((always_inline))\n#endif\n"
    header += expand(_NAX_HEADER)
    header += _OPERATORS
    return header, visited


def get_nax_provenance():
    """CPU-only metadata, containing no machine-specific absolute paths."""
    header, dependencies = _header()
    return {
        "kernel": "h3-true-sparse-nax-v1",
        "relaxed_precision": True,
        "math_mode": "safe",
        "key_rows": 32,
        "source_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest(),
        "expanded_header_sha256": hashlib.sha256(header.encode()).hexdigest(),
        "installed_header_sha256": dependencies
    }


def _check_nax_device() -> None:
    """The TensorOps backend is deliberately restricted to validated M5 GPUs."""
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("NAX VSA requires an Apple M5 GPU and Metal 4")
    device_name = str(mx.device_info().get("device_name", ""))
    if re.match(r"^Apple M5(?:\s|$)", device_name) is None:
        raise RuntimeError(f"NAX VSA supports Apple M5 GPUs only, got {device_name or 'unknown GPU'}")


@lru_cache(maxsize=1)
def _nax_kernel():
    import mlx.core as mx

    _check_nax_device()
    header, _ = _header()
    if not hasattr(mx.fast, "metal_kernel"):
        raise RuntimeError("NAX VSA requires mx.fast.metal_kernel with safe math compile_options")
    try:
        return mx.fast.metal_kernel(name="h3_vsa_sparse_nax",
                                    input_names=["q", "k", "v", "block_idx", "block_num", "vbs", "meta_i", "meta_f"],
                                    output_names=["out"],
                                    source=_SOURCE,
                                    header=header,
                                    compile_options={"math_mode": "safe"})
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("NAX VSA kernel construction failed; this backend requires compatible MLX NAX headers "
                           "and Metal 4 TensorOps support") from error


def nax_block_sparse(q_tiled,
                     k_tiled,
                     v_tiled,
                     block_idx,
                     block_num,
                     geometry,
                     scale: float,
                     *,
                     query_rows: int = DEFAULT_QUERY_ROWS):
    """Same [S_pad,H,128] contract as the existing H3 sparse implementations.

    query_rows controls SIMD groups per threadgroup.
    Active indices must be valid and unique, as emitted by the H3 router. Values
    in inactive index slots are never read. Prefix/padding outputs are zero.
    """
    import mlx.core as mx

    if query_rows not in (16, 32, 64, 128):
        raise ValueError("query_rows must be 16, 32, 64 or 128")
    if q_tiled.ndim != 3 or q_tiled.shape != k_tiled.shape or q_tiled.shape != v_tiled.shape:
        raise ValueError("NAX sparse attention requires matching [S_pad,H,D] Q/K/V")
    padded, heads, dim = q_tiled.shape
    tile = geometry.tile_elems
    if tile not in NAX_TILE_SIZES or dim != NAX_HEAD_DIM or query_rows > tile:
        raise ValueError("NAX sparse attention requires tile 64/128, head dim 128, query_rows <= tile")
    if padded != geometry.padded_length or heads < 1:
        raise ValueError("Q/K/V shape does not match geometry")
    if q_tiled.dtype not in (mx.float32, mx.float16, mx.bfloat16):
        raise ValueError("NAX sparse attention supports float32, float16 or bfloat16")
    if q_tiled.dtype != k_tiled.dtype or q_tiled.dtype != v_tiled.dtype:
        raise ValueError("Q/K/V dtypes must match")
    if (block_idx.ndim != 3 or block_idx.shape[:2] != (heads, geometry.num_video_tiles)
            or block_num.shape != (heads, geometry.num_video_tiles)):
        raise ValueError("Routing shapes do not match head/tile geometry")
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if geometry.num_video_tiles == 0:
        return mx.zeros_like(q_tiled)
    threads = (query_rows // 16) * 32
    result, = _nax_kernel()(inputs=[
        mx.contiguous(q_tiled),
        mx.contiguous(k_tiled),
        mx.contiguous(v_tiled),
        mx.contiguous(block_idx.astype(mx.int32)),
        mx.contiguous(block_num.astype(mx.int32)),
        mx.array(geometry.variable_block_sizes, dtype=mx.int32),
        mx.array([block_idx.shape[-1], geometry.num_prefix_tiles, geometry.num_video_tiles, heads], dtype=mx.int32),
        mx.array([scale], dtype=mx.float32)
    ],
                            template=[("T", q_tiled.dtype), ("TILE", tile), ("BQ", query_rows)],
                            grid=(threads * (tile // query_rows), geometry.num_video_tiles, heads),
                            threadgroup=(threads, 1, 1),
                            output_shapes=[q_tiled.shape],
                            output_dtypes=[q_tiled.dtype],
                            init_value=0)
    return result
