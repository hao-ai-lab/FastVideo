/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Adapted from flashinfer-ai/flashinfer @ 8a94642d83cba0939035868fb6c309b4474a13d6
 * (PR #3820), which in turn adapted ThunderKittens' NVLink all-to-all:
 * https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/parallel/all_to_all/all_to_all.cu
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Fused-transpose Ulysses all-to-all over NVLink. Peer addresses come from
// ncclGetLsaPointer and synchronization from ncclLsaBarrierSession; the index
// math and slab decomposition below are upstream's.
//
// head_dim == 2 layout, uniform sequence splits. With
//   W        = ulysses world size
//   H_local  = H / W
//   S_global = S_local * W
//
//   mode == 0 (input  a2a): [B, S_local, H,       D] -> [B, S_global, H_local, D]
//       y_r[b, j*S_local + s, hl, d] = x_j[b, s, r*H_local + hl, d]
//   mode == 1 (output a2a): [B, S_global, H_local, D] -> [B, S_local, H,       D]
//       out_j[b, s, r*H_local + hl, d] = u_r[b, j*S_local + s, hl, d]
//
// In both modes the unit of transfer is a contiguous (H_local * D) block, so
// every cross-GPU store is fully coalesced.

#ifndef FASTVIDEO_COMM_ULYSSES_ALL_TO_ALL_CUH_
#define FASTVIDEO_COMM_ULYSSES_ALL_TO_ALL_CUH_

#include <cstdint>

#include <nccl.h>
#include <nccl_device.h>

namespace fastvideo {
namespace comm {
namespace ulysses {

constexpr int kUlyssesThreads = 512;
// Deliberately modest: this is link-bandwidth bound, so a small grid leaves the
// rest of the GPU free without costing throughput.
constexpr int kMaxBlocks = 36;

// Shared movement body for the fused-transpose all-to-all (no barriers).
//
// Rows are ordered ((b * W + peer) * S_local + s), so consecutive rows share a
// (batch, peer) and are contiguous on the gather side of the transpose. Each
// block takes a contiguous slab of rows and flattens its threads over the 16B
// units in it, so consecutive lanes address one peer buffer back to back and the
// remote writes coalesce into large bursts rather than (H_local * D)-sized
// scattered ones.
template <typename T, int NGPUS, int MODE>
__device__ __forceinline__ void ulysses_a2a_move(const T* __restrict__ local_in,
                                                 void* const* peer_ptrs, int rank, int B,
                                                 int S_local, int H_local, int D) {
  static_assert(MODE == 0 || MODE == 1, "MODE must be 0 or 1");
  const int W = NGPUS;
  const int64_t H = static_cast<int64_t>(H_local) * W;
  const int64_t S_global = static_cast<int64_t>(S_local) * W;
  const int64_t block_len = static_cast<int64_t>(H_local) * D;  // elements/row
  const int64_t num_rows = static_cast<int64_t>(B) * W * S_local;

  // 16B-vectorized fast path when every row is 16B aligned (the common case:
  // contiguous bf16/fp16/fp32 tensors with block_len * sizeof(T) % 16 == 0).
  using Vec = int4;
  constexpr int kVecBytes = sizeof(Vec);
  const int64_t row_bytes = block_len * static_cast<int64_t>(sizeof(T));
  const bool vec_ok =
      (row_bytes % kVecBytes) == 0 && (reinterpret_cast<uintptr_t>(local_in) % kVecBytes) == 0;

  // Contiguous slab of rows for this block.
  const int64_t rows_per_block = (num_rows + gridDim.x - 1) / gridDim.x;
  const int64_t row_lo = static_cast<int64_t>(blockIdx.x) * rows_per_block;
  int64_t row_hi = row_lo + rows_per_block;
  if (row_hi > num_rows) row_hi = num_rows;
  if (row_lo >= row_hi) return;

  const int tid = threadIdx.x;
  const int nthr = blockDim.x;

  // Decode (b, peer, s) and compute src/dst element offsets for a given row.
  auto offsets = [&](int64_t row, int64_t& src_off, int64_t& dst_off) {
    const int64_t s = row % S_local;
    const int64_t tmp = row / S_local;
    const int64_t peer = tmp % W;
    const int64_t b = tmp / W;
    if constexpr (MODE == 0) {
      src_off = ((b * S_local + s) * H + peer * H_local) * D;
      dst_off = (b * S_global + static_cast<int64_t>(rank) * S_local + s) * block_len;
    } else {
      src_off = (b * S_global + peer * S_local + s) * block_len;
      dst_off = ((b * S_local + s) * H + static_cast<int64_t>(rank) * H_local) * D;
    }
    return peer;
  };

  if (vec_ok) {
    const int64_t units_per_row = row_bytes / kVecBytes;
    const int64_t total_units = (row_hi - row_lo) * units_per_row;
    for (int64_t u = tid; u < total_units; u += nthr) {
      const int64_t local_row = u / units_per_row;
      const int64_t unit = u - local_row * units_per_row;
      const int64_t row = row_lo + local_row;
      int64_t src_off, dst_off;
      const int64_t peer = offsets(row, src_off, dst_off);
      const Vec* s4 = reinterpret_cast<const Vec*>(local_in + src_off);
      Vec* d4 = reinterpret_cast<Vec*>((T*)peer_ptrs[peer] + dst_off);
      d4[unit] = s4[unit];
    }
  } else {
    // Scalar fallback (unaligned / odd shapes).
    for (int64_t row = row_lo; row < row_hi; ++row) {
      int64_t src_off, dst_off;
      const int64_t peer = offsets(row, src_off, dst_off);
      const T* s_ptr = local_in + src_off;
      T* d_ptr = (T*)peer_ptrs[peer] + dst_off;
      for (int64_t i = tid; i < block_len; i += nthr) {
        d_ptr[i] = s_ptr[i];
      }
    }
  }
}

// The transfer mode is a compile-time template parameter so the address math
// specializes and the coalesced slab decomposition is fully unrolled per mode.
template <typename T, int NGPUS, int MODE>
__global__ void __launch_bounds__(kUlyssesThreads, 1)
    ulysses_a2a_kernel(const T* __restrict__ local_in, ncclDevComm devComm, ncclWindow_t win,
                       size_t win_offset, int rank, int B, int S_local, int H_local, int D) {
  // Resolved once; the movement loop would otherwise call this per 16B store.
  void* peer_ptrs[NGPUS];
#pragma unroll
  for (int p = 0; p < NGPUS; ++p) {
    peer_ptrs[p] = ncclGetLsaPointer(win, win_offset, p);
  }

  // Each CTA owns one barrier generation. Sharing index 0 across independently
  // scheduled CTAs races the generation counter and is unsupported by NCCL's
  // device API. The host reserves kMaxBlocks slots when it creates devComm.
  ncclLsaBarrierSession<ncclCoopCta> bar(
      ncclCoopCta(), devComm, ncclTeamTagLsa(), /*index=*/blockIdx.x);

  // Every rank must have entered before anyone writes into peer buffers.
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
  ulysses_a2a_move<T, NGPUS, MODE>(local_in, peer_ptrs, rank, B, S_local, H_local, D);
  // Release-acquire: all peer writes visible before a rank reads its window.
  bar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);
}

}  // namespace ulysses
}  // namespace comm
}  // namespace fastvideo

#endif  // FASTVIDEO_COMM_ULYSSES_ALL_TO_ALL_CUH_
