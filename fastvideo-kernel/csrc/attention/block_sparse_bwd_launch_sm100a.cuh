// block_sparse_bwd_launch_sm100a.cuh -- host surface of the VSA block-sparse backward drop:
// argument struct, workspace sizes, the support predicate and the stream-chained launch
// (preprocess -> order -> main -> postprocess). Tensor maps are encoded per call (no static
// cache: a torch caller hands us fresh pointers every time).
#ifndef BLOCK_SPARSE_VSA_BWD_LAUNCH_SM100A_CUH
#define BLOCK_SPARSE_VSA_BWD_LAUNCH_SM100A_CUH

#include <algorithm>
#include <cmath>
#include "block_sparse_bwd_kernel_sm100a.cuh"

#ifndef VSA_BHSD
#define VSA_BHSD false
#endif
#ifndef VSA_BWD_DQ_F16
#define VSA_BWD_DQ_F16 false
#endif
#ifndef VSA_BWD_USE_CLC
#define VSA_BWD_USE_CLC true
#endif

namespace vsa_bwd_blk64 {

#if VSA_BWD_DQ_F16
using dq_accum_t = uint16_t;
#else
using dq_accum_t = float;
#endif

struct BlockSparseVsaBwdArgs {
  // Activations are bf16, contiguous, [B, H, S, 128] under VSA_BHSD, else [B*S, H, 128].
  // nb below = num_kv_blocks_per_seq = S / 64.

  // Forward operands and results.
  const __nv_bfloat16* q;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  const __nv_bfloat16* o;
  // Gradient of the forward output.
  const __nv_bfloat16* dout;
  // [B, H, S] fp32 log-sum-exp in Triton's M form: max(qk * sm_scale * log2e) + log2(l).
  const float* lse;

  // Sparsity metadata, FastVideo's invert_indices layout.
  // [B*H*nb, max_q_blocks] int32: q blocks selecting each kv block; entries past the count unread.
  const int* k2q_idx;
  // [B*H*nb] int32: valid entries per k2q_idx row (0 allowed).
  const int* k2q_num;
  // [nb] int32: valid kv tokens per block (<= 64); kv rows at or past the count are masked.
  const int* variable_block_sizes;

  // Work order: which (batch, head, kv block) item each CTA processes.
  // [B*H*nb] int32 work id -> item ((b*H + h)*nb + kv); nullptr = the launch computes the order.
  const int* workitem_remap;
  // [B*H*nb] int32; required when workitem_remap is nullptr.
  int* order_workspace;

  // Outputs, inputs' layout; dk/dv rows of unselected kv blocks are zeroed by the preprocess.
  __nv_bfloat16* dq;
  __nv_bfloat16* dk;
  __nv_bfloat16* dv;

  // Scratch, caller-allocated; byte sizes from the block_sparse_bwd_*_bytes helpers below.
  // [B*H*S*128] dq_accum_t, drain-native; preprocess zeroes, main reduce-adds, postprocess reads.
  dq_accum_t* dqaccum;
  // [H*128, B*S] Q^T, written by the preprocess.
  __nv_bfloat16* qt;
  // [H*128, B*S] dO^T, written by the preprocess.
  __nv_bfloat16* dot;
  // [B*H*S] fp32 rowsum(bf16(o) * dout), written by the preprocess.
  float* delta;

  int batch;
  int num_heads;
  // S; a multiple of 128 (the preprocess works in 128-token blocks).
  int seqlen;
  // Must be 128.
  int head_dim;
  // nb = seqlen / 64.
  int num_kv_blocks_per_seq;
  // k2q_idx row stride (FastVideo passes nb).
  int max_q_blocks;
  // Softmax scale; dq and dk carry it, dv does not.
  float sm_scale;
};

__host__ inline size_t block_sparse_bwd_dqaccum_bytes(int batch, int num_heads, int seqlen) {
  return (size_t)batch * num_heads * seqlen * HEAD_DIM * sizeof(dq_accum_t);
}
__host__ inline size_t block_sparse_bwd_order_bytes(int batch, int num_heads,
                                                    int num_kv_blocks_per_seq) {
  return (size_t)batch * num_heads * num_kv_blocks_per_seq * sizeof(int);
}
__host__ inline size_t block_sparse_bwd_transposed_bytes(int batch, int num_heads, int seqlen) {
  return (size_t)num_heads * HEAD_DIM * (size_t)batch * seqlen * sizeof(__nv_bfloat16);
}
__host__ inline size_t block_sparse_bwd_delta_bytes(int batch, int num_heads, int seqlen) {
  return (size_t)batch * num_heads * seqlen * sizeof(float);
}

__host__ inline cudaError_t block_sparse_bwd_supported(const BlockSparseVsaBwdArgs& args) {
  if (args.head_dim != HEAD_DIM) {
    return cudaErrorInvalidValue;
  }
  if (args.num_kv_blocks_per_seq < 1 || args.num_kv_blocks_per_seq % PRE_QBLOCKS != 0) {
    return cudaErrorInvalidValue;
  }
  if (args.seqlen != args.num_kv_blocks_per_seq * BLOCK) {
    return cudaErrorInvalidValue;
  }
  if (args.batch < 1 || args.num_heads < 1 || args.max_q_blocks < 1) {
    return cudaErrorInvalidValue;
  }
  if (!std::isfinite(args.sm_scale)) {
    return cudaErrorInvalidValue;
  }
  if (!args.q || !args.k || !args.v || !args.o || !args.dout || !args.lse) {
    return cudaErrorInvalidValue;
  }
  if (!args.dq || !args.dk || !args.dv) {
    return cudaErrorInvalidValue;
  }
  if (!args.k2q_idx || !args.k2q_num || !args.variable_block_sizes) {
    return cudaErrorInvalidValue;
  }
  if (!args.dqaccum || !args.qt || !args.dot || !args.delta) {
    return cudaErrorInvalidValue;
  }
  // No explicit order: the order kernel needs the workspace and two ints of SMEM per kv block.
  if (!args.workitem_remap &&
      (!args.order_workspace || args.num_kv_blocks_per_seq > ORDER_MAX_BLOCKS)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

// K, V, dK, dV tensor maps, one 64-token x 64-hd box per TMA (two per tile):
//   BSHD: 3D [64 hd, B*S tokens, H*2 hd units], strides {H*128*2, 128} bytes.
//   BHSD: 4D [64 hd, S tokens, 2 hd units, B*H], strides {128*2, 128, S*128*2} bytes.
__host__ inline cudaError_t make_tma_kv_units(CUtensorMap* map, const __nv_bfloat16* ptr, int B,
                                              int H, int S) {
  CUresult r;
  if (VSA_BHSD) {
    uint64_t gd[4] = {(uint64_t)SUB_COLS_BF16, (uint64_t)S, (uint64_t)KV_SUBTILES, (uint64_t)B * H};
    uint64_t gs[3] = {(uint64_t)HEAD_DIM * 2, (uint64_t)SUB_COLS_BYTES, (uint64_t)S * HEAD_DIM * 2};
    uint32_t bd[4] = {(uint32_t)SUB_COLS_BF16, (uint32_t)BLOCK, 1u, 1u};
    uint32_t es[4] = {1u, 1u, 1u, 1u};
    r              = cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<__nv_bfloat16*>(ptr), gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  } else {
    uint64_t gd[3] = {(uint64_t)SUB_COLS_BF16, (uint64_t)B * S, (uint64_t)H * KV_SUBTILES};
    uint64_t gs[2] = {(uint64_t)H * HEAD_DIM * 2, (uint64_t)SUB_COLS_BYTES};
    uint32_t bd[3] = {(uint32_t)SUB_COLS_BF16, (uint32_t)BLOCK, 1u};
    uint32_t es[3] = {1u, 1u, 1u};
    r              = cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<__nv_bfloat16*>(ptr), gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }
  return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
}

// Above the L2-capacity transition the main kernel is DRAM-bound and keeping a fractional
// subset of the repeatedly reduced dQ lines resident pays (~0.6% at 524K/1M tokens); below it the
// policy register costs more than it saves. Threshold scales with accumulator BYTES.
constexpr int CACHE_WAVE_MIN_SEQ_LEN = 524288;

// Below this many kv blocks per sequence (S < 65536) the identity order is as fast as the
// length-binned one and the order kernel's own time is not (fv_perf_log.md 2026-09-04: -9% at
// 4k, -2.8% at 16k), so callers pass a cached identity array as workitem_remap there and leave
// the order kernel to the larger shapes.
constexpr int ORDER_MIN_KV_BLOCKS = 1024;

template <bool DQ_L2_KEEP, bool USE_CLC, bool BHSD>
__host__ inline cudaError_t launch_main(const BlockSparseVsaBwdArgs& args, const int* work_remap,
                                        const CUtensorMap& tk, const CUtensorMap& tv,
                                        const CUtensorMap& tqt, const CUtensorMap& tdot,
                                        const CUtensorMap& tdk, const CUtensorMap& tdv, int sms,
                                        cudaStream_t stream) {
  auto kernel = vsa_bwd_main_kernel<DQ_L2_KEEP, USE_CLC, BHSD, dq_accum_t>;
  cudaError_t e =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
  if (e != cudaSuccess) {
    return e;
  }
  const float scale_log2 = args.sm_scale * 1.4426950408889634f;
  const int B = args.batch, H = args.num_heads, S = args.seqlen;
  const int total = B * H * args.num_kv_blocks_per_seq;
  if constexpr (USE_CLC) {
    // Above the L2 transition, one SM-wide launch at a time keeps each list neighbourhood
    // resident; below it one launch of every item lets CLC steal freely.
    const int chunk        = DQ_L2_KEEP ? std::min(sms, total) : total;
    cudaLaunchConfig_t cfg = {};
    cfg.blockDim           = dim3(N_WARPS * 32, 1, 1);
    cfg.dynamicSmemBytes   = SMEM_TOTAL;
    cfg.stream             = stream;
    cudaLaunchAttribute at[1];
    at[0].id               = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.attrs              = at;
    cfg.numAttrs           = 1;
    for (int base = 0; base < total; base += chunk) {
      const int count = std::min(chunk, total - base);
      cfg.gridDim     = dim3((unsigned)count, 1, 1);
      // A chunk starts at work id `base`: it gets the order's sub-array.
      e = cudaLaunchKernelEx(&cfg, kernel, tk, tv, tqt, tdot, tdk, tdv, args.dqaccum, args.lse,
                             args.delta, args.k2q_idx, args.k2q_num, work_remap + base,
                             args.variable_block_sizes, args.max_q_blocks, B, H, S, scale_log2,
                             args.sm_scale);
      if (e != cudaSuccess) {
        return e;
      }
    }
    return cudaSuccess;
  } else {
    const int grid = std::min(total, sms);
    kernel<<<dim3((unsigned)grid, 1, 1), dim3(N_WARPS * 32, 1, 1), SMEM_TOTAL, stream>>>(
        tk, tv, tqt, tdot, tdk, tdv, args.dqaccum, args.lse, args.delta, args.k2q_idx, args.k2q_num,
        work_remap, args.variable_block_sizes, args.max_q_blocks, B, H, S, scale_log2,
        args.sm_scale);
    return cudaGetLastError();
  }
}

__host__ inline cudaError_t launch_block_sparse_bwd_sm100a(const BlockSparseVsaBwdArgs& args,
                                                           cudaStream_t stream) {
  const cudaError_t supported = block_sparse_bwd_supported(args);
  if (supported != cudaSuccess) {
    return supported;
  }
  const int B = args.batch, H = args.num_heads, S = args.seqlen;
  const long n_tokens = (long)B * S;

  CUtensorMap tk, tv, tqt, tdot, tdk, tdv;
  if (make_tma_kv_units(&tk, args.k, B, H, S) != cudaSuccess) {
    return cudaErrorInvalidValue;
  }
  if (make_tma_kv_units(&tv, args.v, B, H, S) != cudaSuccess) {
    return cudaErrorInvalidValue;
  }
  if (make_tma_kv_units(&tdk, args.dk, B, H, S) != cudaSuccess) {
    return cudaErrorInvalidValue;
  }
  if (make_tma_kv_units(&tdv, args.dv, B, H, S) != cudaSuccess) {
    return cudaErrorInvalidValue;
  }
  // Q^T, dO^T ([H*hd rows, B*S cols], token contiguous): box [hd rows, BLOCK cols] = one q64
  // block per TMA.
  if (make_tma_2d_tiled(&tqt, args.qt, H * HEAD_DIM, (int)n_tokens, HEAD_DIM, BLOCK, 2,
                        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16) != cudaSuccess) {
    return cudaErrorInvalidValue;
  }
  if (make_tma_2d_tiled(&tdot, args.dot, H * HEAD_DIM, (int)n_tokens, HEAD_DIM, BLOCK, 2,
                        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16) != cudaSuccess) {
    return cudaErrorInvalidValue;
  }

  int dev = 0, sms = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) {
    return e;
  }
  e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (e != cudaSuccess) {
    return e;
  }

  vsa_bwd_preprocess_kernel<VSA_BHSD, dq_accum_t>
      <<<dim3((unsigned)(S / PRE_TOKENS), (unsigned)(B * H), 1), dim3(256, 1, 1), 0, stream>>>(
          args.q, args.o, args.dout, args.delta, args.dqaccum, args.qt, args.dot, args.dk, args.dv,
          args.k2q_num, B, H, S);
  e = cudaGetLastError();
  if (e != cudaSuccess) {
    return e;
  }

  const bool keep_dq_l2 = (size_t)S * sizeof(dq_accum_t) >= (size_t)CACHE_WAVE_MIN_SEQ_LEN * 2;

  // Work-item order: explicit (the binding passes its cached identity array below
  // ORDER_MIN_KV_BLOCKS), else computed on device into order_workspace (length bins; the same L2
  // transition that selects DQ_L2_KEEP selects the wider bins plus the midpoint snake).
  const int* work_remap = args.workitem_remap;
  if (work_remap == nullptr) {
    const int order_smem = 2 * args.num_kv_blocks_per_seq * (int)sizeof(int);
    if (order_smem > 48 * 1024) {
      e = cudaFuncSetAttribute(vsa_bwd_order_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               order_smem);
      if (e != cudaSuccess) {
        return e;
      }
    }
    const unsigned item_chunks =
        (unsigned)((args.num_kv_blocks_per_seq + ORDER_THREADS - 1) / ORDER_THREADS);
    vsa_bwd_order_kernel<<<dim3((unsigned)(B * H), item_chunks, 1), dim3(ORDER_THREADS, 1, 1),
                           order_smem, stream>>>(args.k2q_idx, args.k2q_num, args.max_q_blocks,
                                                 args.num_kv_blocks_per_seq, keep_dq_l2 ? 12 : 8,
                                                 keep_dq_l2, args.order_workspace);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
      return e;
    }
    work_remap = args.order_workspace;
  }

  e = keep_dq_l2 ? launch_main<true, VSA_BWD_USE_CLC, VSA_BHSD>(args, work_remap, tk, tv, tqt, tdot,
                                                                tdk, tdv, sms, stream)
                 : launch_main<false, VSA_BWD_USE_CLC, VSA_BHSD>(args, work_remap, tk, tv, tqt,
                                                                 tdot, tdk, tdv, sms, stream);
  if (e != cudaSuccess) {
    return e;
  }

  vsa_bwd_postprocess_kernel<VSA_BHSD, dq_accum_t>
      <<<dim3((unsigned)(S / BLOCK), (unsigned)(B * H), 1), dim3(128, 1, 1), 0, stream>>>(
          args.dqaccum, args.dq, H, S, args.sm_scale);
  return cudaGetLastError();
}

}  // namespace vsa_bwd_blk64

using namespace vsa_bwd_blk64;

#endif  // BLOCK_SPARSE_VSA_BWD_LAUNCH_SM100A_CUH
