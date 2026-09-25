// block_sparse_bwd_launch_sm100a_blk128.cuh -- host surface of the 128-token VSA block-sparse
// backward drop: argument struct, workspace sizes, the support predicate and the stream-chained
// launch (preprocess -> main -> postprocess, each a programmatic dependent launch of the one
// before). Tensor maps are encoded per call (a torch caller hands us fresh pointers every time).
#ifndef BLOCK_SPARSE_VSA_BWD_LAUNCH_SM100A_BLK128_CUH
#define BLOCK_SPARSE_VSA_BWD_LAUNCH_SM100A_BLK128_CUH

#include <cmath>
#include "block_sparse_bwd_kernel_sm100a_blk128.cuh"

#ifndef VSA_BHSD
#define VSA_BHSD false
#endif

namespace vsa_bwd_blk128 {

using dq_accum_t = float;

struct BlockSparseVsaBwdArgs {
  // Activations are bf16, contiguous, [B, H, S, 128] under VSA_BHSD, else [B*S, H, 128].
  // nb below = num_kv_blocks_per_seq = S / 128.

  // Forward operands and results.
  const __nv_bfloat16* q;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  const __nv_bfloat16* o;
  // Gradient of the forward output.
  const __nv_bfloat16* dout;
  // [B, H, S] fp32 log-sum-exp in Triton's M form: max(qk * sm_scale * log2e) + log2(l).
  const float* lse;

  // Sparsity metadata, FastVideo's invert_indices layout, in 128-token blocks.
  // [B*H*nb, max_q_blocks] int32: q blocks selecting each kv block; entries past the count unread.
  const int* k2q_idx;
  // [B*H*nb] int32: valid entries per k2q_idx row (0 allowed: the kernel writes zero dk/dv rows).
  const int* k2q_num;
  // [nb] int32: valid kv tokens per block (<= 128); kv rows at or past the count are masked.
  const int* variable_block_sizes;

  // Outputs, inputs' layout; every row is written, including the rows of unselected kv blocks.
  __nv_bfloat16* dq;
  __nv_bfloat16* dk;
  __nv_bfloat16* dv;

  // Scratch, caller-allocated; byte sizes from the block_sparse_bwd_*_bytes helpers below.
  // [B*H*S*128] fp32, drain-native; preprocess zeroes, main reduce-adds, postprocess reads.
  dq_accum_t* dqaccum;
  // [B*H*S] fp32 rowsum(bf16(o) * dout), written by the preprocess.
  float* delta;

  int batch;
  int num_heads;
  // S = nb * 128.
  int seqlen;
  // Must be 128.
  int head_dim;
  // nb = seqlen / 128.
  int num_kv_blocks_per_seq;
  // k2q_idx row stride (FastVideo passes nb).
  int max_q_blocks;
  // Softmax scale; dq and dk carry it, dv does not.
  float sm_scale;
};

__host__ inline size_t block_sparse_bwd_dqaccum_bytes(int batch, int num_heads, int seqlen) {
  return (size_t)batch * num_heads * seqlen * HEAD_DIM * sizeof(dq_accum_t);
}
__host__ inline size_t block_sparse_bwd_delta_bytes(int batch, int num_heads, int seqlen) {
  return (size_t)batch * num_heads * seqlen * sizeof(float);
}

__host__ inline cudaError_t block_sparse_bwd_supported(const BlockSparseVsaBwdArgs& args) {
  if (args.head_dim != HEAD_DIM) {
    return cudaErrorInvalidValue;
  }
  if (args.num_kv_blocks_per_seq < 1) {
    return cudaErrorInvalidValue;
  }
  if (args.seqlen != args.num_kv_blocks_per_seq * BLOCK) {
    return cudaErrorInvalidValue;
  }
  if (args.batch < 1 || args.num_heads < 1 || args.max_q_blocks < 1) {
    return cudaErrorInvalidValue;
  }
  // The main grid is (kv block, batch * head): gridDim.y is limited to 65535.
  if ((long)args.batch * args.num_heads > 65535) {
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
  if (!args.dqaccum || !args.delta) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

// Q, K, V, dO, dK, dV tensor maps, one box_tokens x 64-hd box per TMA (two per tile):
//   BSHD: 3D [64 hd, B*S tokens, H*2 hd units], strides {H*128*2, 128} bytes.
//   BHSD: 4D [64 hd, S tokens, 2 hd units, B*H], strides {128*2, 128, S*128*2} bytes.
__host__ inline cudaError_t make_tma_tile_units(CUtensorMap* map, const __nv_bfloat16* ptr, int B,
                                                int H, int S, int box_tokens) {
  CUresult r;
  if (VSA_BHSD) {
    uint64_t gd[4] = {(uint64_t)SUB_COLS_BF16, (uint64_t)S, (uint64_t)HD_SUBTILES, (uint64_t)B * H};
    uint64_t gs[3] = {(uint64_t)HEAD_DIM * 2, (uint64_t)SUB_COLS_BYTES, (uint64_t)S * HEAD_DIM * 2};
    uint32_t bd[4] = {(uint32_t)SUB_COLS_BF16, (uint32_t)box_tokens, 1u, 1u};
    uint32_t es[4] = {1u, 1u, 1u, 1u};
    r              = cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<__nv_bfloat16*>(ptr), gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  } else {
    uint64_t gd[3] = {(uint64_t)SUB_COLS_BF16, (uint64_t)B * S, (uint64_t)H * HD_SUBTILES};
    uint64_t gs[2] = {(uint64_t)H * HEAD_DIM * 2, (uint64_t)SUB_COLS_BYTES};
    uint32_t bd[3] = {(uint32_t)SUB_COLS_BF16, (uint32_t)box_tokens, 1u};
    uint32_t es[3] = {1u, 1u, 1u};
    r              = cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<__nv_bfloat16*>(ptr), gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }
  return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
}

// Every launch carries the programmatic-stream-serialization attribute when the kernels are
// built with KERNEL_PDL (their griddepcontrol.wait guards the predecessor's data).
__host__ inline cudaLaunchConfig_t pdl_launch_config(dim3 grid, dim3 block, size_t smem,
                                                     cudaStream_t stream, cudaLaunchAttribute* at) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim            = grid;
  cfg.blockDim           = block;
  cfg.dynamicSmemBytes   = smem;
  cfg.stream             = stream;
  at->id                 = cudaLaunchAttributeProgrammaticStreamSerialization;
  at->val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs                                      = at;
  cfg.numAttrs                                   = KERNEL_PDL ? 1 : 0;
  return cfg;
}

__host__ inline cudaError_t launch_block_sparse_bwd_sm100a(const BlockSparseVsaBwdArgs& args,
                                                           cudaStream_t stream) {
  const cudaError_t supported = block_sparse_bwd_supported(args);
  if (supported != cudaSuccess) {
    return supported;
  }
  const int B = args.batch, H = args.num_heads, S = args.seqlen;

  CUtensorMap tq, tk, tv, tdo, tdk, tdv;
  if (make_tma_tile_units(&tq, args.q, B, H, S, Q_TILE) != cudaSuccess ||
      make_tma_tile_units(&tk, args.k, B, H, S, KV_TILE) != cudaSuccess ||
      make_tma_tile_units(&tv, args.v, B, H, S, KV_TILE) != cudaSuccess ||
      make_tma_tile_units(&tdo, args.dout, B, H, S, Q_TILE) != cudaSuccess ||
      make_tma_tile_units(&tdk, args.dk, B, H, S, KV_TILE) != cudaSuccess ||
      make_tma_tile_units(&tdv, args.dv, B, H, S, KV_TILE) != cudaSuccess) {
    return cudaErrorInvalidValue;
  }

  auto main_kernel        = vsa_bwd_main_kernel<VSA_BHSD, Sched::NON_PERSISTENT>;
  auto postprocess_kernel = vsa_bwd_postprocess_kernel<VSA_BHSD>;
  constexpr int POST_SMEM_BYTES = DQ::DQ_BLOCK_ELEMS * (int)sizeof(float);
  cudaError_t e =
      cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
  if (e != cudaSuccess) {
    return e;
  }
  e = cudaFuncSetAttribute(postprocess_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           POST_SMEM_BYTES);
  if (e != cudaSuccess) {
    return e;
  }

  // Preprocess: dqaccum zero + Delta, one CTA per (q block, batch * head).
  cudaLaunchAttribute pre_at[1];
  cudaLaunchConfig_t pre_cfg = pdl_launch_config(
      dim3((unsigned)(S / Q_TILE), (unsigned)(B * H), 1), dim3(256, 1, 1), 0, stream, pre_at);
  e = cudaLaunchKernelEx(&pre_cfg, vsa_bwd_preprocess_kernel<VSA_BHSD>, args.o, args.dout,
                         args.delta, args.dqaccum, H, S);
  if (e != cudaSuccess) {
    return e;
  }

  // Main: one CTA per (kv block, batch * head).
  const float scale_log2 = args.sm_scale * 1.4426950408889634f;
  cudaLaunchAttribute main_at[2];
  cudaLaunchConfig_t main_cfg = {};
  main_cfg.gridDim            = dim3((unsigned)args.num_kv_blocks_per_seq, (unsigned)(B * H), 1);
  main_cfg.blockDim           = dim3(N_WARPS * 32, 1, 1);
  main_cfg.dynamicSmemBytes   = SMEM_TOTAL;
  main_cfg.stream             = stream;
  main_at[0].id               = cudaLaunchAttributeClusterDimension;
  main_at[0].val.clusterDim.x = 1;
  main_at[0].val.clusterDim.y = 1;
  main_at[0].val.clusterDim.z = 1;
  main_at[1].id               = cudaLaunchAttributeProgrammaticStreamSerialization;
  main_at[1].val.programmaticStreamSerializationAllowed = 1;
  main_cfg.attrs                                        = main_at;
  main_cfg.numAttrs                                     = KERNEL_PDL ? 2 : 1;
  const int* workitem_remap                             = nullptr;
  e = cudaLaunchKernelEx(&main_cfg, main_kernel, tq, tk, tv, tdo, tdk, tdv, args.dqaccum, args.lse,
                         args.delta, args.k2q_idx, args.k2q_num, args.max_q_blocks, workitem_remap,
                         args.variable_block_sizes, B, H, S, scale_log2, args.sm_scale);
  if (e != cudaSuccess) {
    return e;
  }

  // Postprocess: drain-native dqaccum -> bf16 dq, one CTA per (q block, batch * head).
  cudaLaunchAttribute post_at[1];
  cudaLaunchConfig_t post_cfg =
      pdl_launch_config(dim3((unsigned)(S / Q_TILE), (unsigned)(B * H), 1), dim3(128, 1, 1),
                        POST_SMEM_BYTES, stream, post_at);
  return cudaLaunchKernelEx(&post_cfg, postprocess_kernel, args.dqaccum, args.dq, H, S,
                            args.sm_scale);
}

}  // namespace vsa_bwd_blk128

using namespace vsa_bwd_blk128;

#endif  // BLOCK_SPARSE_VSA_BWD_LAUNCH_SM100A_BLK128_CUH
