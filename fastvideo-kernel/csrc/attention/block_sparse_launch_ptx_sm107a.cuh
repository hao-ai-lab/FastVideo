#ifndef BLOCK_SPARSE_VSA_LAUNCH_PTX_SM107A_CUH
#define BLOCK_SPARSE_VSA_LAUNCH_PTX_SM107A_CUH

#include "block_sparse_kernel_ptx_sm107a.cuh"

namespace vsa_ptx_sm107a {

struct BlockSparseVsaArgs {
  const __nv_bfloat16* q;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  __nv_bfloat16* o;
  float* lse;

  const int* q2k_idx;
  const int* q2k_num;
  const int* variable_block_sizes;

  int batch;
  int num_heads;
  int seqlen;
  int head_dim;
  int num_blocks;
  int max_kv;
  float sm_scale;
  bool q_prescaled;
};

__host__ inline cudaError_t block_sparse_supported(const BlockSparseVsaArgs& a) {
  if (a.head_dim != HEAD_DIM) return cudaErrorInvalidValue;
  if (a.num_blocks < 2 || a.num_blocks % 2 != 0) return cudaErrorInvalidValue;
  if (a.seqlen != a.num_blocks * BLOCK) return cudaErrorInvalidValue;
  if (a.max_kv < 1 || a.batch < 1 || a.num_heads < 1) return cudaErrorInvalidValue;
  if ((long)a.batch * a.num_heads * a.num_blocks > (1L << 30)) return cudaErrorInvalidValue;
  if (a.q == nullptr || a.k == nullptr || a.v == nullptr || a.o == nullptr) return cudaErrorInvalidValue;
  if (a.q2k_idx == nullptr || a.q2k_num == nullptr) return cudaErrorInvalidValue;
  if (a.variable_block_sizes == nullptr) return cudaErrorInvalidValue;
  return cudaSuccess;
}

namespace detail {

__host__ inline cudaError_t encode_4d(CUtensorMap* m, const void* base, const uint64_t (&gd)[4],
                                      const uint64_t (&gs)[3], const uint32_t (&bd)[4]) {
  const uint32_t es[4] = { 1u, 1u, 1u, 1u };
  const CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(base), gd, gs,
                                            bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

template <bool M_SPLIT, bool Q_PRESCALED>
__host__ inline auto kernel_fn() {
  return &fmha_context_bf16_gen_kernel<64, false, true, false, true, true, true, 8, Exp2Mode::kF32, M_SPLIT,
                                       Q_PRESCALED, !M_SPLIT, false>;
}

}

__host__ inline cudaError_t launch_block_sparse_ptx_sm107a(const BlockSparseVsaArgs& a, cudaStream_t stream) {
  cudaError_t e = block_sparse_supported(a);
  if (e != cudaSuccess) return e;

  const int B = a.batch, H = a.num_heads, S = a.seqlen, hd = a.head_dim;
  const int num_blocks = a.num_blocks, max_kv = a.max_kv;

  int dev = 0, numSM = 0;
  if ((e = cudaGetDevice(&dev)) != cudaSuccess) return e;
  if ((e = cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev)) != cudaSuccess) return e;

  const long dual_q_items = (long)B * H * (num_blocks / 2);
  const double rounds_dual_q = (double)((dual_q_items + numSM - 1) / numSM);
  const double rounds_m_split = 0.5 * (double)((2 * dual_q_items + numSM - 1) / numSM);
  const bool m_split = rounds_m_split <= 0.9 * rounds_dual_q;
  const int packed_mtiles_per_seq = m_split ? num_blocks : num_blocks / 2;
  const int total_work = B * H * packed_mtiles_per_seq;

  CUtensorMap tq_, tk_, tv_, to_;
  {
    const uint64_t gd[4] = { (uint64_t)SUB_COLS_BF16, (uint64_t)((long)B * H), (uint64_t)S, (uint64_t)Q_SUBTILES };
    const uint64_t gs[3] = { (uint64_t)((long)S * hd) * 2u, (uint64_t)hd * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    const uint32_t bq[4] = { (uint32_t)SUB_COLS_BF16, 1u, (uint32_t)M_TILE, (uint32_t)Q_SUBTILES };
    const uint32_t bo[4] = { (uint32_t)SUB_COLS_BF16, 1u, (uint32_t)M_TILE, m_split ? 1u : (uint32_t)Q_SUBTILES };
    if ((e = detail::encode_4d(&tq_, a.q, gd, gs, bq)) != cudaSuccess) return e;
    if ((e = detail::encode_4d(&to_, a.o, gd, gs, bo)) != cudaSuccess) return e;
  }
  {
    const uint64_t gd[4] = { (uint64_t)SUB_COLS_BF16, (uint64_t)S, (uint64_t)(hd / SUB_COLS_BF16),
                             (uint64_t)((long)B * H) };
    const uint64_t gs[3] = { (uint64_t)hd * 2u, (uint64_t)SUB_COLS_BF16 * 2u, (uint64_t)((long)S * hd) * 2u };
    const uint32_t bd[4] = { (uint32_t)SUB_COLS_BF16, (uint32_t)BLOCK, (uint32_t)K_SUBTILES, 1u };
    if ((e = detail::encode_4d(&tk_, a.k, gd, gs, bd)) != cudaSuccess) return e;
    if ((e = detail::encode_4d(&tv_, a.v, gd, gs, bd)) != cudaSuccess) return e;
  }

  const size_t smem =
        (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * KV_RING_SLOT_BYTES
      + (size_t)2 * M_TILE * HEAD_DIM * sizeof(__nv_bfloat16)
      + (2 * NUM_KV_STAGES + 22) * 8
      + (size_t)CLC_STAGES * (2 * 8 + 16) + 16
      + 8
      + (size_t)2 * STAT_REGIONS * STATS * sizeof(float)
      + 4 * 32 * sizeof(int) + 16
      + 256;

  auto kfn = m_split ? (a.q_prescaled ? detail::kernel_fn<true, true>() : detail::kernel_fn<true, false>())
                     : (a.q_prescaled ? detail::kernel_fn<false, true>() : detail::kernel_fn<false, false>());
  {
    int oversized = 0;
    if (cuDeviceGetAttribute(&oversized, CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK, dev) !=
            CUDA_SUCCESS || (size_t)oversized < smem)
      return cudaErrorInvalidConfiguration;
    cudaFunction_t f = nullptr;
    if ((e = cudaGetFuncBySymbol(&f, reinterpret_cast<const void*>(kfn))) != cudaSuccess) return e;
    if (cuFuncSetAttribute(reinterpret_cast<CUfunction>(f), CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE,
                           CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY) != CUDA_SUCCESS)
      return cudaErrorInvalidConfiguration;
  }

  const unsigned long long magic0 = make_magic((unsigned)(H * packed_mtiles_per_seq));
  const unsigned long long magic1 = make_magic((unsigned)H);
  const unsigned long long magic2 = make_magic((unsigned)packed_mtiles_per_seq);
  const float scale_log2 = a.sm_scale * (float)M_LOG2E;

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(total_work, 1, 1);
  cfg.blockDim = dim3(N_WARPS * 32, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 1; attrs[0].val.clusterDim.y = 1; attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  return cudaLaunchKernelEx(&cfg, kfn, tq_, tk_, tv_, to_, S, H, scale_log2, B, num_blocks, packed_mtiles_per_seq,
                            max_kv, magic0, magic1, magic2, a.q2k_idx, a.q2k_num, a.variable_block_sizes, a.lse);
}

}

#endif
