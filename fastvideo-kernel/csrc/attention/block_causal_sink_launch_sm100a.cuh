// block_causal_sink_launch_sm100a.cuh -- callable entry point for the block-causal + sink +
// sliding-window FMHA kernel (sm_100a, forward only). No env vars, no allocation, no host-side
// reordering: everything comes from the caller, so a torch extension and our benchmark harness
// share it.
//
// LAYOUT. Q/K/V/O are [B, H, L, D] with head_dim CONTIGUOUS. Only head_dim contiguity is
// required -- the outer strides are read from the caller, so a torch tensor that has merely
// been permuted (no .contiguous()) is consumed as-is with no copy. V is read MN-major, so it
// needs no transpose either.
//
// SCALE. sm_scale is the caller's, matching FastVideo's qk_scale = sm_scale * LOG2E. Getting
// it wrong corrupts both O and lse and does so plausibly, so it is never derived here.
#pragma once
#include "block_causal_sink_kernel_sm100a.cuh"

struct BlockCausalSinkArgs {
  const __nv_bfloat16* q = nullptr;       // [B, H_q,  L, D]
  const __nv_bfloat16* k = nullptr;       // [B, H_kv, L, D]
  const __nv_bfloat16* v = nullptr;       // [B, H_kv, L, D]  (MN-major, NOT transposed)
  const __nv_bfloat16* q_sink = nullptr;  // [B, H_q,  L, D], required iff has_delta
  __nv_bfloat16* o = nullptr;             // [B, H_q,  L, D]
  float* lse = nullptr;                   // [B*H_q, L] fp32; nullptr skips the store

  int batch = 0, seqlen = 0, num_q_heads = 0, num_kv_heads = 0, head_dim = 128;

  int tokens_per_block = 0;       // num_frame_per_block * frame_seqlen
  int sink_tokens = 0;            // sink_size          * frame_seqlen
  int rolling_window_tokens = 0;  // local_attn_size    * frame_seqlen
  float sm_scale = 0.f;           // 0 -> 1/sqrt(head_dim)
  bool has_delta = false;         // relativistic sink RoPE correction
};

// Returns cudaErrorInvalidValue for an unsupported configuration rather than computing
// silently-wrong results.
inline cudaError_t block_causal_sink_supported(const BlockCausalSinkArgs& a) {
  if (a.head_dim != HEAD_DIM) return cudaErrorInvalidValue;
  if (a.num_q_heads % a.num_kv_heads) return cudaErrorInvalidValue;
  if (a.tokens_per_block <= 0) return cudaErrorInvalidValue;
  if (a.seqlen % a.tokens_per_block) return cudaErrorInvalidValue;  // no partial last block
  if (a.sink_tokens - a.tokens_per_block > K_TILE)
    return cudaErrorInvalidValue;  // large-sink regime
  if (a.has_delta && a.q_sink == nullptr) return cudaErrorInvalidValue;
  return cudaSuccess;
}

template <bool MHA, bool LPT, bool HAS_SINK_ROPE_DELTA>
static cudaError_t launch_block_causal_sink_impl(const BlockCausalSinkArgs& a,
                                                 cudaStream_t stream) {
  const int gqa_group_size = a.num_q_heads / a.num_kv_heads;  // q-heads per kv-head
  const int q_tokens_per_mtile = M_TILE / gqa_group_size;     // q-tokens per M-tile
  const int q_tokens_per_cta = 2 * q_tokens_per_mtile;        // q-tokens per CTA (2 M-tiles)
  CUtensorMap tmap_q, tmap_k, tmap_v_t, tmap_o, tmap_q_sink;
  {
    uint64_t global_dims[4] = {(uint64_t)a.head_dim, (uint64_t)a.num_q_heads, (uint64_t)a.seqlen,
                               (uint64_t)a.batch};
    // FV [B,H,L,D]: head strides by a whole sequence, token by one head_dim row. Same dims, same
    // box, same coordinates as the old [B,L,H,D] map -- only these two strides swap roles.
    uint64_t global_strides[3] = {(uint64_t)a.seqlen * a.head_dim * 2u, (uint64_t)a.head_dim * 2u,
                                  (uint64_t)a.num_q_heads * a.seqlen * a.head_dim * 2u};
    uint32_t box_dims[4] = {(uint32_t)SUB_COLS_BF16, (uint32_t)gqa_group_size,
                            (uint32_t)q_tokens_per_mtile, 1u};
    uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
    CUresult r = cuTensorMapEncodeTiled(
        &tmap_q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<__nv_bfloat16*>(a.q), global_dims,
        global_strides, box_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
    r = cuTensorMapEncodeTiled(&tmap_o, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
                               const_cast<__nv_bfloat16*>(a.o), global_dims, global_strides,
                               box_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
    // q_sink map: identical 4D layout, base = a.q_sink (or a.q when !has_delta -- unused
    // placeholder).
    r = cuTensorMapEncodeTiled(
        &tmap_q_sink, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        const_cast<__nv_bfloat16*>(a.has_delta ? a.q_sink : a.q), global_dims, global_strides,
        box_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // K: ONE 3D TMA copy folds the 2 head-dim swizzle atoms (HEAD_DIM = 2 x SUB_COLS_BF16) into the
  // box (vs looping 2 x 2D copies). dims [atom-col SUB_COLS_BF16, token ((long)a.batch * a.seqlen),
  // atom (num_kv_heads*head_dim)/SUB_COLS_BF16]; box [SUB_COLS_BF16, K_TILE, K_SUBTILES]; strides
  // token=(num_kv_heads*head_dim)*2B, atom=SUB_COLS_BF16*2B. The box dim order (atom outermost)
  // reproduces the atom-outer smem layout the MMA reads (atom0 then atom1).
  {
    // FV [B,H,L,D]: head and sample are adjacent with stride L*D (sample stride = HK * L*D), so the
    // two fold into ONE dim indexed sample*HK + h_kv. Atom stays the outermost box dim to keep the
    // atom-outer smem order the MMA reads.
    uint64_t global_dims[4] = {(uint64_t)SUB_COLS_BF16, (uint64_t)a.seqlen,
                               (uint64_t)(a.head_dim / SUB_COLS_BF16),
                               (uint64_t)((long)a.batch * a.num_kv_heads)};
    uint64_t global_strides[3] = {(uint64_t)a.head_dim * 2u, (uint64_t)SUB_COLS_BF16 * 2u,
                                  (uint64_t)a.seqlen * a.head_dim * 2u};
    uint32_t box_dims[4] = {(uint32_t)SUB_COLS_BF16, (uint32_t)K_TILE, (uint32_t)K_SUBTILES, 1u};
    uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
    CUresult r = cuTensorMapEncodeTiled(
        &tmap_k, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<__nv_bfloat16*>(a.k), global_dims,
        global_strides, box_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  {  // FV SPIKE: V map is now byte-for-byte the K map, just over dV.
    uint64_t global_dims[4] = {(uint64_t)SUB_COLS_BF16, (uint64_t)a.seqlen,
                               (uint64_t)(a.head_dim / SUB_COLS_BF16),
                               (uint64_t)((long)a.batch * a.num_kv_heads)};
    uint64_t global_strides[3] = {(uint64_t)a.head_dim * 2u, (uint64_t)SUB_COLS_BF16 * 2u,
                                  (uint64_t)a.seqlen * a.head_dim * 2u};
    uint32_t box_dims[4] = {(uint32_t)SUB_COLS_BF16, (uint32_t)K_TILE, (uint32_t)K_SUBTILES, 1u};
    uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
    CUresult r = cuTensorMapEncodeTiled(
        &tmap_v_t, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<__nv_bfloat16*>(a.v),
        global_dims, global_strides, box_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }

  // ---- shared memory budget ----
  const int packed_mtiles_per_seq =
      (a.seqlen + q_tokens_per_cta - 1) / q_tokens_per_cta;  // packed-M tiles per (sample, kv-head)
  // FastDivmod magics for decode_workitem's divides:
  //   magic0 = mtiles_per_sample (workitem_id -> sample), magic1 = mtiles_per_seq (rr -> kv_head),
  //   magic2 = num_kv_heads (swizzle path + non-Q_RASTER rr -> tile_index).
  const unsigned long long magic0 = make_magic((unsigned)(packed_mtiles_per_seq * a.num_kv_heads));
  const unsigned long long magic1 = make_magic((unsigned)packed_mtiles_per_seq);
  const unsigned long long magic2 = make_magic((unsigned)a.num_kv_heads);
  int lpt_swz_log2 = 0, lpt_hb_quot = 0, lpt_hb_rem = 1;
  unsigned long long lpt_major_magic = 1, lpt_rem_magic = 1;
  {
    const long kv_head_bytes = (long)a.seqlen * (a.head_dim + a.head_dim) * 2;  // K + V per kv-head
    const long size_l2 = 100L << 20;  // GB200 L2 ~126MB; leave headroom
    int swz = 1;
    while (((long)swz << 1) * kv_head_bytes <= size_l2) swz <<= 1;
    const int hb_total = a.batch * a.num_kv_heads;
    while (swz > hb_total && swz > 1) swz >>= 1;  // clamp to problem
    lpt_swz_log2 = 0;
    while ((1 << (lpt_swz_log2 + 1)) <= swz) ++lpt_swz_log2;
    lpt_hb_quot = hb_total >> lpt_swz_log2;
    lpt_hb_rem = hb_total - (lpt_hb_quot << lpt_swz_log2);
    if (lpt_hb_rem == 0) lpt_hb_rem = 1;
    lpt_major_magic = make_magic((unsigned)(packed_mtiles_per_seq << lpt_swz_log2));
    lpt_rem_magic = make_magic((unsigned)lpt_hb_rem);
  }
  // block-causal-sink runtime bounds (0 tokens_per_block => plain full/causal path).
  const int tokens_per_block_arg = (a.tokens_per_block > 0) ? a.tokens_per_block : 0;
  const int sink_tokens_arg = (a.tokens_per_block > 0) ? a.sink_tokens : 0;
  const int rolling_window_tokens_arg = (a.tokens_per_block > 0) ? a.rolling_window_tokens : 0;
  const size_t smem =
      (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * K_TILE_BYTES  // Q (x2) + shared K/V ring
      + (size_t)2 * M_TILE * HEAD_DIM * sizeof(__nv_bfloat16)  // 2 sO bufs for TMA-O
      + (2 * NUM_KV_STAGES + 22) * 8            // mbarriers (incl full/empty_bar_o_epi)
      + (size_t)CLC_STAGES * (2 * 8 + 16) + 16  // CLC: clc_full+clc_empty + response (16B aligned)
      + 8                                       // tmem_slot
      + (size_t)2 * M_TILE * sizeof(float)      // alpha_and_l_smem [2][M_TILE]
      + 512;  // slack / alignment + isolated wait_scale bar granule

  constexpr bool FULL_NAMED_BAR = true, EX2_EMU = true, SPLIT_P = true, SOFTMAX_THROTTLE = true,
                 Q_RASTER = true;
  constexpr bool USE_CLC = false;  // PROBE: static + swizzle (FA4's causal config)
  auto kernel_fn =
      &block_causal_sink_sm100a_kernel<32, FULL_NAMED_BAR, EX2_EMU, SPLIT_P, SOFTMAX_THROTTLE,
                                       USE_CLC, Q_RASTER, MHA, LPT, 8, HAS_SINK_ROPE_DELTA>;
  CUDA_CHECK(
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

  // ---- launch geometry: CLC persistent. Launch the FULL problem grid (one CTA per work tile);
  // clusterlaunchcontrol keeps only ~#SMs CTAs resident and hands the rest of the CTA-ids out via
  // try_cancel (HW work-stealing scheduler), so the grid-size is the tile count, not #SMs. ----
  // exp2-domain scale: qk * (sm_scale * log2e), matching FastVideo's qk_scale = sm_scale * LOG2E.
  // Wrong here corrupts BOTH O and lse (lse = m*scale_log2 + log2 l), and does so plausibly.
  const float sm_scale = (a.sm_scale > 0.f) ? a.sm_scale : (1.0f / sqrtf((float)a.head_dim));
  const float scale_log2 = sm_scale * (float)M_LOG2E;
  int numSM = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0));
  const int total_workitems_host = a.batch * packed_mtiles_per_seq *
                                   a.num_kv_heads;  // one CTA per (sample, packed-M tile, kv-head)
  const int nblk = USE_CLC ? total_workitems_host : std::min(total_workitems_host, numSM);
  (void)numSM;
  dim3 grid(nblk, 1, 1), block(N_WARPS * 32, 1, 1);

  // CLC must be launched via cudaLaunchKernelEx with a cluster-dimension attribute -- a plain
  // <<<grid,block>>> launch does NOT enable clusterlaunchcontrol (try_cancel silently misbehaves
  // and tiles get skipped). cluster {1,1,1} (matches __cluster_dims__(1,1,1)); no PSS attribute
  // (we don't drive griddepcontrol, so leave the dependent-launch serialization off).
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute cfgAttr[1];
  cfgAttr[0].id = cudaLaunchAttributeClusterDimension;
  cfgAttr[0].val.clusterDim.x = 1;
  cfgAttr[0].val.clusterDim.y = 1;
  cfgAttr[0].val.clusterDim.z = 1;
  cfg.attrs = cfgAttr;
  cfg.numAttrs = 1;
  auto launch = [&]() {
    if (USE_CLC)
      return cudaLaunchKernelEx(
          &cfg, kernel_fn, tmap_q, tmap_k, tmap_v_t, tmap_o, tmap_q_sink, a.lse, a.seqlen,
          a.num_q_heads, a.num_kv_heads, scale_log2, packed_mtiles_per_seq, a.batch, magic0, magic1,
          magic2, lpt_swz_log2, lpt_hb_quot, lpt_hb_rem, lpt_major_magic, lpt_rem_magic,
          tokens_per_block_arg, sink_tokens_arg, rolling_window_tokens_arg);
    kernel_fn<<<grid, block, smem, stream>>>(
        tmap_q, tmap_k, tmap_v_t, tmap_o, tmap_q_sink, a.lse, a.seqlen, a.num_q_heads,
        a.num_kv_heads, scale_log2, packed_mtiles_per_seq, a.batch, magic0, magic1, magic2,
        lpt_swz_log2, lpt_hb_quot, lpt_hb_rem, lpt_major_magic, lpt_rem_magic, tokens_per_block_arg,
        sink_tokens_arg, rolling_window_tokens_arg);
    return cudaGetLastError();
  };

  return launch();
}

// Runtime -> template dispatch. MHA (H_q == H_kv) and the relativistic sink correction are
// compile-time in the kernel; the caller only knows them at runtime, so pick the instantiation
// here. LPT (heaviest-first causal balance) is a fixed tuning choice.
inline cudaError_t launch_block_causal_sink_sm100a(const BlockCausalSinkArgs& a,
                                                   cudaStream_t stream) {
  const cudaError_t bad = block_causal_sink_supported(a);
  if (bad != cudaSuccess) return bad;
  constexpr bool LPT = true;
  const bool mha = (a.num_q_heads == a.num_kv_heads);
  if (mha) {
    return a.has_delta ? launch_block_causal_sink_impl<true, LPT, true>(a, stream)
                       : launch_block_causal_sink_impl<true, LPT, false>(a, stream);
  }
  return a.has_delta ? launch_block_causal_sink_impl<false, LPT, true>(a, stream)
                     : launch_block_causal_sink_impl<false, LPT, false>(a, stream);
}
