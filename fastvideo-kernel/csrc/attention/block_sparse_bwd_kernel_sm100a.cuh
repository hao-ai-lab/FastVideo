// block_sparse_bwd_kernel_sm100a.cuh -- VSA block-sparse attention BACKWARD (blk64
// one-pass), sm_100a. Warp-specialized: load / MMA (tcgen05) / softmax (P^T, dS^T) /
// epilogue (dQ drain) / scheduler. Three kernels: preprocess (Delta, Q^T, dO^T, dqaccum
// zero), main (dK, dV, dQ partials), postprocess (dQ unscramble + scale).
#ifndef BLOCK_SPARSE_VSA_BWD_KERNEL_SM100A_CUH
#define BLOCK_SPARSE_VSA_BWD_KERNEL_SM100A_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>
#include "primitives.cuh"

namespace vsa_bwd_blk64 {

constexpr int BLOCK                   = 64;
constexpr int KV_TILE                 = BLOCK;
constexpr int QBLOCKS_PER_QUAD        = 4;
constexpr int Q_QUAD                  = QBLOCKS_PER_QUAD * BLOCK;
constexpr int M_TILE                  = KV_TILE;
[[maybe_unused]] constexpr int K_TILE = Q_QUAD;
constexpr int HEAD_DIM                = 128;
constexpr int SUB_COLS_BF16           = 64;
constexpr int SUB_COLS_BYTES          = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
constexpr int KV_SUBTILES             = HEAD_DIM / SUB_COLS_BF16;
constexpr int KV_SUB_COLS_BYTES       = KV_TILE * SUB_COLS_BYTES;
constexpr int KV_TILE_BYTES           = KV_SUBTILES * KV_SUB_COLS_BYTES;
constexpr int Q_BLK_BYTES             = HEAD_DIM * SUB_COLS_BYTES;
constexpr int QBLOCKS_PER_SLOT        = 2;
constexpr int Q_RING_SLOT_BYTES       = QBLOCKS_PER_SLOT * Q_BLK_BYTES;
constexpr int SLOTS_PER_QUAD          = QBLOCKS_PER_QUAD / QBLOCKS_PER_SLOT;
constexpr int PRE_QBLOCKS             = 2;
constexpr int PRE_TOKENS              = PRE_QBLOCKS * BLOCK;
constexpr int NUM_Q_STAGES            = 2 * SLOTS_PER_QUAD;
constexpr int DOT_SLOT0               = 0;
constexpr int QT_SLOT0                = SLOTS_PER_QUAD;
constexpr int DST_TILE_BYTES          = KV_TILE * BLOCK * (int)sizeof(__nv_bfloat16);
constexpr int DST_TILES               = 4;
constexpr int DST_BYTES               = DST_TILES * DST_TILE_BYTES;
constexpr int MMA_K                   = 16;
constexpr int K_ATOMS_PER_SUBTILE     = SUB_COLS_BF16 / MMA_K;
constexpr int K_ATOMS_PER_QBLOCK      = BLOCK / MMA_K;
constexpr int K_ATOMS_PER_KV_TILE     = KV_TILE / MMA_K;
constexpr int BF16X2_COLS_PER_K16     = MMA_K / 2;

template <typename DQ_DTYPE = float>
struct DQConfig {
  static constexpr int COLS                 = sizeof(DQ_DTYPE) == 2 ? 64 : 32;
  static constexpr int DQ_ONE_PUSH_BYTES    = BLOCK * COLS * (int)sizeof(DQ_DTYPE);
  static constexpr int WARP_ELEMS           = 32 * COLS;
  static constexpr int DQ_QBLOCKS_PER_STAGE = 2;
  static constexpr int DQ_STAGE_BYTES       = DQ_QBLOCKS_PER_STAGE * DQ_ONE_PUSH_BYTES;
  static constexpr int DQ_STAGE_BUFFERS     = 2;
  static constexpr int DQ_BLOCK_ELEMS       = BLOCK * HEAD_DIM;
};
static_assert(DQConfig<uint16_t>::DQ_STAGE_BYTES == DQConfig<float>::DQ_STAGE_BYTES,
              "f16 and fp32 pushes must match");

constexpr int N_WARPS = 16;
constexpr int W_EPI0 = 0, W_SOFTMAX0 = 4, W_MMA = 12, W_LOAD = 13, W_SCHED = 14;
constexpr int CLC_STAGES   = 2;
constexpr int CLC_ARRIVALS = 15;

constexpr int NUM_BARS   = 2 * NUM_Q_STAGES + 14 + 2 * CLC_STAGES;
constexpr int SMEM_TOTAL = 2 * KV_TILE_BYTES + NUM_Q_STAGES * Q_RING_SLOT_BYTES + DST_BYTES +
                           DQConfig<>::DQ_STAGE_BUFFERS * DQConfig<>::DQ_STAGE_BYTES +
                           2 * Q_QUAD * (int)sizeof(float) + NUM_BARS * 8 + CLC_STAGES * 16 + 48;

constexpr int ST_COLS    = Q_QUAD / 2;
constexpr int DV_COLS    = HEAD_DIM;
constexpr int DK_COLS    = HEAD_DIM;
constexpr int TMEM_TOTAL = ST_COLS + DV_COLS + ST_COLS + DK_COLS;
static_assert(TMEM_TOTAL == 512, "TMEM map must fill exactly 512 columns");

extern __shared__ __align__(1024) uint8_t bwd_smem[];

struct WorkItem {
  int batch;
  int head;
  int kv_block_id_in_seq;
  const int* local_k2q_idx;
  int local_k2q_num;
  int num_quads;
};

__device__ __forceinline__ WorkItem decode_workitem(
    int workitem_id, const int* __restrict__ workitem_remap, const int* __restrict__ k2q_idx,
    const int* __restrict__ k2q_num, int max_q_blocks, int num_heads, int num_kv_blocks_per_seq) {
  WorkItem it;
  const int real_item_id = workitem_remap[workitem_id];
  const int batch_head   = real_item_id / num_kv_blocks_per_seq;
  it.batch               = batch_head / num_heads;
  it.head                = batch_head % num_heads;
  it.kv_block_id_in_seq  = real_item_id % num_kv_blocks_per_seq;
  it.local_k2q_idx       = k2q_idx + (size_t)real_item_id * (size_t)max_q_blocks;
  it.local_k2q_num       = k2q_num[real_item_id];
  static_assert(QBLOCKS_PER_QUAD == 4, "num_quads uses >> 2");
  it.num_quads = (it.local_k2q_num + 3) >> 2;
  return it;
}

template <bool BHSD>
__device__ __forceinline__ size_t token_offset(int batch, int head, int num_heads, int seqlen,
                                               int t) {
  if constexpr (BHSD)
    return ((size_t)(batch * num_heads + head) * seqlen + t) * HEAD_DIM;
  else
    return ((size_t)(batch * seqlen + t) * num_heads + head) * HEAD_DIM;
}

template <bool DQ_L2_KEEP = false, bool USE_CLC = true, bool BHSD = false,
          typename DQ_DTYPE = float>
__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1) vsa_bwd_main_kernel(
    const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v,
    const __grid_constant__ CUtensorMap tmap_qt, const __grid_constant__ CUtensorMap tmap_dot,
    const __grid_constant__ CUtensorMap tmap_dk, const __grid_constant__ CUtensorMap tmap_dv,
    DQ_DTYPE* __restrict__ dqaccum, const float* __restrict__ lse_rows,
    const float* __restrict__ delta_rows, const int* __restrict__ k2q_idx,
    const int* __restrict__ k2q_num, const int* __restrict__ workitem_remap,
    const int* __restrict__ variable_block_sizes, int max_q_blocks, int num_samples, int num_heads,
    int seqlen, float scale_log2, float sm_scale) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))
  using DQ                        = DQConfig<DQ_DTYPE>;
  const int num_kv_blocks_per_seq = seqlen / BLOCK;

  uint8_t* sK              = bwd_smem;
  uint8_t* sV              = sK + KV_TILE_BYTES;
  uint8_t* sRING           = sV + KV_TILE_BYTES;
  uint8_t* sDST            = sRING + NUM_Q_STAGES * Q_RING_SLOT_BYTES;
  uint8_t* sDQ_STAGE_bytes = sDST + DST_BYTES;
  DQ_DTYPE* sDQ_STAGE[2]   = {reinterpret_cast<DQ_DTYPE*>(sDQ_STAGE_bytes),
                              reinterpret_cast<DQ_DTYPE*>(sDQ_STAGE_bytes + DQ::DQ_STAGE_BYTES)};
  __nv_bfloat16* sDV_STAGE = reinterpret_cast<__nv_bfloat16*>(sRING);
  __nv_bfloat16* sDK_STAGE = reinterpret_cast<__nv_bfloat16*>(sRING + KV_TILE_BYTES);
  float* sLSE =
      reinterpret_cast<float*>(sDQ_STAGE_bytes + DQ::DQ_STAGE_BUFFERS * DQ::DQ_STAGE_BYTES);
  float* sDelta = sLSE + Q_QUAD;

  uint64_t* full_bar_ring   = reinterpret_cast<uint64_t*>(sDelta + Q_QUAD);
  uint64_t* empty_bar_ring  = full_bar_ring + NUM_Q_STAGES;
  uint64_t* full_bar_lse    = empty_bar_ring + NUM_Q_STAGES;
  uint64_t* empty_bar_lse   = full_bar_lse + 1;
  uint64_t* full_bar_delta  = empty_bar_lse + 1;
  uint64_t* empty_bar_delta = full_bar_delta + 1;
  uint64_t* full_bar_st     = empty_bar_delta + 1;
  uint64_t* full_bar_dpt    = full_bar_st + 1;
  uint64_t* full_bar_pt     = full_bar_dpt + 1;
  uint64_t* full_bar_dst    = full_bar_pt + 1;
  uint64_t* full_bar_dq     = full_bar_dst + 1;
  uint64_t* empty_bar_dq    = full_bar_dq + 1;
  uint64_t* full_bar_dv     = empty_bar_dq + 1;
  uint64_t* full_bar_dk     = full_bar_dv + 1;
  uint64_t* empty_bar_kv    = full_bar_dk + 1;
  uint64_t* empty_bar_epi   = empty_bar_kv + 1;
  uint64_t* clc_full        = empty_bar_epi + 1;
  uint64_t* clc_empty       = clc_full + CLC_STAGES;
  uint32_t* clc_response    = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(clc_empty + CLC_STAGES) + 15u) & ~uintptr_t(15u));
  uint32_t* tmem_slot = clc_response + CLC_STAGES * 4;

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

  if (warp_id == 0) {
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), TMEM_TOTAL);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();

  const uint32_t tmem_base    = *tmem_slot;
  const uint32_t tmem_st      = tmem_base;
  const uint32_t tmem_dv      = tmem_st + ST_COLS;
  const uint32_t tmem_dpt     = tmem_dv + DV_COLS;
  const uint32_t tmem_dk      = tmem_dpt + ST_COLS;
  const uint32_t tmem_pt_bf16 = tmem_st, tmem_dst_bf16 = tmem_dpt, tmem_dq = tmem_dpt;

  if (tid == 0) {
    #pragma unroll
    for (int s = 0; s < NUM_Q_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar_ring[s]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_ring[s]), 1);
    }
    mbarrier_init(smem_ptr_u32(full_bar_lse), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_lse), 8);
    mbarrier_init(smem_ptr_u32(full_bar_delta), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_delta), 8);
    mbarrier_init(smem_ptr_u32(full_bar_st), 1);
    mbarrier_init(smem_ptr_u32(full_bar_dpt), 1);
    mbarrier_init(smem_ptr_u32(full_bar_pt), 8);
    mbarrier_init(smem_ptr_u32(full_bar_dst), 8);
    mbarrier_init(smem_ptr_u32(full_bar_dq), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_dq), 4);
    mbarrier_init(smem_ptr_u32(full_bar_dv), 1);
    mbarrier_init(smem_ptr_u32(full_bar_dk), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_kv), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_epi), 256);
    if constexpr (USE_CLC) {
      #pragma unroll
      for (int st = 0; st < CLC_STAGES; ++st) {
        mbarrier_init(smem_ptr_u32(&clc_full[st]), 1);
        mbarrier_init(smem_ptr_u32(&clc_empty[st]), CLC_ARRIVALS);
      }
      #pragma unroll
      for (int i = 0; i < CLC_STAGES * 4; ++i) clc_response[i] = 0;
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();

  [[maybe_unused]] const int total_workitems = num_samples * num_heads * num_kv_blocks_per_seq;

  if (warp_id == W_LOAD) {
    setmaxnreg_dec<88>();

    EmptyPhaseTracker<NUM_Q_STAGES> ring_empty_ph;
    EmptyPhaseTracker<1> lse_empty_ph, delta_empty_ph, kv_empty_ph, epi_empty_ph;

    [[maybe_unused]] int clc_stage      = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id                     = (int)blockIdx.x;
    while (workitem_id >= 0) {
      const WorkItem it = decode_workitem(workitem_id, workitem_remap, k2q_idx, k2q_num,
                                          max_q_blocks, num_heads, num_kv_blocks_per_seq);

      if (it.local_k2q_num != 0) {
        auto get_global_qblock_id = [&](int quad_idx, int qblock_id_in_quad) {
          return it.local_k2q_idx[min(QBLOCKS_PER_QUAD * quad_idx + qblock_id_in_quad,
                                      it.local_k2q_num - 1)];
        };

        auto load_kv_tile = [&](uint8_t* dst, const CUtensorMap* map, uint64_t* full_bar) {
          #pragma unroll
          for (int s = 0; s < KV_SUBTILES; ++s) {
            if constexpr (BHSD)
              tma_load_4d(smem_ptr_u32(dst + s * KV_SUB_COLS_BYTES), map, smem_ptr_u32(full_bar), 0,
                          it.kv_block_id_in_seq * KV_TILE, s, it.batch * num_heads + it.head);
            else
              tma_load_3d(smem_ptr_u32(dst + s * KV_SUB_COLS_BYTES), map, smem_ptr_u32(full_bar), 0,
                          it.batch * seqlen + it.kv_block_id_in_seq * KV_TILE,
                          it.head * KV_SUBTILES + s);
          }
        };

        auto load_quad = [&](const CUtensorMap* map, int quad_idx, bool with_kv) {
          if (with_kv) {
            mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_kv), kv_empty_ph.get_phase());
            kv_empty_ph.advance();
          }

          for (int pair_idx = 0; pair_idx < SLOTS_PER_QUAD; ++pair_idx) {
            const int slot = ring_empty_ph.get_stage();
            mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_ring[slot]),
                                         ring_empty_ph.get_phase());
            ring_empty_ph.advance();

            const int qblock_in_quad    = QBLOCKS_PER_SLOT * pair_idx;
            const int global_qblock_id0 = get_global_qblock_id(quad_idx, qblock_in_quad);
            const int global_qblock_id1 = get_global_qblock_id(quad_idx, qblock_in_quad + 1);
            if (elect_one_sync()) {
              mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_ring[slot]),
                                        Q_RING_SLOT_BYTES + (with_kv ? KV_TILE_BYTES : 0));
              tma_load_2d(smem_ptr_u32(sRING + slot * Q_RING_SLOT_BYTES), map,
                          smem_ptr_u32(&full_bar_ring[slot]),
                          it.batch * seqlen + global_qblock_id0 * BLOCK, it.head * HEAD_DIM);
              tma_load_2d(smem_ptr_u32(sRING + slot * Q_RING_SLOT_BYTES + Q_BLK_BYTES), map,
                          smem_ptr_u32(&full_bar_ring[slot]),
                          it.batch * seqlen + global_qblock_id1 * BLOCK, it.head * HEAD_DIM);
              if (with_kv) {
                if (pair_idx == 0)
                  load_kv_tile(sK, &tmap_k, &full_bar_ring[slot]);
                else
                  load_kv_tile(sV, &tmap_v, &full_bar_ring[slot]);
              }
            }
          }
        };

        auto load_lse_and_delta = [&](int quad_idx) {
          mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_lse), lse_empty_ph.get_phase());
          lse_empty_ph.advance();
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(full_bar_lse), Q_QUAD * 4);
            #pragma unroll
            for (int i = 0; i < QBLOCKS_PER_QUAD; ++i)
              cpasync_bulk_load_mbarrier(smem_ptr_u32(sLSE + i * BLOCK),
                                         lse_rows +
                                             (size_t)(it.batch * num_heads + it.head) * seqlen +
                                             (size_t)get_global_qblock_id(quad_idx, i) * BLOCK,
                                         BLOCK * 4, smem_ptr_u32(full_bar_lse));
          }

          mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_delta), delta_empty_ph.get_phase());
          delta_empty_ph.advance();
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(full_bar_delta), Q_QUAD * 4);
            #pragma unroll
            for (int i = 0; i < QBLOCKS_PER_QUAD; ++i)
              cpasync_bulk_load_mbarrier(smem_ptr_u32(sDelta + i * BLOCK),
                                         delta_rows +
                                             (size_t)(it.batch * num_heads + it.head) * seqlen +
                                             (size_t)get_global_qblock_id(quad_idx, i) * BLOCK,
                                         BLOCK * 4, smem_ptr_u32(full_bar_delta));
          }
        };

        mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_epi), epi_empty_ph.get_phase());
        epi_empty_ph.advance();
        load_quad(&tmap_dot, 0, false);
        load_quad(&tmap_qt, 0, true);
        load_lse_and_delta(0);
        for (int j = 1; j < it.num_quads; ++j) {
          load_quad(&tmap_dot, j, false);
          load_lse_and_delta(j);
          load_quad(&tmap_qt, j, false);
        }
      }

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        workitem_id = next.valid ? (int)next.n_tile : -1;
      } else {
        workitem_id += (int)gridDim.x;
        if (workitem_id >= total_workitems) workitem_id = -1;
      }
    }
    return;
  } else if (warp_id == W_MMA) {
    setmaxnreg_dec<88>();

    const uint32_t lead         = elect_one_sync() ? 1u : 0u;
    const uint32_t idesc_st_dpt = make_idesc_bf16_f32(M_TILE, 2 * BLOCK, false, true);
    const uint32_t idesc_dv_dk  = make_idesc_bf16_f32(M_TILE, 2 * HEAD_DIM, false, false);
    const uint32_t idesc_dq     = make_idesc_bf16_f32(Q_QUAD / 2, HEAD_DIM, true, true);

    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    auto make_smem_desc = [](const uint8_t* smem, uint32_t leading_byte_offset) {
      return build_smem_desc_blackwell(smem_ptr_u32(smem), DESC_SBO, leading_byte_offset,
                                       SmemSwizzleBlackwell::B128);
    };

    const uint64_t desc_k       = make_smem_desc(sK, DESC_LBO);
    const uint64_t desc_v       = make_smem_desc(sV, DESC_LBO);
    const uint64_t desc_ring    = make_smem_desc(sRING, DESC_LBO);
    const uint64_t desc_ring_mn = make_smem_desc(sRING, Q_BLK_BYTES);
    const uint64_t desc_k_mn    = make_smem_desc(sK, KV_SUB_COLS_BYTES);
    const uint64_t desc_dst0    = make_smem_desc(sDST, DST_TILE_BYTES);
    const uint64_t desc_dst1    = make_smem_desc(sDST + 2 * DST_TILE_BYTES, DST_TILE_BYTES);

    constexpr uint32_t K16_ROWS_DELTA    = (MMA_K * SUB_COLS_BYTES) >> 4;
    constexpr uint64_t K16_COLS_DELTA    = (MMA_K * (int)sizeof(__nv_bfloat16)) >> 4;
    constexpr uint64_t KV_SUB_COLS_DELTA = KV_SUB_COLS_BYTES >> 4;
    constexpr uint64_t RING_DELTA        = Q_RING_SLOT_BYTES >> 4;

    PhaseTracker<1> pt_ph, dst_ph;
    PhaseTracker<1> dq_empty_ph;
    EmptyPhaseTracker<1> epi_empty_ph;
    PhaseTracker<1> ring_full_ph;

    auto gemm12_st_dpt = [&](auto is_st_const) {
      constexpr bool is_st    = decltype(is_st_const)::value;
      const uint32_t tmem_acc = is_st ? tmem_st : tmem_dpt;
      const uint64_t da_base  = is_st ? desc_k : desc_v;
      uint64_t* commit_bar    = is_st ? full_bar_st : full_bar_dpt;
      #pragma unroll
      for (int u = 0; u < SLOTS_PER_QUAD; ++u) {
        const int slot = (is_st ? QT_SLOT0 : DOT_SLOT0) + u;
        mbarrier_wait_parity(smem_ptr_u32(&full_bar_ring[slot]), ring_full_ph.get_phase());
        #pragma unroll
        for (int s = 0; s < KV_SUBTILES; ++s) {
          const uint64_t da = da_base + (uint64_t)s * KV_SUB_COLS_DELTA;
          const uint64_t db = desc_ring_mn + (uint64_t)slot * RING_DELTA +
                              (uint64_t)s * K_ATOMS_PER_SUBTILE * K16_ROWS_DELTA;
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_SUBTILE; ++ki) {
            const bool enable_d = (s != 0) || (ki != 0);
            tcgen05_mma_ws_f16_ss_1sm_predicated(
                lead, tmem_acc + (uint32_t)(BLOCK * u), da + ki * K16_COLS_DELTA,
                db + (uint64_t)ki * K16_ROWS_DELTA, idesc_st_dpt, enable_d);
          }
        }
      }
      tcgen05_commit1_lead(lead, smem_ptr_u32(commit_bar));
    };

    auto gemm35_dv_dk = [&](auto is_dv_const, bool first) {
      constexpr bool is_dv       = decltype(is_dv_const)::value;
      const uint32_t tmem_acc    = is_dv ? tmem_dv : tmem_dk;
      const uint32_t tmem_a_base = is_dv ? tmem_pt_bf16 : tmem_dst_bf16;
      #pragma unroll
      for (int p = 0; p < SLOTS_PER_QUAD; ++p) {
        const int slot    = (is_dv ? DOT_SLOT0 : QT_SLOT0) + p;
        const uint64_t db = desc_ring + (uint64_t)slot * RING_DELTA;
        #pragma unroll
        for (int ki = 0; ki < K_ATOMS_PER_QBLOCK; ++ki) {
          const int a           = p * K_ATOMS_PER_QBLOCK + ki;
          const uint32_t tmem_a = tmem_a_base + (uint32_t)(a * BF16X2_COLS_PER_K16);
          const bool accumulate = (!first) || (a != 0);
          tcgen05_mma_ws_f16_ts_1sm_predicated(lead, tmem_acc, tmem_a, db + ki * K16_COLS_DELTA,
                                               idesc_dv_dk, accumulate);
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar_ring[slot]));
      }
    };

    auto gemm4_dq_half = [&](uint64_t desc_dst_h) {
      uint64_t adst = desc_dst_h;
      uint64_t bk   = desc_k_mn;
      #pragma unroll
      for (int ki = 0; ki < K_ATOMS_PER_KV_TILE; ++ki) {
        tcgen05_mma_f16_ss_lead(lead, tmem_dq, adst, bk, idesc_dq, ki != 0);
        smem_desc_add_lo(adst, K16_ROWS_DELTA);
        smem_desc_add_lo(bk, K16_ROWS_DELTA);
      }
      tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dq));
    };

    auto gemm4_dq = [&]() {
      gemm4_dq_half(desc_dst0);
      mbarrier_wait_parity(smem_ptr_u32(empty_bar_dq), dq_empty_ph.get_phase());
      dq_empty_ph.advance();
      gemm4_dq_half(desc_dst1);
    };

    [[maybe_unused]] int clc_stage      = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id                     = (int)blockIdx.x;
    while (workitem_id >= 0) {
      const WorkItem it = decode_workitem(workitem_id, workitem_remap, k2q_idx, k2q_num,
                                          max_q_blocks, num_heads, num_kv_blocks_per_seq);

      if (it.local_k2q_num != 0) {
        gemm12_st_dpt(std::true_type{});
        mbarrier_wait_parity(smem_ptr_u32(empty_bar_dq), dq_empty_ph.get_phase());
        dq_empty_ph.advance();
        gemm12_st_dpt(std::false_type{});
        ring_full_ph.advance();
        mbarrier_wait_parity(smem_ptr_u32(full_bar_pt), pt_ph.get_phase());
        pt_ph.advance();
        mbarrier_wait_parity(smem_ptr_u32(empty_bar_epi), epi_empty_ph.get_phase());
        epi_empty_ph.advance();
        gemm35_dv_dk(std::true_type{}, true);

        for (int j = 0; j < it.num_quads - 1; ++j) {
          mbarrier_wait_parity(smem_ptr_u32(full_bar_dst), dst_ph.get_phase());
          dst_ph.advance();
          gemm35_dv_dk(std::false_type{}, j == 0);
          gemm4_dq();
          gemm12_st_dpt(std::true_type{});
          mbarrier_wait_parity(smem_ptr_u32(empty_bar_dq), dq_empty_ph.get_phase());
          dq_empty_ph.advance();
          gemm12_st_dpt(std::false_type{});
          ring_full_ph.advance();
          mbarrier_wait_parity(smem_ptr_u32(full_bar_pt), pt_ph.get_phase());
          pt_ph.advance();
          gemm35_dv_dk(std::true_type{}, false);
        }

        tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dv));
        mbarrier_wait_parity(smem_ptr_u32(full_bar_dst), dst_ph.get_phase());
        dst_ph.advance();
        gemm35_dv_dk(std::false_type{}, it.num_quads == 1);
        tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dk));
        gemm4_dq();
        tcgen05_commit1_lead(lead, smem_ptr_u32(empty_bar_kv));
      }

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        workitem_id = next.valid ? (int)next.n_tile : -1;
      } else {
        workitem_id += (int)gridDim.x;
        if (workitem_id >= total_workitems) workitem_id = -1;
      }
    }

    bar_sync<10>(416);
    tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL);
    return;
  } else if (warp_id == W_SCHED) {
    setmaxnreg_dec<88>();

    if constexpr (USE_CLC) {
      int prod_stage      = 0;
      uint32_t prod_phase = 1;
      int cons_stage      = 0;
      uint32_t cons_phase = 0;

      while (true) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        clc_arrive_expect_tx_cta(smem_ptr_u32(&clc_full[prod_stage]), 16);
        if (lane == 0)
          clc_try_cancel_async(smem_ptr_u32(&clc_response[prod_stage * 4]),
                               smem_ptr_u32(&clc_full[prod_stage]));
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
        ClcTileInfo n = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, cons_stage, cons_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(cons_stage, cons_phase);
        if (!n.valid) break;
      }

      #pragma unroll
      for (int st = 0; st < CLC_STAGES; ++st) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
      }
    }
    return;
  } else if (warp_id >= W_SOFTMAX0 && warp_id < W_MMA) {
    setmaxnreg_inc<136>();

    PhaseTracker<1> st_ph, dpt_ph, lse_ph, delta_ph, dv_ph, dk_ph;
    constexpr int HALF_COLS = SUB_COLS_BF16;
    static_assert(ST_COLS == 2 * HALF_COLS && DV_COLS == 2 * HALF_COLS, "two column halves");

    const int softmax_warp_id = warp_id - W_SOFTMAX0;
    const int lane_group      = softmax_warp_id & 3;
    const int col_half        = softmax_warp_id >> 2;
    const int q_half          = lane_group >> 1;
    const int kv_row          = (lane_group & 1) * 32 + lane;
    const int qblock_in_quad  = 2 * col_half + q_half;

    const uint32_t tmem_lane_base     = (uint32_t)(lane_group * 32) << 16;
    const uint32_t tmem_f32_offset    = tmem_lane_base + (uint32_t)(col_half * HALF_COLS);
    const uint32_t tmem_bf16x2_offset = tmem_lane_base + (uint32_t)(col_half * HALF_COLS / 2);

    const float2* lse2   = reinterpret_cast<const float2*>(sLSE + qblock_in_quad * BLOCK);
    const float2* delta2 = reinterpret_cast<const float2*>(sDelta + qblock_in_quad * BLOCK);

    constexpr int CHUNK_BF16     = 16 / (int)sizeof(__nv_bfloat16);
    constexpr int CHUNKS_PER_ROW = SUB_COLS_BF16 / CHUNK_BF16;
    __nv_bfloat16* sdst_row =
        reinterpret_cast<__nv_bfloat16*>(sDST + (size_t)(2 * q_half + col_half) * DST_TILE_BYTES) +
        kv_row * SUB_COLS_BF16;

    auto merge_lane_halves_and_store = [&](uint32_t tmem_acc, auto apply_sm_scale_const,
                                           const CUtensorMap* map, __nv_bfloat16* stage_tile,
                                           const WorkItem& it) {
      constexpr bool apply_sm_scale = decltype(apply_sm_scale_const)::value;
      __nv_bfloat16* stage_row =
          stage_tile + (size_t)col_half * (KV_TILE * SUB_COLS_BF16) + kv_row * SUB_COLS_BF16;
      uint32_t acc_regs[HALF_COLS];
      tcgen05_ld_32x32b_x64(tmem_acc + tmem_f32_offset, acc_regs);
      tcgen05_wait_ld();
      tcgen05_fence_before_thread_sync();
      const float2* acc2  = reinterpret_cast<const float2*>(acc_regs);
      const float2 scale2 = f32x2_splat(sm_scale);

      if (q_half == 1) {
        #pragma unroll
        for (int v = 0; v < CHUNKS_PER_ROW; ++v) {
          const float2* a2 = acc2 + v * (CHUNK_BF16 / 2);
          const float2 r0  = apply_sm_scale ? fmul2(a2[0], scale2) : a2[0];
          const float2 r1  = apply_sm_scale ? fmul2(a2[1], scale2) : a2[1];
          const float2 r2  = apply_sm_scale ? fmul2(a2[2], scale2) : a2[2];
          const float2 r3  = apply_sm_scale ? fmul2(a2[3], scale2) : a2[3];
          uint4 packed;
          packed.x = cvt_f32x2_to_bf16x2(r0.x, r0.y);
          packed.y = cvt_f32x2_to_bf16x2(r1.x, r1.y);
          packed.z = cvt_f32x2_to_bf16x2(r2.x, r2.y);
          packed.w = cvt_f32x2_to_bf16x2(r3.x, r3.y);
          *reinterpret_cast<uint4*>(stage_row + (v ^ (kv_row & 7)) * CHUNK_BF16) = packed;
        }
      }
      bar_sync<14>(256);

      if (q_half == 0) {
        #pragma unroll
        for (int v = 0; v < CHUNKS_PER_ROW; ++v) {
          const float2* a2 = acc2 + v * (CHUNK_BF16 / 2);
          uint4* chunk_ptr = reinterpret_cast<uint4*>(stage_row + (v ^ (kv_row & 7)) * CHUNK_BF16);
          const uint4 staged                 = *chunk_ptr;
          const __nv_bfloat162* staged_pairs = reinterpret_cast<const __nv_bfloat162*>(&staged);
          const float2 s0                    = __bfloat1622float2(staged_pairs[0]);
          const float2 s1                    = __bfloat1622float2(staged_pairs[1]);
          const float2 s2                    = __bfloat1622float2(staged_pairs[2]);
          const float2 s3                    = __bfloat1622float2(staged_pairs[3]);
          const float2 r0 = apply_sm_scale ? ffma2(a2[0], scale2, s0) : fadd2(a2[0], s0);
          const float2 r1 = apply_sm_scale ? ffma2(a2[1], scale2, s1) : fadd2(a2[1], s1);
          const float2 r2 = apply_sm_scale ? ffma2(a2[2], scale2, s2) : fadd2(a2[2], s2);
          const float2 r3 = apply_sm_scale ? ffma2(a2[3], scale2, s3) : fadd2(a2[3], s3);
          uint4 packed;
          packed.x   = cvt_f32x2_to_bf16x2(r0.x, r0.y);
          packed.y   = cvt_f32x2_to_bf16x2(r1.x, r1.y);
          packed.z   = cvt_f32x2_to_bf16x2(r2.x, r2.y);
          packed.w   = cvt_f32x2_to_bf16x2(r3.x, r3.y);
          *chunk_ptr = packed;
        }
      }
      fence_proxy_async_shared();
      bar_sync<14>(256);

      if (softmax_warp_id == 0 && elect_one_sync()) {
        #pragma unroll
        for (int s = 0; s < KV_SUBTILES; ++s) {
          const uint32_t src = smem_ptr_u32(stage_tile + (size_t)s * KV_TILE * SUB_COLS_BF16);
          if constexpr (BHSD)
            tma_store_4d(map, 0, it.kv_block_id_in_seq * KV_TILE, s, it.batch * num_heads + it.head,
                         src);
          else
            tma_store_3d(map, 0, it.batch * seqlen + it.kv_block_id_in_seq * KV_TILE,
                         it.head * KV_SUBTILES + s, src);
        }
        cp_async_bulk_commit_group();
      }
      bar_sync<14>(256);
    };

    [[maybe_unused]] int clc_stage      = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id                     = (int)blockIdx.x;
    while (workitem_id >= 0) {
      const WorkItem it = decode_workitem(workitem_id, workitem_remap, k2q_idx, k2q_num,
                                          max_q_blocks, num_heads, num_kv_blocks_per_seq);

      if (it.local_k2q_num != 0) {
        const bool kv_row_valid = kv_row < variable_block_sizes[it.kv_block_id_in_seq];

        for (int j = 0; j < it.num_quads; ++j) {
          const bool p_valid =
              kv_row_valid && (qblock_in_quad < it.local_k2q_num - QBLOCKS_PER_QUAD * j);

          mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_lse), lse_ph.get_phase());
          lse_ph.advance();
          mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_st), st_ph.get_phase());
          st_ph.advance();

          uint32_t st_regs[HALF_COLS];
          tcgen05_ld_32x32b_x64(tmem_st + tmem_f32_offset, st_regs);
          tcgen05_wait_ld();
          tcgen05_fence_before_thread_sync();

          float2* pt_fp32 = reinterpret_cast<float2*>(st_regs);
          uint32_t pt_bf16x2[HALF_COLS / 2];
          const float2 scale2 = f32x2_splat(scale_log2);
          #pragma unroll
          for (int c = 0; c < HALF_COLS / 2; ++c) {
            const float2 z = ffma2(pt_fp32[c], scale2, make_float2(-lse2[c].x, -lse2[c].y));
            float2 p       = make_float2(ex2_approx_f32(z.x), ex2_approx_f32(z.y));
            if (!p_valid) p = make_float2(0.f, 0.f);
            pt_fp32[c]   = p;
            pt_bf16x2[c] = cvt_f32x2_to_bf16x2(p.x, p.y);
          }

          tcgen05_st_32x32b_x32(tmem_pt_bf16 + tmem_bf16x2_offset, pt_bf16x2);
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          if (elect_one_sync()) {
            mbarrier_arrive(smem_ptr_u32(full_bar_pt));
            mbarrier_arrive(smem_ptr_u32(empty_bar_lse));
          }

          mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_delta), delta_ph.get_phase());
          delta_ph.advance();
          mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dpt), dpt_ph.get_phase());
          dpt_ph.advance();

          uint32_t dpt_regs[HALF_COLS];
          tcgen05_ld_32x32b_x64(tmem_dpt + tmem_f32_offset, dpt_regs);
          tcgen05_wait_ld();
          tcgen05_fence_before_thread_sync();

          const float2* dpt2 = reinterpret_cast<const float2*>(dpt_regs);
          uint32_t (&dst_bf16x2)[HALF_COLS / 2] =
              reinterpret_cast<uint32_t (&)[HALF_COLS / 2]>(st_regs);
          #pragma unroll
          for (int c = 0; c < HALF_COLS / 2; ++c) {
            const float2 ds =
                fmul2(pt_fp32[c], fadd2(dpt2[c], make_float2(-delta2[c].x, -delta2[c].y)));
            dst_bf16x2[c] = cvt_f32x2_to_bf16x2(ds.x, ds.y);
          }

          tcgen05_st_32x32b_x32(tmem_dst_bf16 + tmem_bf16x2_offset, dst_bf16x2);
          const uint4* dst_chunks = reinterpret_cast<const uint4*>(dst_bf16x2);
          #pragma unroll
          for (int v = 0; v < CHUNKS_PER_ROW; ++v)
            *reinterpret_cast<uint4*>(sdst_row + (v ^ (kv_row & 7)) * CHUNK_BF16) = dst_chunks[v];
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          fence_proxy_async_shared();
          if (elect_one_sync()) {
            mbarrier_arrive(smem_ptr_u32(full_bar_dst));
            mbarrier_arrive(smem_ptr_u32(empty_bar_delta));
          }
        }

        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dv), dv_ph.get_phase());
        dv_ph.advance();
        merge_lane_halves_and_store(tmem_dv, std::false_type{}, &tmap_dv, sDV_STAGE, it);
        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dk), dk_ph.get_phase());
        dk_ph.advance();
        merge_lane_halves_and_store(tmem_dk, std::true_type{}, &tmap_dk, sDK_STAGE, it);

        if (softmax_warp_id == 0 && elect_one_sync()) {
          cp_async_bulk_wait_group_read<0>();
        }
        bar_sync<14>(256);
        mbarrier_arrive(smem_ptr_u32(empty_bar_epi));
      }

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        workitem_id = next.valid ? (int)next.n_tile : -1;
      } else {
        workitem_id += (int)gridDim.x;
        if (workitem_id >= total_workitems) workitem_id = -1;
      }
    }

    bar_sync<10>(416);
    return;
  } else if (warp_id < W_SOFTMAX0) {
    setmaxnreg_inc<152>();

    const int epi_warp_id         = warp_id - W_EPI0;
    const uint32_t tmem_lane_base = (uint32_t)(epi_warp_id * 32) << 16;
    const bool is_leader          = epi_warp_id == 0 && lane == 0;

    uint64_t dqaccum_l2_policy = 0;
    if constexpr (DQ_L2_KEEP) {
      dqaccum_l2_policy = make_l2cache_policy_fractional_evict_last_unchanged(0.25f);
    }

    auto reduce_add_push = [&](DQ_DTYPE* dst, uint32_t src_smem) {
      if constexpr (sizeof(DQ_DTYPE) == 4) {
        if constexpr (DQ_L2_KEEP) {
          cpasync_reduce_bulk_add_f32_l2hint(dst, src_smem, DQ::DQ_ONE_PUSH_BYTES,
                                             dqaccum_l2_policy);
        } else {
          cpasync_reduce_bulk_add_f32(dst, src_smem, DQ::DQ_ONE_PUSH_BYTES);
        }
      } else {
        if constexpr (DQ_L2_KEEP) {
          cpasync_reduce_bulk_add_f16_l2hint(dst, src_smem, DQ::DQ_ONE_PUSH_BYTES,
                                             dqaccum_l2_policy);
        } else {
          cpasync_reduce_bulk_add_f16(dst, src_smem, DQ::DQ_ONE_PUSH_BYTES);
        }
      }
    };

    PhaseTracker<1> dq_full_ph;
    if (elect_one_sync()) {
      mbarrier_arrive(smem_ptr_u32(empty_bar_dq));
    }

    [[maybe_unused]] int clc_stage      = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id                     = (int)blockIdx.x;
    while (workitem_id >= 0) {
      const WorkItem it = decode_workitem(workitem_id, workitem_remap, k2q_idx, k2q_num,
                                          max_q_blocks, num_heads, num_kv_blocks_per_seq);

      if (it.local_k2q_num != 0) {
        DQ_DTYPE* dqaccum_head = dqaccum + (size_t)(it.batch * num_heads + it.head) *
                                               num_kv_blocks_per_seq * DQ::DQ_BLOCK_ELEMS;

        for (int j = 0; j < it.num_quads; ++j) {
          #pragma unroll 1
          for (int h = 0; h < 2; ++h) {
            const int qblock_lo =
                it.local_k2q_idx[min(QBLOCKS_PER_QUAD * j + h, it.local_k2q_num - 1)];
            const int qblock_hi =
                it.local_k2q_idx[min(QBLOCKS_PER_QUAD * j + h + 2, it.local_k2q_num - 1)];
            DQ_DTYPE* dqaccum_lo = dqaccum_head + (size_t)qblock_lo * DQ::DQ_BLOCK_ELEMS;
            DQ_DTYPE* dqaccum_hi = dqaccum_head + (size_t)qblock_hi * DQ::DQ_BLOCK_ELEMS;

            mbarrier_wait_parity(smem_ptr_u32(full_bar_dq), dq_full_ph.get_phase());
            dq_full_ph.advance();

            uint32_t dq_regs[HEAD_DIM];
            #pragma unroll
            for (int c = 0; c < HEAD_DIM / 64; ++c) {
              tcgen05_ld_32x32b_x64(tmem_dq + tmem_lane_base + (uint32_t)(c * 64),
                                    reinterpret_cast<uint32_t (&)[64]>(dq_regs[c * 64]));
            }
            tcgen05_wait_ld();
            tcgen05_fence_before_thread_sync();
            if (elect_one_sync()) {
              mbarrier_arrive(smem_ptr_u32(empty_bar_dq));
            }

            #pragma unroll
            for (int hd_slice = 0; hd_slice < HEAD_DIM / DQ::COLS; ++hd_slice) {
              const int stage_buf = hd_slice & 1;
              const float4* dq_row4 =
                  reinterpret_cast<const float4*>(dq_regs + hd_slice * DQ::COLS);
              DQ_DTYPE* stage_row = sDQ_STAGE[stage_buf] + epi_warp_id * DQ::WARP_ELEMS + lane * 4;
              #pragma unroll
              for (int v4 = 0; v4 < DQ::COLS / 4; ++v4) {
                const float4 v = dq_row4[v4];
                if constexpr (sizeof(DQ_DTYPE) == 2) {
                  uint2 packed;
                  packed.x                                           = cvt_f32x2_to_f16x2(v.x, v.y);
                  packed.y                                           = cvt_f32x2_to_f16x2(v.z, v.w);
                  *reinterpret_cast<uint2*>(stage_row + v4 * 32 * 4) = packed;
                } else {
                  *reinterpret_cast<float4*>(stage_row + v4 * 32 * 4) = v;
                }
              }
              fence_proxy_async_shared();
              bar_sync<11>(128);

              if (is_leader) {
                const size_t slice_offset = (size_t)hd_slice * BLOCK * DQ::COLS;
                const uint32_t stage_lo   = smem_ptr_u32(sDQ_STAGE[stage_buf]);
                reduce_add_push(dqaccum_lo + slice_offset, stage_lo);
                reduce_add_push(dqaccum_hi + slice_offset, stage_lo + DQ::DQ_ONE_PUSH_BYTES);
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_group_read<1>();
              }
              bar_sync<11>(128);
            }
          }
        }
      }

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        workitem_id = next.valid ? (int)next.n_tile : -1;
      } else {
        workitem_id += (int)gridDim.x;
        if (workitem_id >= total_workitems) workitem_id = -1;
      }
    }

    if (is_leader) {
      cp_async_bulk_wait_group_read<0>();
    }
    bar_sync<11>(128);
    bar_sync<10>(416);
    return;
  } else {
    setmaxnreg_dec<24>();
    return;
  }
#endif
}

constexpr int ORDER_THREADS    = 1024;
constexpr int ORDER_SMEM_MAX   = 227 * 1024;
constexpr int ORDER_MAX_BLOCKS = ORDER_SMEM_MAX / (2 * (int)sizeof(int));

__global__ void __launch_bounds__(ORDER_THREADS, 1)
    vsa_bwd_order_kernel(const int* __restrict__ k2q_idx, const int* __restrict__ k2q_num,
                         int max_q_blocks, int num_kv_blocks_per_seq, int order_bin, bool snake,
                         int* __restrict__ order_out) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))
  extern __shared__ int order_smem[];
  int* sbin      = order_smem;
  int* smid      = order_smem + num_kv_blocks_per_seq;
  const int base = (int)blockIdx.x * num_kv_blocks_per_seq;

  for (int i = (int)threadIdx.x; i < num_kv_blocks_per_seq; i += ORDER_THREADS) {
    const int count = k2q_num[base + i];
    sbin[i]         = count / order_bin;
    smid[i] = snake ? (count > 0 ? k2q_idx[(size_t)(base + i) * max_q_blocks + count / 2] : -1) : 0;
  }
  __syncthreads();

  const int i = (int)blockIdx.y * ORDER_THREADS + (int)threadIdx.x;
  if (i < num_kv_blocks_per_seq) {
    const int bin = sbin[i], mid = smid[i];
    const bool descending = snake && (bin & 1);
    int rank              = 0;

    for (int j = 0; j < num_kv_blocks_per_seq; ++j) {
      const int bin_j = sbin[j];
      if (bin_j < bin) {
        ++rank;
      } else if (bin_j == bin) {
        const int mid_j   = smid[j];
        const bool before = (mid_j == mid) ? (j < i) : (descending ? (mid_j > mid) : (mid_j < mid));
        if (before) ++rank;
      }
    }

    order_out[base + rank] = base + i;
  }
#endif
}

template <bool BHSD = false, typename DQ_DTYPE = float>
__global__ void __launch_bounds__(256, 1)
    vsa_bwd_preprocess_kernel(const __nv_bfloat16* __restrict__ q,
                              const __nv_bfloat16* __restrict__ o,
                              const __nv_bfloat16* __restrict__ dout,
                              float* __restrict__ delta_rows, DQ_DTYPE* __restrict__ dqaccum,
                              __nv_bfloat16* __restrict__ qt, __nv_bfloat16* __restrict__ dot,
                              __nv_bfloat16* __restrict__ dk, __nv_bfloat16* __restrict__ dv,
                              const int* __restrict__ k2q_num, int num_samples, int num_heads,
                              int seqlen) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))
  __shared__ __align__(128) __nv_bfloat16 tile[PRE_TOKENS][SUB_COLS_BF16 + 4];
  const int num_kv_blocks_per_seq = seqlen / BLOCK;
  const int token_block_id        = (int)blockIdx.x;
  const int batch_head            = (int)blockIdx.y;
  const int batch = batch_head / num_heads, head = batch_head % num_heads;
  const size_t total_tokens = (size_t)num_samples * seqlen;
  const int token_begin     = token_block_id * PRE_TOKENS;

  uint4* dqaccum_zero_destination =
      reinterpret_cast<uint4*>(dqaccum + (size_t)batch_head * seqlen * HEAD_DIM +
                               (size_t)token_block_id * PRE_TOKENS * HEAD_DIM);
  const uint4 zero_uint4       = make_uint4(0u, 0u, 0u, 0u);
  constexpr int DQ_ZERO_CHUNKS = (PRE_TOKENS * HEAD_DIM * (int)sizeof(DQ_DTYPE) / 16) / 256;
  #pragma unroll
  for (int chunk = 0; chunk < DQ_ZERO_CHUNKS; ++chunk)
    dqaccum_zero_destination[chunk * 256 + threadIdx.x] = zero_uint4;

  static_assert(PRE_TOKENS == 2 * BLOCK, "one preprocess CTA covers two kv64 blocks");
  constexpr int KV_ZERO_CHUNKS = (BLOCK * HEAD_DIM / 8) / 256;
  #pragma unroll
  for (int kv_block_in_cta = 0; kv_block_in_cta < PRE_QBLOCKS; ++kv_block_in_cta) {
    const int kv_block_id_in_seq = token_block_id * PRE_QBLOCKS + kv_block_in_cta;
    if (k2q_num[batch_head * num_kv_blocks_per_seq + kv_block_id_in_seq] == 0) {
      #pragma unroll
      for (int chunk = 0; chunk < KV_ZERO_CHUNKS; ++chunk) {
        const int vector_index = chunk * 256 + (int)threadIdx.x;
        const int row          = vector_index >> 4;
        const int column       = (vector_index & 15) * 8;
        const size_t token_element_offset =
            token_offset<BHSD>(batch, head, num_heads, seqlen, kv_block_id_in_seq * BLOCK + row) +
            column;
        *reinterpret_cast<uint4*>(dk + token_element_offset) = zero_uint4;
        *reinterpret_cast<uint4*>(dv + token_element_offset) = zero_uint4;
      }
    }
  }

  const int row_base        = (int)threadIdx.x >> 4;
  const int dimension_begin = ((int)threadIdx.x & 15) * 4;
  #pragma unroll
  for (int dimension_block = 0; dimension_block < HEAD_DIM; dimension_block += 64) {
    #pragma unroll
    for (int row_pass = 0; row_pass < 8; ++row_pass) {
      const int row                                          = row_base + row_pass * 16;
      *reinterpret_cast<uint2*>(&tile[row][dimension_begin]) = *reinterpret_cast<const uint2*>(
          q + token_offset<BHSD>(batch, head, num_heads, seqlen, token_begin + row) +
          dimension_block + dimension_begin);
    }
    __syncthreads();

    const int dimension         = (int)threadIdx.x >> 2;
    const int token_group_begin = ((int)threadIdx.x & 3) * 4;
    #pragma unroll
    for (int row_pass = 0; row_pass < 8; ++row_pass) {
      const int row = token_group_begin + row_pass * 16;
      uint2 packed_tokens;
      uint16_t* packed_halves = reinterpret_cast<uint16_t*>(&packed_tokens);
      #pragma unroll
      for (int token_in_group = 0; token_in_group < 4; ++token_in_group)
        packed_halves[token_in_group] =
            *reinterpret_cast<const uint16_t*>(&tile[row + token_in_group][dimension]);
      __nv_bfloat16* transposed_row =
          qt + ((size_t)head * HEAD_DIM + dimension_block + dimension) * total_tokens;
      *reinterpret_cast<uint2*>(transposed_row + (size_t)batch * seqlen + token_begin + row) =
          packed_tokens;
    }
    __syncthreads();
  }

  float delta_accumulator[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  #pragma unroll
  for (int dimension_block = 0; dimension_block < HEAD_DIM; dimension_block += 64) {
    #pragma unroll
    for (int row_pass = 0; row_pass < 8; ++row_pass) {
      const int row = row_base + row_pass * 16;
      const size_t token_element_offset =
          token_offset<BHSD>(batch, head, num_heads, seqlen, token_begin + row) + dimension_block +
          dimension_begin;
      const uint2 dout_vector = *reinterpret_cast<const uint2*>(dout + token_element_offset);
      *reinterpret_cast<uint2*>(&tile[row][dimension_begin]) = dout_vector;

      const uint2 o_vector             = *reinterpret_cast<const uint2*>(o + token_element_offset);
      const __nv_bfloat162* dout_pairs = reinterpret_cast<const __nv_bfloat162*>(&dout_vector);
      const __nv_bfloat162* o_pairs    = reinterpret_cast<const __nv_bfloat162*>(&o_vector);
      #pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const float2 dout_pair_as_float = __bfloat1622float2(dout_pairs[pair]);
        const float2 o_pair_as_float    = __bfloat1622float2(o_pairs[pair]);
        delta_accumulator[row_pass] +=
            o_pair_as_float.x * dout_pair_as_float.x + o_pair_as_float.y * dout_pair_as_float.y;
      }
    }
    __syncthreads();

    const int dimension         = (int)threadIdx.x >> 2;
    const int token_group_begin = ((int)threadIdx.x & 3) * 4;
    #pragma unroll
    for (int row_pass = 0; row_pass < 8; ++row_pass) {
      const int row = token_group_begin + row_pass * 16;
      uint2 packed_tokens;
      uint16_t* packed_halves = reinterpret_cast<uint16_t*>(&packed_tokens);
      #pragma unroll
      for (int token_in_group = 0; token_in_group < 4; ++token_in_group)
        packed_halves[token_in_group] =
            *reinterpret_cast<const uint16_t*>(&tile[row + token_in_group][dimension]);
      __nv_bfloat16* transposed_row =
          dot + ((size_t)head * HEAD_DIM + dimension_block + dimension) * total_tokens;
      *reinterpret_cast<uint2*>(transposed_row + (size_t)batch * seqlen + token_begin + row) =
          packed_tokens;
    }
    __syncthreads();
  }

  #pragma unroll
  for (int row_pass = 0; row_pass < 8; ++row_pass) {
    float delta_sum = delta_accumulator[row_pass];
    #pragma unroll
    for (int shuffle_offset = 8; shuffle_offset > 0; shuffle_offset >>= 1)
      delta_sum += __shfl_down_sync(0xffffffffu, delta_sum, shuffle_offset, 16);
    if (((int)threadIdx.x & 15) == 0)
      delta_rows[(size_t)batch_head * seqlen + token_begin + row_base + row_pass * 16] = delta_sum;
  }
#endif
}

template <bool BHSD = false, typename DQ_DTYPE = float>
__global__ void __launch_bounds__(128, 1)
    vsa_bwd_postprocess_kernel(const DQ_DTYPE* __restrict__ dqaccum, __nv_bfloat16* __restrict__ dq,
                               int num_heads, int seqlen, float sm_scale) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))
  const int q_block_id = (int)blockIdx.x;
  const int batch_head = (int)blockIdx.y;
  const int batch = batch_head / num_heads, head = batch_head % num_heads;
  const DQ_DTYPE* dqaccum_block = dqaccum + ((size_t)batch_head * (seqlen / BLOCK) + q_block_id) *
                                                DQConfig<DQ_DTYPE>::DQ_BLOCK_ELEMS;
  const int row                 = (int)threadIdx.x & 63;
  const int dimension_half      = (int)threadIdx.x >> 6;

  uint32_t dq_packed_bf16_pairs[32];
  #pragma unroll
  for (int dimension_in_half = 0; dimension_in_half < 64; dimension_in_half += 4) {
    const int dimension = dimension_half * 64 + dimension_in_half;
    const DQ_DTYPE* dqaccum_element =
        dqaccum_block +
        (dimension / DQConfig<DQ_DTYPE>::COLS) * (BLOCK * DQConfig<DQ_DTYPE>::COLS) +
        ((row & 32) >> 5) * DQConfig<DQ_DTYPE>::WARP_ELEMS +
        ((dimension % DQConfig<DQ_DTYPE>::COLS) >> 2) * 128 + (row & 31) * 4;
    float2 dq_pair_low, dq_pair_high;
    if constexpr (sizeof(DQ_DTYPE) == 2) {
      const uint2 dq_four_halves = *reinterpret_cast<const uint2*>(dqaccum_element);
      dq_pair_low  = __half22float2(*reinterpret_cast<const __half2*>(&dq_four_halves.x));
      dq_pair_high = __half22float2(*reinterpret_cast<const __half2*>(&dq_four_halves.y));
    } else {
      const float4 dq_four_floats = *reinterpret_cast<const float4*>(dqaccum_element);
      dq_pair_low                 = make_float2(dq_four_floats.x, dq_four_floats.y);
      dq_pair_high                = make_float2(dq_four_floats.z, dq_four_floats.w);
    }
    dq_packed_bf16_pairs[dimension_in_half / 2 + 0] =
        cvt_f32x2_to_bf16x2(dq_pair_low.x * sm_scale, dq_pair_low.y * sm_scale);
    dq_packed_bf16_pairs[dimension_in_half / 2 + 1] =
        cvt_f32x2_to_bf16x2(dq_pair_high.x * sm_scale, dq_pair_high.y * sm_scale);
  }

  const int token       = q_block_id * BLOCK + row;
  uint4* dq_destination = reinterpret_cast<uint4*>(
      dq + token_offset<BHSD>(batch, head, num_heads, seqlen, token) + dimension_half * 64);
  const uint4* dq_packed_uint4 = reinterpret_cast<const uint4*>(dq_packed_bf16_pairs);
  #pragma unroll
  for (int vector_index = 0; vector_index < 8; ++vector_index)
    dq_destination[vector_index] = dq_packed_uint4[vector_index];
#endif
}

}

#endif
