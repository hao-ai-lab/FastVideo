// block_sparse_bwd_kernel_sm100a_blk128.cuh -- VSA block-sparse attention BACKWARD,
// 128-token blocks, sm_100a (one CTA per kv block). Warp-specialized: load /
// MMA (tcgen05) / softmax (P^T, dS^T) / epilogue (dQ drain). Three kernels: preprocess
// (Delta, dqaccum zero), main (dK, dV, dQ partials), postprocess (dQ unscramble + scale).
#ifndef BLOCK_SPARSE_VSA_BWD_KERNEL_SM100A_BLK128_CUH
#define BLOCK_SPARSE_VSA_BWD_KERNEL_SM100A_BLK128_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>
#include "primitives.cuh"

namespace vsa_bwd_blk128 {

#ifndef VSA_BHSD
#define VSA_BHSD false
#endif

enum class Sched { NON_PERSISTENT, STATIC_PERSISTENT, CLC };
#ifndef KERNEL_SCHED
#define KERNEL_SCHED Sched::NON_PERSISTENT
#endif

#ifndef KERNEL_PDL
#define KERNEL_PDL true
#endif

constexpr int BLOCK           = 128;
constexpr int KV_TILE         = BLOCK;
constexpr int Q_TILE          = BLOCK;
constexpr int HEAD_DIM        = 128;
constexpr int SUB_COLS_BF16   = 64;
constexpr int SUB_COLS_BYTES  = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
constexpr int HD_SUBTILES     = HEAD_DIM / SUB_COLS_BF16;
constexpr int KV_SUB_COLS_BYTES = KV_TILE * SUB_COLS_BYTES;
constexpr int KV_TILE_BYTES     = HD_SUBTILES * KV_SUB_COLS_BYTES;
constexpr int Q_SUB_COLS_BYTES  = Q_TILE * SUB_COLS_BYTES;
constexpr int Q_TILE_BYTES      = HD_SUBTILES * Q_SUB_COLS_BYTES;
constexpr int NUM_Q_STAGES      = 2;
constexpr int DST_TILE_BYTES    = KV_TILE * SUB_COLS_BYTES;
constexpr int DST_BYTES         = 2 * DST_TILE_BYTES;
constexpr int MMA_K               = 16;
constexpr int K_ATOMS_PER_SUBTILE = SUB_COLS_BF16 / MMA_K;
constexpr int K_ATOMS_PER_Q_HALF  = SUB_COLS_BF16 / MMA_K;
constexpr int K_ATOMS_PER_KV_TILE = KV_TILE / MMA_K;
constexpr int BF16X2_COLS_PER_K16 = MMA_K / 2;

struct DQConfig {
  static constexpr int COLS = 32;

  static constexpr int DQ_ONE_PUSH_BYTES = Q_TILE * COLS * (int)sizeof(float);
  static constexpr int DQ_STAGE_BYTES    = DQ_ONE_PUSH_BYTES;
  static constexpr int DQ_STAGE_BUFFERS  = 2;
  static constexpr int DQ_BLOCK_ELEMS    = Q_TILE * HEAD_DIM;
};
using DQ = DQConfig;

constexpr int N_WARPS = 16;
constexpr int W_EPI0 = 0, W_SOFTMAX0 = 4, W_MMA = 12, W_LOAD = 13;
constexpr int W_SCHED = 14;
constexpr int CLC_STAGES   = 2;
constexpr int CLC_ARRIVALS = 15;

constexpr int NUM_BARS   = 4 * NUM_Q_STAGES + 14 + 2 * CLC_STAGES;
constexpr int SMEM_TOTAL = 2 * KV_TILE_BYTES + NUM_Q_STAGES * Q_TILE_BYTES + Q_TILE_BYTES +
                           DST_BYTES + DQ::DQ_STAGE_BUFFERS * DQ::DQ_STAGE_BYTES +
                           (NUM_Q_STAGES + 1) * Q_TILE * (int)sizeof(float) + NUM_BARS * 8 +
                           CLC_STAGES * 16 + 48;

constexpr int ST_COLS      = Q_TILE;
constexpr int ST_HALF_COLS = SUB_COLS_BF16;
constexpr int DV_COLS      = HEAD_DIM;
constexpr int DK_COLS      = HEAD_DIM;
constexpr int TMEM_TOTAL   = ST_COLS + DV_COLS + ST_COLS + DK_COLS;
static_assert(ST_COLS == 2 * ST_HALF_COLS, "two q-halves per tile");
static_assert(TMEM_TOTAL == 512, "TMEM map must fill exactly 512 columns");

extern __shared__ __align__(1024) uint8_t bwd_smem[];

template <bool BHSD>
__device__ __forceinline__ size_t token_offset(int batch, int head, int num_heads, int seqlen,
                                               int t) {
  if constexpr (BHSD)
    return ((size_t)(batch * num_heads + head) * seqlen + t) * HEAD_DIM;
  else
    return ((size_t)(batch * seqlen + t) * num_heads + head) * HEAD_DIM;
}

struct WorkItem {
  int batch;
  int head;
  int kv_block_id_in_seq;
  int batch_head;
  const int* local_k2q_idx;

  int local_k2q_num;
};

__device__ __forceinline__ WorkItem decode_workitem(int batch_head, int kv_block_id_in_seq,
                                                    const int* __restrict__ k2q_idx,
                                                    const int* __restrict__ k2q_num,
                                                    int max_q_blocks, int num_heads,
                                                    int num_kv_blocks_per_seq) {
  WorkItem it;
  it.batch_head         = batch_head;
  it.batch              = batch_head / num_heads;
  it.head               = batch_head % num_heads;
  it.kv_block_id_in_seq = kv_block_id_in_seq;
  const int item        = batch_head * num_kv_blocks_per_seq + kv_block_id_in_seq;
  it.local_k2q_idx      = k2q_idx + (size_t)item * (size_t)max_q_blocks;
  it.local_k2q_num      = k2q_num[item];
  return it;
}

template <bool BHSD = false, Sched SCHED = Sched::NON_PERSISTENT>
__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
    vsa_bwd_main_kernel(const __grid_constant__ CUtensorMap tmap_q,
                        const __grid_constant__ CUtensorMap tmap_k,
                        const __grid_constant__ CUtensorMap tmap_v,
                        const __grid_constant__ CUtensorMap tmap_do,
                        const __grid_constant__ CUtensorMap tmap_dk,
                        const __grid_constant__ CUtensorMap tmap_dv, float* __restrict__ dqaccum,
                        const float* __restrict__ lse_rows, const float* __restrict__ delta_rows,
                        const int* __restrict__ k2q_idx, const int* __restrict__ k2q_num,
                        int max_q_blocks, const int* __restrict__ workitem_remap,
                        const int* __restrict__ variable_block_sizes, int num_samples,
                        int num_heads, int seqlen, float scale_log2, float sm_scale) {
#if !defined(__CUDA_ARCH__) || \
    ((__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL)) || \
     (__CUDA_ARCH__ == 1030 && defined(__CUDA_ARCH_FEAT_SM103_ALL)))
  constexpr bool PERSISTENT = SCHED != Sched::NON_PERSISTENT;
  constexpr bool CLC        = SCHED == Sched::CLC;
  const int num_kv_blocks_per_seq = seqlen / BLOCK;
  [[maybe_unused]] const int total_workitems = num_samples * num_heads * num_kv_blocks_per_seq;
  uint8_t* sK                = bwd_smem;
  uint8_t* sV                = sK + KV_TILE_BYTES;
  uint8_t* sQ[NUM_Q_STAGES]  = {sV + KV_TILE_BYTES, sV + KV_TILE_BYTES + Q_TILE_BYTES};
  uint8_t* sDO               = sQ[0] + NUM_Q_STAGES * Q_TILE_BYTES;
  __nv_bfloat16* sDST        = reinterpret_cast<__nv_bfloat16*>(sDO + Q_TILE_BYTES);
  uint8_t* sDQ_STAGE_bytes   = sDO + Q_TILE_BYTES + DST_BYTES;
  float* sDQ_STAGE[DQ::DQ_STAGE_BUFFERS] = {
      reinterpret_cast<float*>(sDQ_STAGE_bytes),
      reinterpret_cast<float*>(sDQ_STAGE_bytes + DQ::DQ_STAGE_BYTES)};
  float* sLSE =
      reinterpret_cast<float*>(sDQ_STAGE_bytes + DQ::DQ_STAGE_BUFFERS * DQ::DQ_STAGE_BYTES);
  float* sDelta = sLSE + NUM_Q_STAGES * Q_TILE;

  uint64_t* full_bar_q      = reinterpret_cast<uint64_t*>(sDelta + Q_TILE);
  uint64_t* empty_bar_q     = full_bar_q + NUM_Q_STAGES;
  uint64_t* full_bar_do     = empty_bar_q + NUM_Q_STAGES;
  uint64_t* empty_bar_do    = full_bar_do + 1;
  uint64_t* full_bar_lse    = empty_bar_do + 1;
  uint64_t* empty_bar_lse   = full_bar_lse + NUM_Q_STAGES;
  uint64_t* full_bar_delta  = empty_bar_lse + NUM_Q_STAGES;
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
  uint32_t* tmem_slot       = clc_response + CLC_STAGES * 4;

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

  if (tid == 0) {
    #pragma unroll
    for (int s = 0; s < NUM_Q_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar_q[s]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_q[s]), 1);
    }
    mbarrier_init(smem_ptr_u32(full_bar_do), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_do), 1);
    #pragma unroll
    for (int s = 0; s < NUM_Q_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar_lse[s]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_lse[s]), 8);
    }
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
    if constexpr (PERSISTENT) {
      mbarrier_init(smem_ptr_u32(empty_bar_kv), 1);
      mbarrier_init(smem_ptr_u32(empty_bar_epi), 8 * 32);
    }
    if constexpr (CLC) {
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

  if (warp_id == W_LOAD) {
    setmaxnreg_dec<88>();

    if constexpr (KERNEL_PDL) griddepcontrol_wait();

    EmptyPhaseTracker<NUM_Q_STAGES> q_empty_ph, lse_empty_ph;
    EmptyPhaseTracker<1> do_empty_ph, delta_empty_ph;
    [[maybe_unused]] EmptyPhaseTracker<1> epi_empty_ph, kv_empty_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;

    int workitem_id = PERSISTENT ? (int)blockIdx.x : 0;
    do {
      int batch_head, kv_block_id_in_seq;
      if constexpr (PERSISTENT) {
        const int real_id  = workitem_remap ? workitem_remap[workitem_id] : workitem_id;
        batch_head         = real_id / num_kv_blocks_per_seq;
        kv_block_id_in_seq = real_id % num_kv_blocks_per_seq;
      } else {
        batch_head         = (int)blockIdx.y;
        kv_block_id_in_seq = (int)blockIdx.x;
      }
      const WorkItem it = decode_workitem(batch_head, kv_block_id_in_seq, k2q_idx, k2q_num,
                                          max_q_blocks, num_heads, num_kv_blocks_per_seq);
      if (it.local_k2q_num != 0) {
        auto load_tile = [&](uint8_t* dst, const CUtensorMap* map, uint64_t* full_bar,
                             int token_begin) {
          static_assert(Q_SUB_COLS_BYTES == KV_SUB_COLS_BYTES,
                        "one subtile stride for Q/dO and K/V");
          #pragma unroll
          for (int s = 0; s < HD_SUBTILES; ++s) {
            if constexpr (BHSD)
              tma_load_4d(smem_ptr_u32(dst + s * Q_SUB_COLS_BYTES), map, smem_ptr_u32(full_bar),
                          0, token_begin, s, it.batch_head);
            else
              tma_load_3d(smem_ptr_u32(dst + s * Q_SUB_COLS_BYTES), map, smem_ptr_u32(full_bar),
                          0, it.batch * seqlen + token_begin, it.head * HD_SUBTILES + s);
          }
        };
        auto load_lse = [&](int qblock) {
          const int stage = lse_empty_ph.get_stage();
          mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_lse[stage]),
                                       lse_empty_ph.get_phase());
          lse_empty_ph.advance();

          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_lse[stage]),
                                      Q_TILE * (int)sizeof(float));
            cpasync_bulk_load_mbarrier(
                smem_ptr_u32(sLSE + stage * Q_TILE),
                lse_rows + (size_t)it.batch_head * seqlen + (size_t)qblock * Q_TILE,
                Q_TILE * sizeof(float), smem_ptr_u32(&full_bar_lse[stage]));
          }
        };
        auto load_delta = [&](int qblock) {
          mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_delta), delta_empty_ph.get_phase());
          delta_empty_ph.advance();

          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(full_bar_delta), Q_TILE * (int)sizeof(float));
            cpasync_bulk_load_mbarrier(
                smem_ptr_u32(sDelta),
                delta_rows + (size_t)it.batch_head * seqlen + (size_t)qblock * Q_TILE,
                Q_TILE * sizeof(float), smem_ptr_u32(full_bar_delta));
          }
        };

        auto load_q = [&](int qblock, auto with_kv_const) {
          constexpr bool with_kv = decltype(with_kv_const)::value;
          const int stage = q_empty_ph.get_stage();
          mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[stage]), q_empty_ph.get_phase());
          q_empty_ph.advance();

          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[stage]),
                                      Q_TILE_BYTES + (with_kv ? KV_TILE_BYTES : 0));
            load_tile(sQ[stage], &tmap_q, &full_bar_q[stage], qblock * Q_TILE);
            if constexpr (with_kv)
              load_tile(sK, &tmap_k, &full_bar_q[stage], it.kv_block_id_in_seq * KV_TILE);
          }
        };

        auto load_do = [&](int qblock, auto with_kv_const) {
          constexpr bool with_kv = decltype(with_kv_const)::value;
          mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_do), do_empty_ph.get_phase());
          do_empty_ph.advance();

          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(full_bar_do),
                                      Q_TILE_BYTES + (with_kv ? KV_TILE_BYTES : 0));
            load_tile(sDO, &tmap_do, full_bar_do, qblock * Q_TILE);
            if constexpr (with_kv)
              load_tile(sV, &tmap_v, full_bar_do, it.kv_block_id_in_seq * KV_TILE);
          }
        };

        if constexpr (PERSISTENT) {
          mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_epi), epi_empty_ph.get_phase());
          epi_empty_ph.advance();

          mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_kv), kv_empty_ph.get_phase());
          kv_empty_ph.advance();
        }

        load_q(it.local_k2q_idx[0], std::true_type{});
        load_lse(it.local_k2q_idx[0]);
        load_do(it.local_k2q_idx[0], std::true_type{});
        load_delta(it.local_k2q_idx[0]);
        for (int j = 1; j < it.local_k2q_num; ++j) {

          const int qblock = it.local_k2q_idx[j];
          load_q(qblock, std::false_type{});
          load_lse(qblock);
          load_do(qblock, std::false_type{});
          load_delta(qblock);
        }
      }

      if constexpr (CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        workitem_id = next.valid ? (int)next.n_tile : -1;
      } else if constexpr (PERSISTENT) {
        workitem_id += (int)gridDim.x;
        if (workitem_id >= total_workitems) workitem_id = -1;
      }
    } while (PERSISTENT && workitem_id >= 0);
    return;
  }
  else if (warp_id == W_MMA) {
    setmaxnreg_dec<88>();

    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), TMEM_TOTAL);
    bar_sync<10>(416);
    const uint32_t tmem_base    = *tmem_slot;
    const uint32_t tmem_st      = tmem_base;
    const uint32_t tmem_dv      = tmem_st + ST_COLS;
    const uint32_t tmem_dpt     = tmem_dv + DV_COLS;
    const uint32_t tmem_dk      = tmem_dpt + ST_COLS;
    const uint32_t tmem_pt_bf16 = tmem_st, tmem_dst_bf16 = tmem_dpt, tmem_dq = tmem_dpt;
    const uint32_t lead         = elect_one_sync() ? 1u : 0u;

    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    auto make_smem_desc = [](const void* smem, uint32_t leading_byte_offset) {
      return build_smem_desc_blackwell(smem_ptr_u32(smem), DESC_SBO, leading_byte_offset,
                                       SmemSwizzleBlackwell::B128);
    };
    const uint64_t desc_k      = make_smem_desc(sK, DESC_LBO);
    const uint64_t desc_v      = make_smem_desc(sV, DESC_LBO);
    const uint64_t desc_q0     = make_smem_desc(sQ[0], DESC_LBO);
    const uint64_t desc_do     = make_smem_desc(sDO, DESC_LBO);
    const uint64_t desc_q0_mn  = make_smem_desc(sQ[0], Q_SUB_COLS_BYTES);
    const uint64_t desc_do_mn  = make_smem_desc(sDO, Q_SUB_COLS_BYTES);
    const uint64_t desc_k_mn   = make_smem_desc(sK, KV_SUB_COLS_BYTES);
    const uint64_t desc_dst_mn = make_smem_desc(sDST, DST_TILE_BYTES);

    constexpr uint64_t K16_COLS_DELTA = (MMA_K * (int)sizeof(__nv_bfloat16)) >> 4;
    constexpr uint32_t K16_ROWS_DELTA = (MMA_K * SUB_COLS_BYTES) >> 4;
    constexpr uint64_t Q_SUB_DELTA    = Q_SUB_COLS_BYTES >> 4;
    constexpr uint64_t KV_SUB_DELTA   = KV_SUB_COLS_BYTES >> 4;
    constexpr uint64_t Q_STAGE_DELTA  = Q_TILE_BYTES >> 4;

    const uint32_t idesc_st_dpt = make_idesc_bf16_f32(KV_TILE, Q_TILE, false, false);
    const uint32_t idesc_dv_dk  = make_idesc_bf16_f32(KV_TILE, HEAD_DIM, false, true);
    const uint32_t idesc_dq     = make_idesc_bf16_f32(Q_TILE, HEAD_DIM, true, true);

    PhaseTracker<NUM_Q_STAGES> q_full_ph;
    PhaseTracker<1> do_full_ph, pt_ph, dst_ph, dq_empty_ph;
    [[maybe_unused]] EmptyPhaseTracker<1> epi_empty_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = PERSISTENT ? (int)blockIdx.x : 0;

    auto gemm12_st_dpt = [&](auto is_st_const, int stage) {
      constexpr bool is_st    = decltype(is_st_const)::value;
      const uint32_t tmem_acc = is_st ? tmem_st : tmem_dpt;
      const uint64_t da_base  = is_st ? desc_k : desc_v;
      const uint64_t db_base  = is_st ? desc_q0 + (uint64_t)stage * Q_STAGE_DELTA : desc_do;
      uint64_t* commit_bar    = is_st ? full_bar_st : full_bar_dpt;
      if constexpr (is_st) {
        mbarrier_wait_parity(smem_ptr_u32(&full_bar_q[stage]), q_full_ph.get_phase());
      } else {
        mbarrier_wait_parity(smem_ptr_u32(full_bar_do), do_full_ph.get_phase());
        do_full_ph.advance();
      }

      #pragma unroll
      for (int s = 0; s < HD_SUBTILES; ++s) {
        #pragma unroll
        for (int ki = 0; ki < K_ATOMS_PER_SUBTILE; ++ki) {
          const bool accumulate = (s | ki) != 0;
          tcgen05_mma_f16_ss_lead(lead, tmem_acc, da_base + s * KV_SUB_DELTA + ki * K16_COLS_DELTA,
                                  db_base + s * Q_SUB_DELTA + ki * K16_COLS_DELTA, idesc_st_dpt,
                                  accumulate);
        }
      }

      tcgen05_commit1_lead(lead, smem_ptr_u32(commit_bar));
    };

    auto gemm35_dv_dk = [&](auto is_dv_const, int stage, bool first) {
      constexpr bool is_dv       = decltype(is_dv_const)::value;
      const uint32_t tmem_acc    = is_dv ? tmem_dv : tmem_dk;
      const uint32_t tmem_a_base = is_dv ? tmem_pt_bf16 : tmem_dst_bf16;
      uint64_t db          = is_dv ? desc_do_mn : desc_q0_mn + (uint64_t)stage * Q_STAGE_DELTA;
      uint64_t* commit_bar = is_dv ? empty_bar_do : &empty_bar_q[stage];
      #pragma unroll
      for (int q_half = 0; q_half < 2; ++q_half) {
        #pragma unroll
        for (int ki = 0; ki < K_ATOMS_PER_Q_HALF; ++ki) {
          const uint32_t tmem_a = tmem_a_base + (uint32_t)(q_half * ST_HALF_COLS +
                                                           ki * BF16X2_COLS_PER_K16);
          const bool accumulate = !(first && (q_half | ki) == 0);
          tcgen05_mma_f16_ts_1sm_lead(lead, tmem_acc, tmem_a, db, idesc_dv_dk, accumulate);
          smem_desc_add_lo(db, K16_ROWS_DELTA);
        }
      }

      tcgen05_commit1_lead(lead, smem_ptr_u32(commit_bar));
    };

    auto gemm4_dq = [&]() {
      uint64_t adst = desc_dst_mn;
      uint64_t bk   = desc_k_mn;
      #pragma unroll
      for (int ki = 0; ki < K_ATOMS_PER_KV_TILE; ++ki) {
        tcgen05_mma_f16_ss_lead(lead, tmem_dq, adst, bk, idesc_dq, ki != 0);
        smem_desc_add_lo(adst, K16_ROWS_DELTA);
        smem_desc_add_lo(bk, K16_ROWS_DELTA);
      }

      tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dq));
    };

    do {
      int batch_head, kv_block_id_in_seq;
      if constexpr (PERSISTENT) {
        const int real_id  = workitem_remap ? workitem_remap[workitem_id] : workitem_id;
        batch_head         = real_id / num_kv_blocks_per_seq;
        kv_block_id_in_seq = real_id % num_kv_blocks_per_seq;
      } else {
        batch_head         = (int)blockIdx.y;
        kv_block_id_in_seq = (int)blockIdx.x;
      }
      const WorkItem it = decode_workitem(batch_head, kv_block_id_in_seq, k2q_idx, k2q_num,
                                          max_q_blocks, num_heads, num_kv_blocks_per_seq);
      if (it.local_k2q_num != 0) {

        gemm12_st_dpt(std::true_type{}, q_full_ph.get_stage());

        if constexpr (PERSISTENT) {
          mbarrier_wait_parity(smem_ptr_u32(empty_bar_dq), dq_empty_ph.get_phase());
          dq_empty_ph.advance();
        }

        gemm12_st_dpt(std::false_type{}, 0);

        mbarrier_wait_parity(smem_ptr_u32(full_bar_pt), pt_ph.get_phase());
        pt_ph.advance();

        if constexpr (PERSISTENT) {
          mbarrier_wait_parity(smem_ptr_u32(empty_bar_epi), epi_empty_ph.get_phase());
          epi_empty_ph.advance();
        }

        gemm35_dv_dk(std::true_type{}, 0, true);

        for (int j = 0; j < it.local_k2q_num; ++j) {

          const int stage = q_full_ph.get_stage();
          q_full_ph.advance();
          const int next_stage = q_full_ph.get_stage();
          const bool last      = j + 1 == it.local_k2q_num;

          if (!last) gemm12_st_dpt(std::true_type{}, next_stage);

          if (last) {
            tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dv));
          }
          mbarrier_wait_parity(smem_ptr_u32(full_bar_dst), dst_ph.get_phase());
          dst_ph.advance();
          gemm35_dv_dk(std::false_type{}, stage, j == 0);
          if (last) {
            tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dk));
          }
          gemm4_dq();
          if (!last) {
            mbarrier_wait_parity(smem_ptr_u32(empty_bar_dq), dq_empty_ph.get_phase());
            dq_empty_ph.advance();
            gemm12_st_dpt(std::false_type{}, 0);
            mbarrier_wait_parity(smem_ptr_u32(full_bar_pt), pt_ph.get_phase());
            pt_ph.advance();
            gemm35_dv_dk(std::true_type{}, 0, false);
          }
        }

        if constexpr (PERSISTENT) {
          tcgen05_commit1_lead(lead, smem_ptr_u32(empty_bar_kv));
        }
      }

      if constexpr (CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        workitem_id = next.valid ? (int)next.n_tile : -1;
      } else if constexpr (PERSISTENT) {
        workitem_id += (int)gridDim.x;
        if (workitem_id >= total_workitems) workitem_id = -1;
      }
    } while (PERSISTENT && workitem_id >= 0);
    tcgen05_relinquish_alloc_permit<1>();
    bar_sync<10>(416);
    tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL);
    return;
  }
  else if (warp_id == W_SCHED) {

    if constexpr (CLC) {
      setmaxnreg_dec<88>();
      int prod_stage = 0; uint32_t prod_phase = 1;
      int cons_stage = 0; uint32_t cons_phase = 0;
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
    } else {
      setmaxnreg_dec<24>();
    }
    return;
  }
  else if (warp_id >= W_SOFTMAX0 && warp_id < W_MMA) {
    setmaxnreg_inc<136>();
    bar_sync<10>(416);
    const uint32_t tmem_base    = *tmem_slot;
    const uint32_t tmem_st      = tmem_base;
    const uint32_t tmem_dv      = tmem_st + ST_COLS;
    const uint32_t tmem_dpt     = tmem_dv + DV_COLS;
    const uint32_t tmem_dk      = tmem_dpt + ST_COLS;
    const uint32_t tmem_pt_bf16 = tmem_st, tmem_dst_bf16 = tmem_dpt;

    const int softmax_warp_id = warp_id - W_SOFTMAX0;
    const int lane_group      = softmax_warp_id & 3;
    const int col_half        = softmax_warp_id >> 2;
    const int kv_row          = lane_group * 32 + lane;

    const uint32_t tmem_lane_base     = (uint32_t)(lane_group * 32) << 16;
    const uint32_t tmem_f32_offset    = tmem_lane_base + (uint32_t)(col_half * ST_HALF_COLS);
    const uint32_t tmem_bf16x2_offset = tmem_f32_offset;

    constexpr int CHUNK_BF16     = 16 / (int)sizeof(__nv_bfloat16);
    constexpr int CHUNKS_PER_ROW = SUB_COLS_BF16 / CHUNK_BF16;

    __nv_bfloat16* sdst_row =
        sDST + (size_t)col_half * KV_TILE * SUB_COLS_BF16 + kv_row * SUB_COLS_BF16;

    PhaseTracker<1> st_ph, dpt_ph, delta_ph, dv_ph, dk_ph;
    PhaseTracker<NUM_Q_STAGES> lse_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = PERSISTENT ? (int)blockIdx.x : 0;

    do {
      int batch_head, kv_block_id_in_seq;
      if constexpr (PERSISTENT) {
        const int real_id  = workitem_remap ? workitem_remap[workitem_id] : workitem_id;
        batch_head         = real_id / num_kv_blocks_per_seq;
        kv_block_id_in_seq = real_id % num_kv_blocks_per_seq;
      } else {
        batch_head         = (int)blockIdx.y;
        kv_block_id_in_seq = (int)blockIdx.x;
      }
      const WorkItem it = decode_workitem(batch_head, kv_block_id_in_seq, k2q_idx, k2q_num,
                                          max_q_blocks, num_heads, num_kv_blocks_per_seq);

      auto store_dv_dk_tile = [&](uint32_t tmem_acc, auto apply_sm_scale_const, uint8_t* bounce,
                                  const CUtensorMap* map) {
        constexpr bool apply_sm_scale = decltype(apply_sm_scale_const)::value;
        uint8_t* bounce_subtile       = bounce + col_half * KV_SUB_COLS_BYTES;
        __nv_bfloat16* stage_row =
            reinterpret_cast<__nv_bfloat16*>(bounce_subtile) + kv_row * SUB_COLS_BF16;
        #pragma unroll
        for (int c0 = 0; c0 < ST_HALF_COLS; c0 += 32) {
          uint32_t acc_regs[32];
          tcgen05_ld_32x32b_x32(tmem_acc + tmem_f32_offset + (uint32_t)c0, acc_regs);
          tcgen05_fence_before_thread_sync();
          const float* acc = reinterpret_cast<const float*>(acc_regs);
          #pragma unroll
          for (int v = 0; v < 32 / CHUNK_BF16; ++v) {
            float value[CHUNK_BF16];
            #pragma unroll
            for (int e = 0; e < CHUNK_BF16; ++e)
              value[e] = apply_sm_scale ? acc[v * CHUNK_BF16 + e] * sm_scale
                                        : acc[v * CHUNK_BF16 + e];
            uint4 packed;
            packed.x = cvt_f32x2_to_bf16x2(value[0], value[1]);
            packed.y = cvt_f32x2_to_bf16x2(value[2], value[3]);
            packed.z = cvt_f32x2_to_bf16x2(value[4], value[5]);
            packed.w = cvt_f32x2_to_bf16x2(value[6], value[7]);
            const int chunk = c0 / CHUNK_BF16 + v;
            *reinterpret_cast<uint4*>(stage_row + ((chunk ^ (kv_row & 7)) * CHUNK_BF16)) = packed;
          }
        }
        fence_proxy_async_shared_cta();
        if (col_half == 0) bar_sync<12>(128); else bar_sync<13>(128);
        if (lane_group == 0 && elect_one_sync()) {
          if constexpr (BHSD)
            tma_store_4d(map, 0, it.kv_block_id_in_seq * KV_TILE, col_half, it.batch_head,
                         smem_ptr_u32(bounce_subtile));
          else
            tma_store_3d(map, 0, it.batch * seqlen + it.kv_block_id_in_seq * KV_TILE,
                         it.head * HD_SUBTILES + col_half, smem_ptr_u32(bounce_subtile));
          cp_async_bulk_commit_group();
        }
      };

      const bool kv_row_valid = kv_row < variable_block_sizes[it.kv_block_id_in_seq];

      for (int j = 0; j < it.local_k2q_num; ++j) {
        const int lse_stage     = lse_ph.get_stage();
        const float2* lse2   = reinterpret_cast<const float2*>(sLSE + lse_stage * Q_TILE +
                                                               col_half * ST_HALF_COLS);
        const float2* delta2 = reinterpret_cast<const float2*>(sDelta + col_half * ST_HALF_COLS);

        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_lse[lse_stage]), lse_ph.get_phase());
        lse_ph.advance();

        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_st), st_ph.get_phase());
        st_ph.advance();

        uint32_t st_regs[ST_HALF_COLS];
        uint32_t pt_bf16x2[ST_HALF_COLS / 2];
        float2* pt_fp32 = reinterpret_cast<float2*>(st_regs);
        tcgen05_ld_32x32b_x64(tmem_st + tmem_f32_offset, st_regs);
        tcgen05_fence_before_thread_sync();
        const float2 scale2 = f32x2_splat(scale_log2);
        #pragma unroll
        for (int c = 0; c < ST_HALF_COLS / 2; ++c) {
          const float2 z = ffma2(pt_fp32[c], scale2, make_float2(-lse2[c].x, -lse2[c].y));
          float2 p       = make_float2(ex2_approx_f32(z.x), ex2_approx_f32(z.y));
          if (!kv_row_valid) p = make_float2(0.f, 0.f);
          pt_fp32[c]   = p;
          pt_bf16x2[c] = cvt_f32x2_to_bf16x2(p.x, p.y);
        }
        tcgen05_st_32x32b_x32(tmem_pt_bf16 + tmem_bf16x2_offset, pt_bf16x2);
        tcgen05_wait_st();
        tcgen05_fence_before_thread_sync();
        if (elect_one_sync()) {
          mbarrier_arrive(smem_ptr_u32(full_bar_pt));
          mbarrier_arrive(smem_ptr_u32(&empty_bar_lse[lse_stage]));
        }

        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_delta), delta_ph.get_phase());
        delta_ph.advance();

        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dpt), dpt_ph.get_phase());
        dpt_ph.advance();

        uint32_t dpt_regs[ST_HALF_COLS];
        tcgen05_ld_32x32b_x64(tmem_dpt + tmem_f32_offset, dpt_regs);
        tcgen05_fence_before_thread_sync();
        const float2* dpt2 = reinterpret_cast<const float2*>(dpt_regs);
        #pragma unroll
        for (int c = 0; c < ST_HALF_COLS / 2; ++c) {
          const float2 ds =
              fmul2(pt_fp32[c], fadd2(dpt2[c], make_float2(-delta2[c].x, -delta2[c].y)));
          st_regs[c] = cvt_f32x2_to_bf16x2(ds.x, ds.y);
        }
        uint32_t (&dst_bf16x2)[ST_HALF_COLS / 2] =
            reinterpret_cast<uint32_t (&)[ST_HALF_COLS / 2]>(st_regs);
        tcgen05_st_32x32b_x32(tmem_dst_bf16 + tmem_bf16x2_offset, dst_bf16x2);
        const uint4* dst_chunks = reinterpret_cast<const uint4*>(dst_bf16x2);
        #pragma unroll
        for (int v = 0; v < CHUNKS_PER_ROW; ++v)
          *reinterpret_cast<uint4*>(sdst_row + (v ^ (kv_row & 7)) * CHUNK_BF16) = dst_chunks[v];
        tcgen05_wait_st();
        tcgen05_fence_before_thread_sync();
        fence_proxy_async_shared_cta();
        if (elect_one_sync()) {
          mbarrier_arrive(smem_ptr_u32(full_bar_dst));
          mbarrier_arrive(smem_ptr_u32(empty_bar_delta));
        }
      }

      if (it.local_k2q_num != 0) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dv), dv_ph.get_phase());
        dv_ph.advance();

        store_dv_dk_tile(tmem_dv, std::false_type{}, sDO, &tmap_dv);

        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dk), dk_ph.get_phase());
        dk_ph.advance();

        store_dv_dk_tile(tmem_dk, std::true_type{}, sQ[0], &tmap_dk);

        if constexpr (PERSISTENT) {
          if (lane_group == 0 && elect_one_sync()) cp_async_bulk_wait_group_read<0>();
          bar_sync<14>(256);
          mbarrier_arrive(smem_ptr_u32(empty_bar_epi));
        }
      }

      if constexpr (CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        workitem_id = next.valid ? (int)next.n_tile : -1;
      } else if constexpr (PERSISTENT) {
        workitem_id += (int)gridDim.x;
        if (workitem_id >= total_workitems) workitem_id = -1;
      }
    } while (PERSISTENT && workitem_id >= 0);
    bar_sync<10>(416);
    return;
  }
  else if (warp_id < W_SOFTMAX0) {
    setmaxnreg_inc<152>();

    bar_sync<10>(416);
    const uint32_t tmem_base = *tmem_slot;
    const uint32_t tmem_dq = tmem_base + ST_COLS + DV_COLS;

    const int epi_warp_id         = warp_id - W_EPI0;
    const uint32_t tmem_lane_base = (uint32_t)(epi_warp_id * 32) << 16;
    const int q_row               = epi_warp_id * 32 + lane;
    const bool is_leader          = epi_warp_id == 0 && lane == 0;

    PhaseTracker<1> dq_full_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = PERSISTENT ? (int)blockIdx.x : 0;

    if constexpr (PERSISTENT) {
      if (elect_one_sync()) mbarrier_arrive(smem_ptr_u32(empty_bar_dq));
    }

    do {

      int batch_head, kv_block_id_in_seq;
      if constexpr (PERSISTENT) {
        const int real_id  = workitem_remap ? workitem_remap[workitem_id] : workitem_id;
        batch_head         = real_id / num_kv_blocks_per_seq;
        kv_block_id_in_seq = real_id % num_kv_blocks_per_seq;
      } else {
        batch_head         = (int)blockIdx.y;
        kv_block_id_in_seq = (int)blockIdx.x;
      }
      const WorkItem it = decode_workitem(batch_head, kv_block_id_in_seq, k2q_idx, k2q_num,
                                          max_q_blocks, num_heads, num_kv_blocks_per_seq);
      float* dqaccum_head =
          dqaccum + (size_t)it.batch_head * num_kv_blocks_per_seq * DQ::DQ_BLOCK_ELEMS;

      for (int j = 0; j < it.local_k2q_num; ++j) {
        float* dqaccum_block = dqaccum_head + (size_t)it.local_k2q_idx[j] * DQ::DQ_BLOCK_ELEMS;

        mbarrier_wait_parity(smem_ptr_u32(full_bar_dq), dq_full_ph.get_phase());
        dq_full_ph.advance();

        uint32_t dq_regs[HEAD_DIM];
        #pragma unroll
        for (int c = 0; c < HEAD_DIM / 64; ++c)
          tcgen05_ld_32x32b_x64(tmem_dq + tmem_lane_base + (uint32_t)(c * 64),
                                reinterpret_cast<uint32_t (&)[64]>(dq_regs[c * 64]));

        tcgen05_wait_ld();
        tcgen05_fence_before_thread_sync();
        if (elect_one_sync()) mbarrier_arrive(smem_ptr_u32(empty_bar_dq));

        #pragma unroll
        for (int hd_slice = 0; hd_slice < HEAD_DIM / DQ::COLS; ++hd_slice) {
          const int stage_buf   = hd_slice & 1;
          const float4* dq_row4 = reinterpret_cast<const float4*>(dq_regs + hd_slice * DQ::COLS);
          #pragma unroll
          for (int v4 = 0; v4 < DQ::COLS / 4; ++v4)
            *reinterpret_cast<float4*>(sDQ_STAGE[stage_buf] + v4 * Q_TILE * 4 + q_row * 4) =
                dq_row4[v4];
          fence_proxy_async_shared_cta();
          bar_sync<11>(128);
          if (is_leader) {
            const size_t slice_offset = (size_t)hd_slice * Q_TILE * DQ::COLS;
            cpasync_reduce_bulk_add_f32(dqaccum_block + slice_offset,
                                        smem_ptr_u32(sDQ_STAGE[stage_buf]), DQ::DQ_ONE_PUSH_BYTES);
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<1>();
          }
          bar_sync<11>(128);
        }
      }

      if (it.local_k2q_num == 0) {

        if (is_leader) cp_async_bulk_wait_group_read<0>();
        bar_sync<11>(128);
        static_assert(DQ::DQ_STAGE_BUFFERS * DQ::DQ_STAGE_BYTES == HD_SUBTILES * KV_SUB_COLS_BYTES,
                      "the dQ stage buffers hold one 128 kv x 128 hd bf16 tile");
        constexpr int ZERO_UINT4_PER_THREAD =
            DQ::DQ_STAGE_BUFFERS * DQ::DQ_STAGE_BYTES / (128 * 16);
        uint4* zero_tile = reinterpret_cast<uint4*>(sDQ_STAGE_bytes) + epi_warp_id * 32 + lane;
        #pragma unroll
        for (int v = 0; v < ZERO_UINT4_PER_THREAD; ++v)
          zero_tile[v * 128] = make_uint4(0u, 0u, 0u, 0u);
        fence_proxy_async_shared_cta();
        bar_sync<11>(128);
        if (is_leader) {
          #pragma unroll
          for (int which = 0; which < 2; ++which) {
            const CUtensorMap* map = (which == 0) ? &tmap_dk : &tmap_dv;
            #pragma unroll
            for (int s = 0; s < HD_SUBTILES; ++s) {
              const uint32_t src = smem_ptr_u32(sDQ_STAGE_bytes + (size_t)s * KV_SUB_COLS_BYTES);
              if constexpr (BHSD)
                tma_store_4d(map, 0, it.kv_block_id_in_seq * KV_TILE, s, it.batch_head, src);
              else
                tma_store_3d(map, 0, it.batch * seqlen + it.kv_block_id_in_seq * KV_TILE,
                             it.head * HD_SUBTILES + s, src);
            }
          }
          cp_async_bulk_commit_group();
          cp_async_bulk_wait_group_read<0>();
        }
        bar_sync<11>(128);
      }

      if constexpr (CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        workitem_id = next.valid ? (int)next.n_tile : -1;
      } else if constexpr (PERSISTENT) {
        workitem_id += (int)gridDim.x;
        if (workitem_id >= total_workitems) workitem_id = -1;
      }
    } while (PERSISTENT && workitem_id >= 0);

    if (is_leader) cp_async_bulk_wait_group_read<0>();
    bar_sync<11>(128);

    if constexpr (KERNEL_PDL) {
      if (is_leader) griddepcontrol_launch_dependents();
    }
    bar_sync<10>(416);
    return;
  }
  else {
    setmaxnreg_dec<24>();
    return;
  }
#endif
}

template <bool BHSD = false>
__global__ void __launch_bounds__(256, 1)
    vsa_bwd_preprocess_kernel(const __nv_bfloat16* __restrict__ o,
                              const __nv_bfloat16* __restrict__ dout,
                              float* __restrict__ delta_rows, float* __restrict__ dqaccum,
                              int num_heads, int seqlen) {
#if !defined(__CUDA_ARCH__) || \
    ((__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL)) || \
     (__CUDA_ARCH__ == 1030 && defined(__CUDA_ARCH_FEAT_SM103_ALL)))
  const int q_block_id  = (int)blockIdx.x;
  const int batch_head  = (int)blockIdx.y;
  const int batch = batch_head / num_heads, head = batch_head % num_heads;
  const int token_begin = q_block_id * Q_TILE;

  if constexpr (KERNEL_PDL) griddepcontrol_wait();

  float4* dqaccum_zero_destination = reinterpret_cast<float4*>(
      dqaccum + (size_t)batch_head * seqlen * HEAD_DIM + (size_t)q_block_id * DQ::DQ_BLOCK_ELEMS);
  const float4 zero_float4 = make_float4(0.f, 0.f, 0.f, 0.f);
  #pragma unroll
  for (int i = 0; i < (DQ::DQ_BLOCK_ELEMS / 4) / 256; ++i)
    dqaccum_zero_destination[i * 256 + threadIdx.x] = zero_float4;

  if constexpr (KERNEL_PDL) griddepcontrol_launch_dependents();

  const int row_in_pass     = (int)threadIdx.x / 16;
  const int dimension_begin = ((int)threadIdx.x % 16) * 8;
  #pragma unroll
  for (int row_pass = 0; row_pass < Q_TILE / 16; ++row_pass) {
    const int token      = token_begin + row_pass * 16 + row_in_pass;
    const size_t element =
        token_offset<BHSD>(batch, head, num_heads, seqlen, token) + dimension_begin;
    const uint4 o_vector    = *reinterpret_cast<const uint4*>(o + element);
    const uint4 dout_vector = *reinterpret_cast<const uint4*>(dout + element);
    const __nv_bfloat162* o_pairs    = reinterpret_cast<const __nv_bfloat162*>(&o_vector);
    const __nv_bfloat162* dout_pairs = reinterpret_cast<const __nv_bfloat162*>(&dout_vector);
    float delta_accumulator = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 o_pair_as_float    = __bfloat1622float2(o_pairs[i]);
      const float2 dout_pair_as_float = __bfloat1622float2(dout_pairs[i]);
      delta_accumulator += o_pair_as_float.x * dout_pair_as_float.x +
                           o_pair_as_float.y * dout_pair_as_float.y;
    }
    #pragma unroll
    for (int shuffle_offset = 8; shuffle_offset > 0; shuffle_offset >>= 1)
      delta_accumulator += __shfl_down_sync(0xffffffffu, delta_accumulator, shuffle_offset);
    if (dimension_begin == 0) delta_rows[(size_t)batch_head * seqlen + token] = delta_accumulator;
  }
#endif
}

template <bool BHSD = false>
__global__ void __launch_bounds__(128, 1)
    vsa_bwd_postprocess_kernel(const float* __restrict__ dqaccum, __nv_bfloat16* __restrict__ dq,
                               int num_heads, int seqlen, float sm_scale) {
#if !defined(__CUDA_ARCH__) || \
    ((__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL)) || \
     (__CUDA_ARCH__ == 1030 && defined(__CUDA_ARCH_FEAT_SM103_ALL)))
  extern __shared__ __align__(16) float post_smem[];
  const int q_block_id  = (int)blockIdx.x;
  const int batch_head  = (int)blockIdx.y;
  const int batch = batch_head / num_heads, head = batch_head % num_heads;
  const int token_begin = q_block_id * Q_TILE;
  const int thread      = (int)threadIdx.x;
  constexpr int CHUNK_BF16 = 16 / (int)sizeof(__nv_bfloat16);

  const float4* dqaccum_block = reinterpret_cast<const float4*>(
      dqaccum + (size_t)batch_head * seqlen * HEAD_DIM + (size_t)q_block_id * DQ::DQ_BLOCK_ELEMS);
  if constexpr (KERNEL_PDL) griddepcontrol_wait();
  #pragma unroll
  for (int i = 0; i < (DQ::DQ_BLOCK_ELEMS / 4) / 128; ++i)
    cp_async_cg_16(smem_ptr_u32(post_smem + (i * 128 + thread) * 4),
                   dqaccum_block + i * 128 + thread);
  cp_async_commit_group();
  cp_async_wait_group<0>();
  __syncthreads();

  if constexpr (KERNEL_PDL) griddepcontrol_launch_dependents();

  uint32_t dq_packed[HEAD_DIM / 2];
  #pragma unroll
  for (int hd_slice = 0; hd_slice < HEAD_DIM / DQ::COLS; ++hd_slice) {
    #pragma unroll
    for (int v4 = 0; v4 < DQ::COLS / 4; ++v4) {
      const float4 value = *reinterpret_cast<const float4*>(
          post_smem + hd_slice * Q_TILE * DQ::COLS + v4 * Q_TILE * 4 + thread * 4);
      const int pair_index = (hd_slice * DQ::COLS + v4 * 4) / 2;
      dq_packed[pair_index + 0] = cvt_f32x2_to_bf16x2(value.x * sm_scale, value.y * sm_scale);
      dq_packed[pair_index + 1] = cvt_f32x2_to_bf16x2(value.z * sm_scale, value.w * sm_scale);
    }
  }
  __syncthreads();

  __nv_bfloat16* dq_tile_bf16 = reinterpret_cast<__nv_bfloat16*>(post_smem);
  const uint4* dq_chunks      = reinterpret_cast<const uint4*>(dq_packed);
  __nv_bfloat16* dq_row_bf16  = dq_tile_bf16 + thread * HEAD_DIM;
  #pragma unroll
  for (int v = 0; v < HEAD_DIM / CHUNK_BF16; ++v)
    *reinterpret_cast<uint4*>(dq_row_bf16 + (v / 8) * 64 + ((v % 8) ^ (thread & 7)) * CHUNK_BF16) =
        dq_chunks[v];
  __syncthreads();

  const int row_in_pass     = thread / 16;
  const int dimension_begin = (thread % 16) * 8;
  #pragma unroll
  for (int row_pass = 0; row_pass < Q_TILE / 8; ++row_pass) {
    const int row           = row_pass * 8 + row_in_pass;
    const int half          = dimension_begin / 64;
    const int chunk         = ((dimension_begin % 64) / CHUNK_BF16) ^ (row & 7);
    const uint4 value       = *reinterpret_cast<const uint4*>(
        dq_tile_bf16 + row * HEAD_DIM + half * 64 + chunk * CHUNK_BF16);
    const size_t element =
        token_offset<BHSD>(batch, head, num_heads, seqlen, token_begin + row) + dimension_begin;
    *reinterpret_cast<uint4*>(dq + element) = value;
  }
#endif
}

}

#endif
