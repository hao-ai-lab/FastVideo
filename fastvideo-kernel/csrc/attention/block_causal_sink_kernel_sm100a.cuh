// block_causal_sink_kernel_sm100a.cuh -- block-causal + sink + sliding-window FMHA forward, sm_100a.
// Warp-specialized: load / MMA (tcgen05) / softmax / correction / epilogue / scheduler.
// Writes O and, when asked, the log-sum-exp the backward consumes.
//
// Generated (comments stripped). Do not edit by hand.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <cstring>
#include <cmath>
#include <vector>
#include "primitives.cuh"

constexpr int M_TILE = 128;
constexpr int M_TILES_PER_CTA = 2;
constexpr int K_TILE = 128;
constexpr int HEAD_DIM = 128;

constexpr int SUB_COLS_BF16 = 64;
constexpr int SUB_COLS_BYTES = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
constexpr int Q_SUBTILES = HEAD_DIM / SUB_COLS_BF16;
constexpr int K_SUBTILES = HEAD_DIM / SUB_COLS_BF16;
constexpr int V_SUBTILES = K_TILE / SUB_COLS_BF16;
constexpr int P_SUBTILES = K_TILE / SUB_COLS_BF16;
constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;
constexpr int K_SUB_COLS_BYTES = K_TILE * SUB_COLS_BYTES;
constexpr int Q_TILE_BYTES = Q_SUBTILES * Q_SUB_COLS_BYTES;
constexpr int K_TILE_BYTES = K_SUBTILES * K_SUB_COLS_BYTES;
constexpr int V_TILE_BYTES = V_SUBTILES * Q_SUB_COLS_BYTES;
constexpr int P_TILE_BYTES = P_SUBTILES * Q_SUB_COLS_BYTES;
constexpr int K_ATOMS_PER_TILE = SUB_COLS_BF16 / 16;
constexpr int SPLIT_P_N    = K_TILE / 4 * 3;
constexpr int SPLIT_P_ATOM = SPLIT_P_N / 16;
constexpr int SPLIT_P_COL  = SPLIT_P_N / 2;
constexpr int EX2_FRG_PAIRS = 16;
constexpr int EX2_FRG_CNT   = K_TILE / 32;

constexpr int EX2_RES       = 4;
constexpr int EX2_START_FRG = 1;
constexpr int NUM_KV_STAGES = 3;
constexpr int S_COLS = K_TILE;
constexpr int O_COLS = HEAD_DIM;
constexpr int TMEM_TOTAL = 512;
constexpr int W_CORR0 = 8, W_MMA = 12, W_EPI = 13, W_LOAD = 14, W_SCHED = 15;
constexpr int N_WARPS = 16;
constexpr int CLC_STAGES = 4;

extern __shared__ __align__(1024) uint8_t fmha_smem[];

union SmemDescPair { uint64_t u64; uint2 w; };

__device__ __forceinline__ void desc_add_lo(SmemDescPair& d, uint32_t inc) {
  asm volatile("{\n\t"
      ".reg .b32 lo, hi;\n\t"
      "mov.b64 {lo, hi}, %0;\n\t"
      "add.u32 lo, lo, %1;\n\t"
      "mov.b64 %0, {lo, hi};\n\t"
      "}" : "+l"(d.u64) : "r"(inc));
}

__device__ __forceinline__ int tile_for_processing(int step, int window_tiles, int window_hi_tile, int k_tiles) {
  return (step < window_tiles) ? (window_hi_tile - 1 - step) : (k_tiles - 1 - step);
}

template <bool Q_RASTER, bool LPT, bool HAS_SINK_ROPE_DELTA = false>
__device__ __forceinline__ void decode_workitem(
    int workitem_id, int seqlen, int num_kv_heads,
    int packed_mtiles_per_seq, int packed_mtiles_per_sample, int q_tile_per_cta,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    int lpt_swz_log2, int lpt_hb_quot, int lpt_hb_rem,
    unsigned long long lpt_major_magic, unsigned long long lpt_rem_magic,
    int tokens_per_block, int rolling_window_tokens, int sink_tokens,
    int& sample, int& h_kv, int& q_tile_base, int& k_tiles,
    int& window_tiles, int& window_hi_tile) {
  int packed_mtiles_index;
  if constexpr (LPT) {

    const int major = (int)fdiv((unsigned)workitem_id, lpt_major_magic);
    const int l2mod = workitem_id - major * (packed_mtiles_per_seq << lpt_swz_log2);
    int block, res;
    if (major < lpt_hb_quot) {
      block = l2mod >> lpt_swz_log2;
      res   = l2mod & ((1 << lpt_swz_log2) - 1);
    } else {
      block = (int)fdiv((unsigned)l2mod, lpt_rem_magic);
      res   = l2mod - block * lpt_hb_rem;
    }
    const int hb = (major << lpt_swz_log2) + res;
    sample = (int)fdiv((unsigned)hb, magic2);
    h_kv   = hb - sample * num_kv_heads;
    packed_mtiles_index = packed_mtiles_per_seq - 1 - block;
    q_tile_base = packed_mtiles_index * q_tile_per_cta;
  } else {
    sample = (int)fdiv((unsigned)workitem_id, magic0);
    const int rr = workitem_id - sample * packed_mtiles_per_sample;
    if constexpr (Q_RASTER) {
      h_kv                = (int)fdiv((unsigned)rr, magic1);
      packed_mtiles_index = rr - h_kv * packed_mtiles_per_seq;
    } else {
      packed_mtiles_index = (int)fdiv((unsigned)rr, magic2);
      h_kv                = rr - packed_mtiles_index * num_kv_heads;
    }
    q_tile_base = packed_mtiles_index * q_tile_per_cta;
  }

  const int first_q = q_tile_base;
  const int last_q  = q_tile_base + q_tile_per_cta - 1;

  int max_block_end = (last_q / tokens_per_block + 1) * tokens_per_block;
  if (max_block_end > seqlen) max_block_end = seqlen;

  int min_window_start = ((first_q / tokens_per_block + 1) * tokens_per_block) - rolling_window_tokens;
  if (min_window_start < 0) min_window_start = 0;

  const int hi_tile = (max_block_end + K_TILE - 1) / K_TILE;
  int lo_tile = min_window_start / K_TILE;
  int sink_tiles = (sink_tokens + K_TILE - 1) / K_TILE;

  if constexpr (HAS_SINK_ROPE_DELTA) {

    if (lo_tile < sink_tiles - 1) lo_tile = sink_tiles - 1;
    if (lo_tile > hi_tile) lo_tile = hi_tile;
  } else {
    if (sink_tiles > lo_tile) sink_tiles = lo_tile;
  }

  window_hi_tile = hi_tile;
  window_tiles   = hi_tile - lo_tile;
  k_tiles        = window_tiles + sink_tiles;
}

template <int K_TILE, bool USE_R2P_ASM = true>
__device__ __forceinline__ void mask_s_row_block_causal_sink(float* scores, int k_tile_offset,
    int block_end, int sink_tokens, int window_start) {
  const int hi  = block_end     - k_tile_offset;
  const int snk = sink_tokens   - k_tile_offset;
  const int lo  = window_start  - k_tile_offset;
  uint32_t* u = reinterpret_cast<uint32_t*>(scores);
  #pragma unroll
  for (int s = 0; s < K_TILE / 32; ++s) {
    const int base = s * 32;
    int hb = hi - base;  hb = hb < 0 ? 0 : (hb > 32 ? 32 : hb);
    int sb = snk - base; sb = sb < 0 ? 0 : (sb > 32 ? 32 : sb);
    int lb = lo - base;  lb = lb < 0 ? 0 : (lb > 32 ? 32 : lb);
    const uint32_t keep_hi   = (hb >= 32) ? 0xFFFFFFFFu : (hb <= 0 ? 0u : ((1u << hb) - 1u));
    const uint32_t keep_sink = (sb >= 32) ? 0xFFFFFFFFu : (sb <= 0 ? 0u : ((1u << sb) - 1u));
    const uint32_t keep_win  = (lb >= 32) ? 0u : (0xFFFFFFFFu << lb);
    const uint32_t keep = keep_hi & (keep_sink | keep_win);
    if constexpr (USE_R2P_ASM) {
      #pragma unroll
      for (int g = 0; g < 32; g += 8) {
        const uint32_t kg = keep >> g;
        asm("{\n\t"
            ".reg .pred p0, p1, p2, p3, p4, p5, p6, p7;\n\t"
            ".reg .b32  t0, t1, t2, t3, t4, t5, t6, t7;\n\t"
            "and.b32 t0, %8, 1;    setp.ne.b32 p0, t0, 0; selp.b32 %0, %0, 0xFF800000, p0;\n\t"
            "and.b32 t1, %8, 2;    setp.ne.b32 p1, t1, 0; selp.b32 %1, %1, 0xFF800000, p1;\n\t"
            "and.b32 t2, %8, 4;    setp.ne.b32 p2, t2, 0; selp.b32 %2, %2, 0xFF800000, p2;\n\t"
            "and.b32 t3, %8, 8;    setp.ne.b32 p3, t3, 0; selp.b32 %3, %3, 0xFF800000, p3;\n\t"
            "and.b32 t4, %8, 16;   setp.ne.b32 p4, t4, 0; selp.b32 %4, %4, 0xFF800000, p4;\n\t"
            "and.b32 t5, %8, 32;   setp.ne.b32 p5, t5, 0; selp.b32 %5, %5, 0xFF800000, p5;\n\t"
            "and.b32 t6, %8, 64;   setp.ne.b32 p6, t6, 0; selp.b32 %6, %6, 0xFF800000, p6;\n\t"
            "and.b32 t7, %8, 128;  setp.ne.b32 p7, t7, 0; selp.b32 %7, %7, 0xFF800000, p7;\n\t"
            "}"
            : "+r"(u[s * 32 + g + 0]), "+r"(u[s * 32 + g + 1]),
              "+r"(u[s * 32 + g + 2]), "+r"(u[s * 32 + g + 3]),
              "+r"(u[s * 32 + g + 4]), "+r"(u[s * 32 + g + 5]),
              "+r"(u[s * 32 + g + 6]), "+r"(u[s * 32 + g + 7])
            : "r"(kg));
      }
    } else {
      #pragma unroll
      for (int i = 0; i < 32; ++i)
        if (!(keep & (1u << i))) scores[s * 32 + i] = -INFINITY;
    }
  }
}

template <int S_LD_COLS = 32, bool FULL_NAMED_BAR = false, bool EX2_EMU = false, bool SPLIT_P = true,
          bool SOFTMAX_THROTTLE = false, bool USE_CLC = true, bool Q_RASTER = true, bool MHA = false,
          bool LPT = false, int RESCALE_THRESHOLD = 8, bool HAS_SINK_ROPE_DELTA = false>
__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
block_causal_sink_sm100a_kernel(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v_t,
    const __grid_constant__ CUtensorMap tmap_o, const __grid_constant__ CUtensorMap tmap_q_sink,
    float* __restrict__ lse_out,
    int seqlen,
    int num_q_heads, int num_kv_heads, float scale_log2,
    int packed_mtiles_per_seq, int num_samples,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    int lpt_swz_log2, int lpt_hb_quot, int lpt_hb_rem,
    unsigned long long lpt_major_magic, unsigned long long lpt_rem_magic,
    int tokens_per_block, int sink_tokens, int rolling_window_tokens) {
  const int gqa_group_size = MHA ? 1 : (num_q_heads / num_kv_heads);
  const int q_tile_per_mtile = M_TILE / gqa_group_size;
  const int q_tile_per_cta   = M_TILES_PER_CTA * q_tile_per_mtile;
  const int total_workitems = num_samples * packed_mtiles_per_seq * num_kv_heads;

  uint8_t* sQ0 = fmha_smem;
  uint8_t* sQ1 = sQ0 + Q_TILE_BYTES;
  uint8_t* sQ[2] = { sQ0, sQ1 };
  uint8_t* sKV = sQ1 + Q_TILE_BYTES;
  __nv_bfloat16* sO0 = reinterpret_cast<__nv_bfloat16*>(sKV + NUM_KV_STAGES * K_TILE_BYTES);
  __nv_bfloat16* sO1 = sO0 + M_TILE * HEAD_DIM;
  __nv_bfloat16* sO_bufs[2] = { sO0, sO1 };
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sO1) + M_TILE * HEAD_DIM * sizeof(__nv_bfloat16));
  uint64_t* empty_bar= full_bar + NUM_KV_STAGES;
  uint64_t* full_bar_q  = empty_bar + NUM_KV_STAGES;
  uint64_t* empty_bar_q   = full_bar_q + 2;
  uint64_t* full_bar_spo  = empty_bar_q + 2;
  uint64_t* empty_bar_spo = full_bar_spo + 2;
  uint64_t* full_bar_o_acc   = empty_bar_spo + 2;
  uint64_t* full_bar_alpha = full_bar_o_acc + 2;
  uint64_t* full_bar_l   = full_bar_alpha + 2;
  uint64_t* full_bar_p_last    = full_bar_l + 2;
  uint64_t* empty_bar_alpha_and_l = full_bar_p_last + 2;
  uint64_t* full_bar_o_epi  = empty_bar_alpha_and_l + 2;
  uint64_t* empty_bar_o_epi = full_bar_o_epi + 2;
  uint64_t* clc_full  = empty_bar_o_epi + 2;
  uint64_t* clc_empty = clc_full + CLC_STAGES;
  uint32_t* clc_response = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(clc_empty + CLC_STAGES) + 15u) & ~uintptr_t(15u));

  uint32_t* tmem_slot = clc_response + CLC_STAGES * 4;
  float* alpha_and_l_smem = reinterpret_cast<float*>(tmem_slot + 2);

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

  if (warp_id == 0) {
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), TMEM_TOTAL);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  if (*tmem_slot != 0u) __trap();
  const uint32_t tmem_base = 0u;

  const int packed_mtiles_per_sample = packed_mtiles_per_seq * num_kv_heads;
  if (tid == 0) {
    #pragma unroll
    for (int s = 0; s < NUM_KV_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar[s]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar[s]), 1);
    }
    for (int i = 0; i < 2; ++i) {
      mbarrier_init(smem_ptr_u32(&full_bar_q[i]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_q[i]), 1);
      mbarrier_init(smem_ptr_u32(&full_bar_l[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_spo[i]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_spo[i]), 256);
      mbarrier_init(smem_ptr_u32(&full_bar_o_acc[i]), 1);
      mbarrier_init(smem_ptr_u32(&full_bar_alpha[i]), 128);
      mbarrier_init(smem_ptr_u32(&empty_bar_alpha_and_l[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_p_last[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_o_epi[i]), 128);
      mbarrier_init(smem_ptr_u32(&empty_bar_o_epi[i]), 1);
    }
    if constexpr (USE_CLC) {
      #pragma unroll
      for (int s = 0; s < CLC_STAGES; ++s) {
        mbarrier_init(smem_ptr_u32(&clc_full[s]), 1);
        mbarrier_init(smem_ptr_u32(&clc_empty[s]), N_WARPS);
      }
      #pragma unroll
      for (int i = 0; i < CLC_STAGES * 4; ++i)
        clc_response[i] = 0;
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();

  if (warp_id == W_LOAD) {
    setmaxnreg_dec<48>();

    EmptyPhaseTracker<NUM_KV_STAGES> kv_empty_ph;
    EmptyPhaseTracker<1> q_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      int sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile;
      decode_workitem<Q_RASTER, LPT, HAS_SINK_ROPE_DELTA>(workitem_id, seqlen, num_kv_heads,
          packed_mtiles_per_seq, packed_mtiles_per_sample, q_tile_per_cta,
          magic0, magic1, magic2,
          lpt_swz_log2, lpt_hb_quot, lpt_hb_rem, lpt_major_magic, lpt_rem_magic,
          tokens_per_block, rolling_window_tokens, sink_tokens,
          sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile);
      const int k_start = sample * seqlen;

      for (int k = 0; k < K_TILES; ++k) {
        const int k_tile_offset = tile_for_processing(k, window_tiles, window_hi_tile, K_TILES) * K_TILE;

        if constexpr (HAS_SINK_ROPE_DELTA) {
          if (k == window_tiles) {
            #pragma unroll
            for (int m = 0; m < M_TILES_PER_CTA; ++m) {
              mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
              const uint32_t qbar = smem_ptr_u32(&full_bar_q[m]);
              const int q_token = q_tile_base + m * q_tile_per_mtile;
              const int q_head  = h_kv * gqa_group_size;
              if (elect_one_sync()) {
                mbarrier_arrive_expect_tx(qbar, Q_TILE_BYTES);
                #pragma unroll
                for (int s = 0; s < Q_SUBTILES; ++s)
                  tma_load_4d(smem_ptr_u32(sQ[m] + s * Q_SUB_COLS_BYTES), &tmap_q_sink, qbar,
                              s * SUB_COLS_BF16, q_head, q_token, sample);
              }
            }
            q_empty_ph.advance();
          }
        }
        int kv_stage = kv_empty_ph.get_stage();

        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();

        const uint32_t kbar = smem_ptr_u32(&full_bar[kv_stage]);
        const uint32_t kdst = smem_ptr_u32(sKV + kv_stage * K_TILE_BYTES);
        const int      k_token = k_tile_offset;
        const int      k_head_atom  = sample * num_kv_heads + h_kv;
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(kbar, K_TILE_BYTES);
          tma_load_4d(kdst, &tmap_k, kbar, 0, k_token, 0, k_head_atom);
        }

        if (k == 0) {

          #pragma unroll
          for (int m = 0; m < M_TILES_PER_CTA; ++m) {
            mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());

            const uint32_t qbar = smem_ptr_u32(&full_bar_q[m]);
            const int q_token = q_tile_base + m * q_tile_per_mtile;
            const int q_head  = h_kv * gqa_group_size;

            if (elect_one_sync()) {
              mbarrier_arrive_expect_tx(qbar, Q_TILE_BYTES);
              #pragma unroll
              for (int s = 0; s < Q_SUBTILES; ++s) {
                tma_load_4d(smem_ptr_u32(sQ[m] + s * Q_SUB_COLS_BYTES), &tmap_q, qbar,
                            s * SUB_COLS_BF16, q_head, q_token, sample);
              }
            }
          }
          q_empty_ph.advance();
          }
        kv_stage = kv_empty_ph.get_stage();

        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();

        const uint32_t vbar = smem_ptr_u32(&full_bar[kv_stage]);
        const uint32_t vdst = smem_ptr_u32(sKV + kv_stage * K_TILE_BYTES);
        const int      v_token = k_tile_offset;
        const int      v_head_col = sample * num_kv_heads + h_kv;
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(vbar, V_TILE_BYTES);

          tma_load_4d(vdst, &tmap_v_t, vbar, 0, v_token, 0, v_head_col);
        }
      }
      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_MMA) {
    setmaxnreg_dec<48>();

    const uint32_t lead = elect_one_sync() ? 1u : 0u;
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;

    const uint32_t idesc_qk = make_idesc_bf16_f32(M_TILE, K_TILE, false, false);
    const uint32_t idesc_pv = make_idesc_bf16_f32(M_TILE, HEAD_DIM,  false, true);
    const uint64_t desc_q0  = build_smem_desc_blackwell(smem_ptr_u32(sQ0), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_kv0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);

    constexpr uint32_t DESC_LBO_MN = (uint32_t)(K_TILE * SUB_COLS_BF16 * 2);
    const uint64_t desc_v0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO_MN, SmemSwizzleBlackwell::B128);
    constexpr int      PV_K_STEPS   = HEAD_DIM / 16;
    constexpr uint32_t PV_DESC_STEP = (uint32_t)((16 * SUB_COLS_BF16 * 2) >> 4);

    constexpr uint64_t KV_DESC_DELTA      = K_TILE_BYTES >> 4;
    constexpr uint64_t SUB_DESC_DELTA     = Q_SUB_COLS_BYTES >> 4;
    constexpr uint64_t Q_MTILE_DESC_DELTA = Q_TILE_BYTES >> 4;

    PhaseTracker<NUM_KV_STAGES> kv_ph;
    PhaseTracker<1> q_ph;
    PhaseTracker<1> spo_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      int sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile;
      decode_workitem<Q_RASTER, LPT, HAS_SINK_ROPE_DELTA>(workitem_id, seqlen, num_kv_heads,
          packed_mtiles_per_seq, packed_mtiles_per_sample, q_tile_per_cta,
          magic0, magic1, magic2,
          lpt_swz_log2, lpt_hb_quot, lpt_hb_rem, lpt_major_magic, lpt_rem_magic,
          tokens_per_block, rolling_window_tokens, sink_tokens,
          sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile);

      int kv_stage = kv_ph.get_stage();

      mbarrier_wait_parity(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
      kv_ph.advance();
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity(smem_ptr_u32(&full_bar_q[i]), q_ph.get_phase());

        {
          const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
          SmemDescPair desc_a, desc_b;
          desc_a.u64 = desc_q0;  desc_a.w.x += (uint32_t)(i * (int)Q_MTILE_DESC_DELTA);
          desc_b.u64 = desc_kv0; desc_b.w.x += (uint32_t)(kv_stage * (int)KV_DESC_DELTA);

          #pragma unroll
          for (int s = 0; s < Q_SUBTILES; ++s) {
            #pragma unroll
            for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
              const bool enable_d = (s != 0) || (ki != 0);
              tcgen05_mma_f16_ss_lead(lead, s_tmem_addr, desc_a.u64, desc_b.u64, idesc_qk, enable_d);
              desc_add_lo(desc_a, 2); desc_add_lo(desc_b, 2);
            }
            desc_add_lo(desc_a, (uint32_t)(SUB_DESC_DELTA - 2 * K_ATOMS_PER_TILE));
            desc_add_lo(desc_b, (uint32_t)(SUB_DESC_DELTA - 2 * K_ATOMS_PER_TILE));
          }
        }

        tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_spo[i]));
      }

      tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[kv_stage]));

      for (int k_tile_id = 0; k_tile_id < K_TILES - 1; ++k_tile_id) {

        const int kv_stage = kv_ph.get_stage();

        mbarrier_wait_parity(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
        kv_ph.advance();

        if constexpr (HAS_SINK_ROPE_DELTA) {
          if (k_tile_id == window_tiles - 1) {
            #pragma unroll
            for (int i = 0; i < M_TILES_PER_CTA; ++i)
              tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar_q[i]));
            q_ph.advance();
            #pragma unroll
            for (int i = 0; i < M_TILES_PER_CTA; ++i)
              mbarrier_wait_parity(smem_ptr_u32(&full_bar_q[i]), q_ph.get_phase());
          }
        }

        int kv_stage_next = 0;
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());

          const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
          const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
          SmemDescPair desc_bv;
          desc_bv.u64 = desc_v0;
          desc_bv.w.x += (uint32_t)(kv_stage * (int)KV_DESC_DELTA);

          {
            #pragma unroll
            for (int a = 0; a < PV_K_STEPS; ++a) {
              if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
                mbarrier_wait_parity(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
              }
              const bool accumulate = (k_tile_id != 0) || (a != 0);
              tcgen05_mma_f16_ts_1sm_lead(lead, o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), desc_bv.u64, idesc_pv, accumulate);
              desc_add_lo(desc_bv, PV_DESC_STEP);
            }
          }

          if (i == 0) {
            kv_stage_next = kv_ph.get_stage();
            mbarrier_wait_parity(smem_ptr_u32(&full_bar[kv_stage_next]), kv_ph.get_phase());
            kv_ph.advance();
          }

          if (i == M_TILES_PER_CTA - 1) {
            tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[kv_stage]));
          }

          {
            SmemDescPair desc_a, desc_b;
            desc_a.u64 = desc_q0;  desc_a.w.x += (uint32_t)(i * (int)Q_MTILE_DESC_DELTA);
            desc_b.u64 = desc_kv0; desc_b.w.x += (uint32_t)(kv_stage_next * (int)KV_DESC_DELTA);
            #pragma unroll
            for (int s = 0; s < Q_SUBTILES; ++s) {
              #pragma unroll
              for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
                const bool enable_d = (s != 0) || (ki != 0);
                tcgen05_mma_f16_ss_lead(lead, s_tmem_addr, desc_a.u64, desc_b.u64, idesc_qk, enable_d);
                desc_add_lo(desc_a, 2); desc_add_lo(desc_b, 2);
              }
              desc_add_lo(desc_a, (uint32_t)(SUB_DESC_DELTA - 2 * K_ATOMS_PER_TILE));
              desc_add_lo(desc_b, (uint32_t)(SUB_DESC_DELTA - 2 * K_ATOMS_PER_TILE));
            }
          }

          tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_spo[i]));
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[kv_stage_next]));

        spo_ph.advance();
      }

      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar_q[i]));
      }

      kv_stage = kv_ph.get_stage();

      mbarrier_wait_parity(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
      kv_ph.advance();

      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());

        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
        SmemDescPair desc_bv;
        desc_bv.u64 = desc_v0;
        desc_bv.w.x += (uint32_t)(kv_stage * (int)KV_DESC_DELTA);

        {
          #pragma unroll
          for (int a = 0; a < PV_K_STEPS; ++a) {
            if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
              mbarrier_wait_parity(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
            }
            const bool accumulate = (K_TILES != 1) || (a != 0);
            tcgen05_mma_f16_ts_1sm_lead(lead, o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), desc_bv.u64, idesc_pv, accumulate);
            desc_add_lo(desc_bv, PV_DESC_STEP);
          }
        }

        tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_o_acc[i]));
      }
      tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[kv_stage]));

      spo_ph.advance();
      q_ph.advance();

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_EPI) {
    setmaxnreg_dec<48>();

    PhaseTracker<1> full_o_ph;

    if (elect_one_sync()) {
      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m)
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[m]));
    }
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      int sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile;
      decode_workitem<Q_RASTER, LPT, HAS_SINK_ROPE_DELTA>(workitem_id, seqlen, num_kv_heads,
          packed_mtiles_per_seq, packed_mtiles_per_sample, q_tile_per_cta,
          magic0, magic1, magic2,
          lpt_swz_log2, lpt_hb_quot, lpt_hb_rem, lpt_major_magic, lpt_rem_magic,
          tokens_per_block, rolling_window_tokens, sink_tokens,
          sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile);

      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_epi[m]), full_o_ph.get_phase());

        if (elect_one_sync()) {
          const int q_token = q_tile_base + m * q_tile_per_mtile;

          #pragma unroll
          for (int s = 0; s < Q_SUBTILES; ++s) {
            tma_store_4d(&tmap_o, s * SUB_COLS_BF16, h_kv * gqa_group_size, q_token, sample,
                         smem_ptr_u32(reinterpret_cast<const uint8_t*>(sO_bufs[m]) + s * Q_SUB_COLS_BYTES));
          }
          cp_async_bulk_commit_group();
        }
      }

      if (elect_one_sync()) {
        cp_async_bulk_wait_group_read<1>();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[0]));
        cp_async_bulk_wait_group_read<0>();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[1]));
      }

      full_o_ph.advance();
      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_SCHED) {
    setmaxnreg_dec<48>();

    if constexpr (USE_CLC) {
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
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, cons_stage, cons_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(cons_stage, cons_phase);
        if (!next.valid) break;
      }

      for (int s = 0; s < CLC_STAGES; ++s) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
      }
    }
  }
  else if (warp_id >= W_CORR0 && warp_id < W_MMA) {
    setmaxnreg_dec<80>();

    const int corr_warp_id = warp_id - W_CORR0;
    [[maybe_unused]] PhaseTracker<1> alpha_ph;
    PhaseTracker<1> o_acc_ph;
    PhaseTracker<1> o_epi_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;

    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
      mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
    }

    int workitem_id = (int)blockIdx.x;
    while (true) {
      int sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile;
      decode_workitem<Q_RASTER, LPT, HAS_SINK_ROPE_DELTA>(workitem_id, seqlen, num_kv_heads,
          packed_mtiles_per_seq, packed_mtiles_per_sample, q_tile_per_cta,
          magic0, magic1, magic2,
          lpt_swz_log2, lpt_hb_quot, lpt_hb_rem, lpt_major_magic, lpt_rem_magic,
          tokens_per_block, rolling_window_tokens, sink_tokens,
          sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile);

      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());

        if constexpr (SOFTMAX_THROTTLE) {
          if (i == 0) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[0]));
        } else {
          mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
        }
      }
      if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();

      for (int k = 1; k < K_TILES; ++k) {
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
          else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());

          float alpha = alpha_and_l_smem[i * M_TILE + corr_warp_id * 32 + lane];
          if constexpr (!SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));

          bool skip = __all_sync(0xffffffffu, alpha == 1.0f);
          if (!skip) {

            const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);

            const float2 alpha2 = f32x2_splat(alpha);

            #pragma unroll
            for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
              uint32_t o_regs[16];
              tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
              float2* o2 = reinterpret_cast<float2*>(o_regs);
              #pragma unroll
              for (int e = 0; e < 8; ++e) o2[e] = fmul2(o2[e], alpha2);
              tcgen05_st_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
            }
            tcgen05_wait_st();

            tcgen05_fence_before_thread_sync();
          }

          if constexpr (SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[M_TILES_PER_CTA - 1 - i]));
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
        }
        if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();
      }

      if constexpr (SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[M_TILES_PER_CTA - 1]));

      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_acc[i]), o_acc_ph.get_phase());

        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_l[i]), o_acc_ph.get_phase());

        const int corr_tid = corr_warp_id * 32 + lane;
        float l = alpha_and_l_smem[i * M_TILE + corr_tid];
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
        float inv_l = (l > 0.f) ? rcp_approx_ftz_f32(l) : 0.f;
        const float2 inv_l2 = f32x2_splat(inv_l);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
        #pragma unroll
        for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
          uint32_t o_regs[16];
          tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
          if (c0 == 0) mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[i]), o_epi_empty_ph.get_phase());
          float2* o2 = reinterpret_cast<float2*>(o_regs);
          const int s = c0 / SUB_COLS_BF16;
          const int v_base = (c0 % SUB_COLS_BF16) / 8;
          __nv_bfloat16* so_sub = sO_bufs[i] + s * (M_TILE * SUB_COLS_BF16);
          #pragma unroll
          for (int vv = 0; vv < 2; ++vv) {
            const int v = v_base + vv;
            const float2 r0 = fmul2(o2[vv * 4 + 0], inv_l2);
            const float2 r1 = fmul2(o2[vv * 4 + 1], inv_l2);
            const float2 r2 = fmul2(o2[vv * 4 + 2], inv_l2);
            const float2 r3 = fmul2(o2[vv * 4 + 3], inv_l2);
            uint4 packed;
            packed.x = cvt_f32x2_to_bf16x2(r0.x, r0.y);
            packed.y = cvt_f32x2_to_bf16x2(r1.x, r1.y);
            packed.z = cvt_f32x2_to_bf16x2(r2.x, r2.y);
            packed.w = cvt_f32x2_to_bf16x2(r3.x, r3.y);
            *reinterpret_cast<uint4*>(&so_sub[corr_tid * SUB_COLS_BF16 + (v ^ (corr_tid & 7)) * 8]) = packed;
          }
        }

        tcgen05_fence_before_thread_sync();

        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));

        fence_proxy_async_shared();

        mbarrier_arrive(smem_ptr_u32(&full_bar_o_epi[i]));
      }
      o_acc_ph.advance();
      o_epi_empty_ph.advance();

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  else {
    setmaxnreg_inc<192>();

    const int warp_id_u = __shfl_sync(0xffffffffu, warp_id, 0);
    const int m_tile = warp_id_u < 4 ? 0 : 1;
    const int warp_in_group = warp_id_u & 3;
    const int row_in_m_tile = warp_in_group * 32 + lane;
    const uint32_t s_tmem_addr = tmem_base + (uint32_t)(m_tile * S_COLS) + ((uint32_t)(warp_in_group * 32) << 16);
    PhaseTracker<1> spo_ph;
    PhaseTracker<1> scale_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      int sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile;
      decode_workitem<Q_RASTER, LPT, HAS_SINK_ROPE_DELTA>(workitem_id, seqlen, num_kv_heads,
          packed_mtiles_per_seq, packed_mtiles_per_sample, q_tile_per_cta,
          magic0, magic1, magic2,
          lpt_swz_log2, lpt_hb_quot, lpt_hb_rem, lpt_major_magic, lpt_rem_magic,
          tokens_per_block, rolling_window_tokens, sink_tokens,
          sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile);

      const int q_pos = q_tile_base + m_tile * q_tile_per_mtile + row_in_m_tile / gqa_group_size;

      int block_end_in_tokens = (q_pos / tokens_per_block + 1) * tokens_per_block;
      int window_start_in_tokens = block_end_in_tokens - rolling_window_tokens;
      if (window_start_in_tokens < 0) window_start_in_tokens = 0;
      if (block_end_in_tokens > seqlen) block_end_in_tokens = seqlen;

      const int cta_last_q = q_tile_base + q_tile_per_cta - 1;
      int cta_min_block_end = (q_tile_base / tokens_per_block + 1) * tokens_per_block;
      if (cta_min_block_end > seqlen) cta_min_block_end = seqlen;
      int cta_max_window_start = ((cta_last_q  / tokens_per_block + 1) * tokens_per_block) - rolling_window_tokens;
      if (cta_max_window_start < 0) cta_max_window_start = 0;

      const int lo_tile = window_hi_tile - window_tiles;
      int bottom_mask_bound = cta_max_window_start;
      if constexpr (HAS_SINK_ROPE_DELTA) bottom_mask_bound = max(bottom_mask_bound, sink_tokens);
      int masked_top_tiles    = window_hi_tile - cta_min_block_end / K_TILE;
      int masked_bottom_tiles = (bottom_mask_bound + K_TILE - 1) / K_TILE - lo_tile;
      masked_top_tiles    = masked_top_tiles    < 0 ? 0 : (masked_top_tiles    > window_tiles ? window_tiles : masked_top_tiles);
      masked_bottom_tiles = masked_bottom_tiles < 0 ? 0 : (masked_bottom_tiles > window_tiles ? window_tiles : masked_bottom_tiles);
      const int interior_begin = masked_top_tiles;
      int interior_end = window_tiles - masked_bottom_tiles;
      if (interior_end < interior_begin) interior_end = interior_begin;

      float m_run = -INFINITY, l_run = 0.f;
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
      scale_empty_ph.advance();
      float* const alpha_slot = &alpha_and_l_smem[m_tile * M_TILE + row_in_m_tile];
      const uint32_t alpha_slot_u32 = smem_ptr_u32(alpha_slot);

      auto softmax_step = [&](auto masked_c, auto is_first_c, int k) {
        constexpr bool MASKED   = decltype(masked_c)::value;
        constexpr bool IS_FIRST = decltype(is_first_c)::value;

        const int k_tile_offset = tile_for_processing(k, window_tiles, window_hi_tile, K_TILES) * K_TILE;

        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_spo[m_tile]), spo_ph.get_phase());

        uint32_t s_regs[K_TILE];
        #pragma unroll
        for (int c0 = 0; c0 < K_TILE; c0 += S_LD_COLS) {
          const uint32_t taddr = s_tmem_addr + (uint32_t)c0;
          if      constexpr (S_LD_COLS == 32)  tcgen05_ld_32x32b_x32 (taddr, *reinterpret_cast<uint32_t(*)[32]>(&s_regs[c0]));
          else if constexpr (S_LD_COLS == 64)  tcgen05_ld_32x32b_x64 (taddr, *reinterpret_cast<uint32_t(*)[64]>(&s_regs[c0]));
          else if constexpr (S_LD_COLS == 128) tcgen05_ld_32x32b_x128(taddr, *reinterpret_cast<uint32_t(*)[128]>(&s_regs[c0]));
        }
        float* scores = reinterpret_cast<float*>(s_regs);
        float2* scores2 = reinterpret_cast<float2*>(s_regs);

        tcgen05_fence_before_thread_sync();

        if constexpr (MASKED) {

          const bool is_sink_tile = (k >= window_tiles);
          int effective_sink_tokens = sink_tokens;
          int effective_window_start = window_start_in_tokens;
          if constexpr (HAS_SINK_ROPE_DELTA) {

            if (is_sink_tile) {

              effective_window_start = block_end_in_tokens;
            } else {

              effective_sink_tokens = 0;
              effective_window_start = max(window_start_in_tokens, sink_tokens);
            }
          }

          mask_s_row_block_causal_sink<K_TILE>(scores, k_tile_offset, block_end_in_tokens,
                                 effective_sink_tokens, effective_window_start);
        }

        float rmax0 = m_run, rmax1 = -INFINITY, rmax2 = -INFINITY, rmax3 = -INFINITY;
        #pragma unroll
        for (int j = 0; j < K_TILE; j += 8) {
          rmax0 = fmaxf(fmaxf(rmax0, scores[j + 0]), scores[j + 1]);
          rmax1 = fmaxf(fmaxf(rmax1, scores[j + 2]), scores[j + 3]);
          rmax2 = fmaxf(fmaxf(rmax2, scores[j + 4]), scores[j + 5]);
          rmax3 = fmaxf(fmaxf(rmax3, scores[j + 6]), scores[j + 7]);
        }
        float new_m = fmaxf(fmaxf(rmax0, rmax1), fmaxf(rmax2, rmax3));

        float row_max_safe = (new_m == -INFINITY) ? 0.0f : new_m;
        float alpha = 0.0f;
        if constexpr (!IS_FIRST) {
          const float acc_scale_ = (m_run - row_max_safe) * scale_log2;
          alpha = ex2_approx_f32(acc_scale_);
          if (acc_scale_ >= -(float)RESCALE_THRESHOLD) {
            new_m = m_run;
            row_max_safe = m_run;
            alpha = 1.0f;
          }

          sts_f32(alpha_slot_u32, alpha);
        }
        if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
        else mbarrier_arrive(smem_ptr_u32(&full_bar_alpha[m_tile]));

        const float2 scale2        = f32x2_splat(scale_log2);
        const float2 neg_m_scaled2 = f32x2_splat(-row_max_safe * scale_log2);
        uint32_t p_regs[K_TILE / 2];

        #pragma unroll
        for (int c = 0; c < K_TILE / 2; ++c) {
          const float2 a2 = ffma2(scores2[c], scale2, neg_m_scaled2);
          if constexpr (EX2_EMU) {
            const int jj = c / EX2_FRG_PAIRS;
            const int kk = 2 * (c % EX2_FRG_PAIRS);
            constexpr int EX2_FREQ = 16;
            const bool use_hw = (kk % EX2_FREQ < EX2_FREQ - EX2_RES) || (jj >= EX2_FRG_CNT - 1) || (jj < EX2_START_FRG);
            scores2[c] = use_hw ? make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y)) : ex2_emu_f32x2(a2.x, a2.y);
          } else {
            scores2[c] = make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y));
          }
          p_regs[c] = cvt_f32x2_to_bf16x2(scores2[c].x, scores2[c].y);
        }
        const uint32_t p_tmem_addr = s_tmem_addr;

        if constexpr (SPLIT_P) {
          tcgen05_st_32x32b_x32(p_tmem_addr, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x16(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[32]));

          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
          tcgen05_st_32x32b_x16(p_tmem_addr + SPLIT_P_COL, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[SPLIT_P_COL]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&full_bar_p_last[m_tile]));
        } else {
          tcgen05_st_32x32b_x32(p_tmem_addr, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x32(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[32]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
        }
        spo_ph.advance();
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
        scale_empty_ph.advance();

        float2 lt2a = make_float2(IS_FIRST ? 0.0f : l_run * alpha, 0.0f);
        float2 lt2b = make_float2(0.f, 0.f), lt2c = make_float2(0.f, 0.f), lt2d = make_float2(0.f, 0.f);
        #pragma unroll
        for (int c = 0; c < K_TILE / 2; c += 4) {
          lt2a = fadd2(lt2a, scores2[c + 0]);
          lt2b = fadd2(lt2b, scores2[c + 1]);
          lt2c = fadd2(lt2c, scores2[c + 2]);
          lt2d = fadd2(lt2d, scores2[c + 3]);
        }
        const float2 lt2 = fadd2(fadd2(lt2a, lt2b), fadd2(lt2c, lt2d));
        l_run = lt2.x + lt2.y;
        m_run = new_m;
      };

      const bool first_masked = !(window_tiles > 0 && interior_begin == 0 && interior_end > 0);
      if (first_masked) softmax_step(std::true_type{},  std::true_type{}, 0);
      else              softmax_step(std::false_type{}, std::true_type{}, 0);
      int k = 1;
      if (window_tiles > 0) {
        for (; k < interior_begin; ++k) softmax_step(std::true_type{},  std::false_type{}, k);
        for (; k < interior_end;   ++k) softmax_step(std::false_type{}, std::false_type{}, k);
        for (; k < window_tiles;   ++k) softmax_step(std::true_type{},  std::false_type{}, k);
        if (window_tiles < K_TILES) { softmax_step(std::true_type{}, std::false_type{}, window_tiles); ++k; }
      }
      for (; k < K_TILES; ++k)          softmax_step(std::false_type{}, std::false_type{}, k);

      if (lse_out != nullptr) {
        const int q_head = h_kv * gqa_group_size + (gqa_group_size == 1 ? 0 : row_in_m_tile % gqa_group_size);
        if (q_pos < seqlen) {
          const float l_safe = (l_run > 0.f) ? l_run : 1.0f;
          lse_out[((long)sample * num_q_heads + q_head) * (long)seqlen + q_pos] =
              m_run * scale_log2 + __log2f(l_safe);
        }
      }
      *alpha_slot = l_run;
      if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
      else mbarrier_arrive(smem_ptr_u32(&full_bar_l[m_tile]));

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL);
}
