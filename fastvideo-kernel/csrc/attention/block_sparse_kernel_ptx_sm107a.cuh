#ifndef BLOCK_SPARSE_VSA_KERNEL_PTX_SM107A_CUH
#define BLOCK_SPARSE_VSA_KERNEL_PTX_SM107A_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include "primitives.cuh"

namespace vsa_ptx_sm107a {

constexpr int BLOCK  = 128;
constexpr int M_TILE = BLOCK;
constexpr int M_TILES_PER_CTA = 2;
constexpr int K_TILE = BLOCK;
constexpr int BLOCKS_PER_KTILE = K_TILE / BLOCK;
constexpr int HEAD_DIM = 128;
constexpr int SUB_COLS_BF16 = 64;
constexpr int SUB_COLS_BYTES = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
constexpr int Q_SUBTILES = HEAD_DIM / SUB_COLS_BF16;
constexpr int K_SUBTILES = HEAD_DIM / SUB_COLS_BF16;
constexpr int V_SUBTILES = K_TILE / SUB_COLS_BF16;
constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;
constexpr int K_SUB_COLS_BYTES = K_TILE * SUB_COLS_BYTES;
constexpr int Q_TILE_BYTES = Q_SUBTILES * Q_SUB_COLS_BYTES;
constexpr int KV_RING_SLOT_BYTES = 32 * 1024;
constexpr int NUM_KV_STAGES = 6;
constexpr int S_COLS = 128;
constexpr int K_ATOMS_PER_TILE = SUB_COLS_BF16 / 16;
constexpr int SPLIT_P_N    = S_COLS / 4 * 3;
constexpr int SPLIT_P_ATOM = SPLIT_P_N / 16;
constexpr int SPLIT_P_COL  = SPLIT_P_N / 2;
constexpr int O_COLS = HEAD_DIM;
constexpr int TMEM_TOTAL = 512;

enum class Exp2Mode { kF32, kF16x2, kBF16x2 };
constexpr float EXP2_F16_SHIFT = 6.0f;

constexpr int STATS = M_TILE;
constexpr int STAT_REGIONS = 3;

constexpr int W_CORR0 = 8, W_MMA = 12, W_EPI = 13, W_LOAD = 14, W_SCHED = 15;
constexpr int N_WARPS = 16;
constexpr int CLC_STAGES = 2;

extern __shared__ __align__(1024) uint8_t fmha_smem[];

template <int B, int E>
__device__ __forceinline__ void st_p_cols(uint32_t taddr, uint32_t* p) {
  if constexpr (E - B >= 32) {
    tcgen05_st_32x32b_x32(taddr + B, *reinterpret_cast<uint32_t(*)[32]>(&p[B]));
    st_p_cols<B + 32, E>(taddr, p);
  } else if constexpr (E - B >= 16) {
    tcgen05_st_32x32b_x16(taddr + B, *reinterpret_cast<uint32_t(*)[16]>(&p[B]));
    st_p_cols<B + 16, E>(taddr, p);
  } else if constexpr (E - B >= 8) {
    tcgen05_st_32x32b_x8(taddr + B, *reinterpret_cast<uint32_t(*)[8]>(&p[B]));
    st_p_cols<B + 8, E>(taddr, p);
  } else {
    static_assert(E == B, "P split must be a multiple of 8 u32 columns");
  }
}

struct WorkItem {
  int sample;
  int head;
  int mtile0;
  int mtile1;
  int global_mtile0;
  int global_mtile1;
  int num_kv_blocks;
  int num_kv_blocks_mt[2];
};
template <bool M_SPLIT>
__device__ __forceinline__ int num_k_tiles_of(const WorkItem& it) {
  return M_SPLIT ? (it.num_kv_blocks + 1) / 2 : it.num_kv_blocks;
}
template <bool M_SPLIT>
__device__ __forceinline__ void set_counts(WorkItem& it, int c0, int c1) {
  it.num_kv_blocks_mt[0] = c0;
  it.num_kv_blocks_mt[1] = M_SPLIT ? c0 : c1;
  it.num_kv_blocks = max(max(it.num_kv_blocks_mt[0], it.num_kv_blocks_mt[1]), 1);
}

union SmemDescPair {
  uint64_t u64;
  uint2 w;
};
__device__ __forceinline__ void desc_add_lo(SmemDescPair& d, uint32_t inc) {
  asm volatile(
      "{\n\t"
      ".reg .b32 lo, hi;\n\t"
      "mov.b64 {lo, hi}, %0;\n\t"
      "add.u32 lo, lo, %1;\n\t"
      "mov.b64 %0, {lo, hi};\n\t"
      "}"
      : "+l"(d.u64)
      : "r"(inc));
}

__device__ __forceinline__ void share_reorder(int lane, int b0, int b1, int n0, int n1, int& S, int& r0, int& r1) {
  n0 = max(0, min(32, n0));
  n1 = max(0, min(32, n1));
  const unsigned valid0 = n0 >= 32 ? 0xffffffffu : ((1u << n0) - 1u);
  const unsigned valid1 = n1 >= 32 ? 0xffffffffu : ((1u << n1) - 1u);
  int pos = 0;
  #pragma unroll
  for (int st = 16; st >= 1; st >>= 1) {
    const int v = __shfl_sync(0xffffffffu, b1, (pos + st - 1) & 31);
    if (pos + st <= n1 && v < b0) pos += st;
  }
  const int at = __shfl_sync(0xffffffffu, b1, pos & 31);
  const bool hit = lane < n0 && pos < n1 && at == b0;
  const unsigned m0 = __ballot_sync(0xffffffffu, hit);
  const unsigned m1 = __reduce_or_sync(0xffffffffu, hit ? (1u << pos) : 0u);
  S = __popc(m0);
  const int src0 = (int)(lane < S ? __fns(m0, 0, lane + 1) : __fns(~m0 & valid0, 0, lane - S + 1)) & 31;
  const int src1 = (int)__fns(~m1 & valid1, 0, max(lane - S + 1, 1)) & 31;
  const int g0 = __shfl_sync(0xffffffffu, b0, src0);
  const int g1 = __shfl_sync(0xffffffffu, b1, src1);
  const int last0 = __shfl_sync(0xffffffffu, b0, max(n0 - 1, 0));
  const int last1 = __shfl_sync(0xffffffffu, b1, max(n1 - 1, 0));
  r0 = lane < n0 ? g0 : (n0 > 0 ? last0 : 0);
  r1 = lane < S ? r0 : (lane < n1 ? g1 : (n1 > 0 ? last1 : 0));
}

template <bool Q_RASTER, bool M_SPLIT>
__device__ __forceinline__ WorkItem decode_workitem_ids(
    int workitem_id, int num_heads, int num_blocks, int packed_mtiles_per_seq,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2) {
  WorkItem it;
  const int per_sample = num_heads * packed_mtiles_per_seq;
  it.sample = (int)fdiv((unsigned)workitem_id, magic0);
  const int rem = workitem_id - it.sample * per_sample;
  int p;
  if constexpr (Q_RASTER) {
    it.head = (int)fdiv((unsigned)rem, magic2);
    p       = rem - it.head * packed_mtiles_per_seq;
  } else {
    p       = (int)fdiv((unsigned)rem, magic1);
    it.head = rem - p * num_heads;
  }
  it.mtile0 = M_SPLIT ? p : 2 * p;
  it.mtile1 = M_SPLIT ? p : 2 * p + 1;
  it.global_mtile0  = (it.sample * num_heads + it.head) * num_blocks + it.mtile0;
  it.global_mtile1  = it.global_mtile0 + (M_SPLIT ? 0 : 1);
  it.num_kv_blocks = 0;
  it.num_kv_blocks_mt[0] = it.num_kv_blocks_mt[1] = 0;
  return it;
}

template <bool Q_RASTER, bool M_SPLIT>
__device__ __forceinline__ WorkItem decode_workitem(
    int workitem_id, int num_heads, int num_blocks, int packed_mtiles_per_seq,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2, const int* q2k_num) {
  WorkItem it = decode_workitem_ids<Q_RASTER, M_SPLIT>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq,
                                                          magic0, magic1, magic2);
  set_counts<M_SPLIT>(it, q2k_num[it.global_mtile0], M_SPLIT ? 0 : q2k_num[it.global_mtile1]);
  return it;
}

template <int S_LD_COLS = 32, bool FULL_NAMED_BAR = false, bool SPLIT_P = false,
          bool SOFTMAX_THROTTLE = false, bool USE_CLC = true, bool Q_RASTER = true, bool MHA = true,
          int RESCALE_THRESHOLD = 8, Exp2Mode EXP2 = Exp2Mode::kF32, bool M_SPLIT = false,
          bool Q_PRESCALED = false, bool SHARE_KV = false, bool DEFER_ROWSUM = false>
__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
fmha_context_bf16_gen_kernel(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v,
    const __grid_constant__ CUtensorMap tmap_o, int seqlen,
    int num_heads, float scale_log2, int num_samples, int num_blocks,
    int packed_mtiles_per_seq, int max_kv,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    const int* __restrict__ q2k_idx, const int* __restrict__ q2k_num,
    const int* __restrict__ variable_block_sizes, float* __restrict__ lse_out) {
  static_assert(!SHARE_KV || !M_SPLIT, "SHARE_KV: dual-Q kernel");
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 1070 && defined(__CUDA_ARCH_FEAT_SM107_ALL))

  const int total_workitems = num_samples * num_heads * packed_mtiles_per_seq;
  if constexpr (Q_PRESCALED) scale_log2 = 1.0f;

  uint8_t* sQ0 = fmha_smem;
  uint8_t* sQ1 = sQ0 + Q_TILE_BYTES;
  uint8_t* sQ[2] = { sQ0, sQ1 };
  uint8_t* sKV = sQ1 + Q_TILE_BYTES;
  __nv_bfloat16* sO0 = reinterpret_cast<__nv_bfloat16*>(sKV + NUM_KV_STAGES * KV_RING_SLOT_BYTES);
  __nv_bfloat16* sO1 = sO0 + M_TILE * HEAD_DIM;
  __nv_bfloat16* sO_bufs[2] = { sO0, sO1 };
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sO1) + M_TILE * HEAD_DIM * sizeof(__nv_bfloat16));
  uint64_t* empty_bar = full_bar + NUM_KV_STAGES;
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
  [[maybe_unused]] volatile int* share_S = reinterpret_cast<volatile int*>(
      (reinterpret_cast<uintptr_t>(alpha_and_l_smem + 2 * STAT_REGIONS * STATS) + 15u) & ~uintptr_t(15u));

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

  if (warp_id == 0) {
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), TMEM_TOTAL);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  const uint32_t tmem_base = *tmem_slot;

  if (tid == 0) {
    prefetch_tensormap(&tmap_q); prefetch_tensormap(&tmap_k); prefetch_tensormap(&tmap_v); prefetch_tensormap(&tmap_o);
    tmem_slot[1] = (uint32_t)(seqlen ^ num_heads ^ num_blocks ^ packed_mtiles_per_seq ^ max_kv)
                 ^ (uint32_t)(magic0 ^ magic1 ^ magic2) ^ (uint32_t)(uintptr_t)q2k_idx ^ (uint32_t)(uintptr_t)q2k_num
                 ^ (uint32_t)(uintptr_t)variable_block_sizes ^ (uint32_t)(uintptr_t)lse_out ^ gridDim.x;
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
    EmptyPhaseTracker<M_SPLIT ? 2 : 1> q_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    [[maybe_unused]] unsigned sh_item = 0;
    int workitem_id = (int)blockIdx.x;
    WorkItem nx = decode_workitem_ids<Q_RASTER, M_SPLIT>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq,
                                                            magic0, magic1, magic2);
    const int win_lane = min(lane, max_kv - 1);
    int nx_len[2], nx_win[2];
    auto prefetch_unit = [&]() {
      nx_len[0] = q2k_num[nx.global_mtile0];
      nx_len[1] = M_SPLIT ? 0 : q2k_num[nx.global_mtile1];
      nx_win[0] = q2k_idx[nx.global_mtile0 * max_kv + win_lane];
      nx_win[1] = M_SPLIT ? nx_win[0] : q2k_idx[nx.global_mtile1 * max_kv + win_lane];
    };
    griddepcontrol_wait();
    prefetch_unit();
    while (true) {
      WorkItem it = nx;
      set_counts<M_SPLIT>(it, nx_len[0], nx_len[1]);
      const int bh = it.sample * num_heads + it.head;
      const int global_mtile[2]  = { it.global_mtile0, it.global_mtile1 };
      const int mtile[2] = { it.mtile0, it.mtile1 };
      const int num_k_tiles = num_k_tiles_of<M_SPLIT>(it);

      int window_start[2] = { 0, 0 };
      int kv_block_id_cache[2];
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int c = it.num_kv_blocks_mt[i];
        const int v = __shfl_sync(0xffffffffu, nx_win[i], min(lane, max(c - 1, 0)));
        kv_block_id_cache[i] = c > 0 ? v : 0;
      }
      [[maybe_unused]] int sh_r0 = 0, sh_r1 = 0, sh_S = 0, sh_n0 = 0, sh_n1 = 0;
      [[maybe_unused]] int sh_j[2] = { -1, -1 }, sh_id[2] = { 0, 0 };
      [[maybe_unused]] bool sh_f = false, sh_fprev = false;
      [[maybe_unused]] auto share_window = [&](int w, int b0, int b1) {
        share_reorder(lane, b0, b1, it.num_kv_blocks_mt[0] - 32 * w, it.num_kv_blocks_mt[1] - 32 * w,
                      sh_S, sh_r0, sh_r1);
        share_S[(sh_item & 3) * 32 + (w & 31)] = sh_S;
      };
      [[maybe_unused]] auto is_shared = [&](int jj) -> bool { return jj == sh_j[0] ? sh_f : sh_fprev; };
      auto get_kv_block_id = [&](int mtile_idx, int jj) -> int {
        if constexpr (SHARE_KV) {
          if (jj == sh_j[mtile_idx]) return sh_id[mtile_idx];
          const int id = __shfl_sync(0xffffffffu, mtile_idx == 0 ? sh_r0 : sh_r1, jj & 31);
          if (mtile_idx == 0) { sh_fprev = sh_f; sh_f = (jj & 31) < sh_S; }
          sh_j[mtile_idx] = jj;
          sh_id[mtile_idx] = id;
          return id;
        }
        const int j = M_SPLIT ? 2 * jj + mtile_idx : jj;
        if (j >= window_start[mtile_idx] + 32) {
          window_start[mtile_idx] = j & ~31;
          const int c = it.num_kv_blocks_mt[mtile_idx];
          const int idx = max(0, min(window_start[mtile_idx] + lane, c - 1));
          kv_block_id_cache[mtile_idx] = c > 0 ? q2k_idx[global_mtile[mtile_idx] * max_kv + idx] : 0;
        }
        return __shfl_sync(0xffffffffu, kv_block_id_cache[mtile_idx], j & 31);
      };
      auto load_k_oneslot = [&](int mtile_idx, int ktile_idx) {
        const int kv_stage = kv_empty_ph.get_stage();

        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();

        int kv_tok[BLOCKS_PER_KTILE];
        #pragma unroll
        for (int blk = 0; blk < BLOCKS_PER_KTILE; ++blk)
          kv_tok[blk] = get_kv_block_id(mtile_idx, ktile_idx * BLOCKS_PER_KTILE + blk) * BLOCK;
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), KV_RING_SLOT_BYTES);
          tma_load_4d(smem_ptr_u32((sKV + kv_stage * KV_RING_SLOT_BYTES)), &tmap_k, smem_ptr_u32(&full_bar[kv_stage]),
                      0, kv_tok[0], 0, bh);
        }
      };
      auto load_v_oneslot = [&](int mtile_idx, int ktile_idx) {
        const int kv_stage = kv_empty_ph.get_stage();

        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();

        const int kv_tok0 = get_kv_block_id(mtile_idx, ktile_idx) * BLOCK;
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), KV_RING_SLOT_BYTES);
          tma_load_4d(smem_ptr_u32((sKV + kv_stage * KV_RING_SLOT_BYTES)), &tmap_v, smem_ptr_u32(&full_bar[kv_stage]),
                      0, kv_tok0, 0, bh);
        }
      };
      auto load_k = [&](int mtile_idx, int ktile_idx) {
        load_k_oneslot(mtile_idx, ktile_idx);
      };
      auto load_v = [&](int mtile_idx, int ktile_idx) {
        load_v_oneslot(mtile_idx, ktile_idx);
      };

      if constexpr (M_SPLIT) {
        const int qb = q_empty_ph.get_stage();
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[qb]), q_empty_ph.get_phase());
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[qb]), Q_TILE_BYTES);
          tma_load_4d(smem_ptr_u32(sQ0) + (uint32_t)(qb * Q_TILE_BYTES), &tmap_q, smem_ptr_u32(&full_bar_q[qb]),
                      0, bh, mtile[0] * BLOCK, 0);
        }
        q_empty_ph.advance();
        load_k(0, 0);
        load_k(1, 0);
      } else {
        if constexpr (SHARE_KV) share_window(0, kv_block_id_cache[0], kv_block_id_cache[1]);
        load_k(0, 0);
        if (!SHARE_KV || !is_shared(0)) load_k(1, 0);

        #pragma unroll
        for (int m = 0; m < M_TILES_PER_CTA; ++m) {
          mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), Q_TILE_BYTES);
            tma_load_4d(smem_ptr_u32(sQ[m]), &tmap_q, smem_ptr_u32(&full_bar_q[m]),
                        0, bh, mtile[m] * BLOCK, 0);
          }
        }
        q_empty_ph.advance();
      }
      [[maybe_unused]] bool more = true;
      if constexpr (!USE_CLC) {
        workitem_id += gridDim.x;
        more = workitem_id < total_workitems;
      }
      if (!USE_CLC && more) {
        nx = decode_workitem_ids<Q_RASTER, M_SPLIT>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq,
                                                       magic0, magic1, magic2);
        prefetch_unit();
      }
      for (int ktile_idx = 0; ktile_idx + 1 < num_k_tiles; ++ktile_idx) {
        load_v(0, ktile_idx);
        load_k(0, ktile_idx + 1);
        if (!SHARE_KV || !is_shared(ktile_idx)) load_v(1, ktile_idx);
        if (!SHARE_KV || !is_shared(ktile_idx + 1)) load_k(1, ktile_idx + 1);
        if constexpr (SHARE_KV) {
          if (((ktile_idx + 18) & 31) == 0 && ktile_idx + 18 < num_k_tiles) {
            const int idx = min(ktile_idx + 18 + lane, max_kv - 1);
            sh_n0 = q2k_idx[global_mtile[0] * max_kv + idx];
            sh_n1 = q2k_idx[global_mtile[1] * max_kv + idx];
          }
          if (((ktile_idx + 2) & 31) == 0 && ktile_idx + 2 < num_k_tiles) share_window((ktile_idx + 2) >> 5, sh_n0, sh_n1);
        }
      }
      load_v(0, num_k_tiles - 1);
      if (!SHARE_KV || !is_shared(num_k_tiles - 1)) load_v(1, num_k_tiles - 1);
      if constexpr (SHARE_KV) ++sh_item;

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
        nx = decode_workitem_ids<Q_RASTER, M_SPLIT>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq,
                                                       magic0, magic1, magic2);
        prefetch_unit();
      } else if (!more) {
        break;
      }
    }
  }
  else if (warp_id == W_MMA) {
    setmaxnreg_dec<48>();
    griddepcontrol_wait();

    const uint32_t lead = elect_one_sync() ? 1u : 0u;
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    const uint32_t idesc_qk = make_idesc_bf16_f32(M_TILE, K_TILE, false, false);
    const uint32_t idesc_pv = make_idesc_bf16_f32(M_TILE, HEAD_DIM, false, true);
    const uint64_t desc_q0  = build_smem_desc_sm107a(smem_ptr_u32(sQ0), DESC_SBO, DESC_LBO, SmemSwizzleSm107a::B128);
    const uint64_t desc_kv0 = build_smem_desc_sm107a(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO, SmemSwizzleSm107a::B128);
    constexpr uint32_t DESC_LBO_MN = (uint32_t)(K_TILE * SUB_COLS_BYTES);
    constexpr uint32_t V_LBO_DELTA = ((DESC_LBO_MN - DESC_LBO) >> 4) << 16;
    constexpr uint32_t PV_DESC_STEP = (uint32_t)((16 * SUB_COLS_BYTES) >> 4);
    constexpr uint64_t KV_DESC_DELTA      = KV_RING_SLOT_BYTES >> 4;
    constexpr uint64_t Q_SUB_DELTA        = Q_SUB_COLS_BYTES >> 4;
    constexpr uint64_t Q_MTILE_DESC_DELTA = Q_TILE_BYTES >> 4;

    PhaseTracker<NUM_KV_STAGES> kv_ph;
    PhaseTracker<M_SPLIT ? 2 : 1> q_ph;
    PhaseTracker<1> spo_ph;
    [[maybe_unused]] unsigned sh_item = 0;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem<Q_RASTER, M_SPLIT>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic1, magic2, q2k_num);
      const int num_k_tiles = num_k_tiles_of<M_SPLIT>(it);
      [[maybe_unused]] int sh_S = 0, sh_kslot = 0, sh_vslot = 0;
      [[maybe_unused]] bool sh_k = false, sh_v = false;

      auto bmm1 = [&](int i, [[maybe_unused]] int j) {
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint64_t da_base = desc_q0 + (uint64_t)(M_SPLIT ? q_ph.get_stage() : i) * Q_MTILE_DESC_DELTA;
        int slot = 0;
        #pragma unroll
        for (int s = 0; s < Q_SUBTILES; ++s) {
          if (s == 0) {
            if (SHARE_KV && i == 1 && sh_k) {
              slot = sh_kslot;
            } else {
              slot = kv_ph.get_stage();
              mbarrier_wait_parity(smem_ptr_u32(&full_bar[slot]), kv_ph.get_phase());
              kv_ph.advance();
              if constexpr (SHARE_KV)
                if (i == 0) {
                  if ((j & 31) == 0) sh_S = share_S[(sh_item & 3) * 32 + ((j >> 5) & 31)];
                  sh_k = __any_sync(0xffffffffu, (j & 31) < sh_S);
                  sh_kslot = slot;
                }
            }
          }
          const uint64_t da = da_base + (uint64_t)s * Q_SUB_DELTA;
          const uint64_t db = desc_kv0 + (uint64_t)slot * KV_DESC_DELTA
                            + (uint64_t)s * (K_SUB_COLS_BYTES >> 4);
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
            const bool enable_d = (s != 0) || (ki != 0);
            tcgen05_mma_f16_ss_lead(lead, s_tmem_addr, da + 2 * ki, db + 2 * ki, idesc_qk, enable_d);
          }
          if (s == Q_SUBTILES - 1 && !(SHARE_KV && i == 0 && sh_k))
            tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[slot]));
        }

        tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_spo[i]));
      };
      auto bmm2 = [&](int i, bool first_ktile, auto last_c) {
        constexpr bool last_ktile = decltype(last_c)::value;
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
        int slot = 0;
        SmemDescPair desc_bv;
        #pragma unroll
        for (int p = 0; p < V_SUBTILES; ++p) {
          if (p == 0) {
            if (SHARE_KV && i == 1 && sh_v) {
              slot = sh_vslot;
            } else {
              slot = kv_ph.get_stage();
              mbarrier_wait_parity(smem_ptr_u32(&full_bar[slot]), kv_ph.get_phase());
              kv_ph.advance();
              sh_vslot = slot;
            }
            desc_bv.u64 = desc_kv0;
            desc_bv.w.x += (uint32_t)(slot * (int)KV_DESC_DELTA) + V_LBO_DELTA;
          }
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
            const int a = p * K_ATOMS_PER_TILE + ki;
            if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
              mbarrier_wait_parity(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
            }
            const bool accumulate = (!first_ktile) || (a != 0);
            tcgen05_mma_f16_ts_1sm_lead(lead, o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), desc_bv.u64, idesc_pv, accumulate);
            desc_add_lo(desc_bv, PV_DESC_STEP);
          }

          if (p == V_SUBTILES - 1 && !(SHARE_KV && i == 0 && sh_v)) {
            tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[slot]));
          }
        }

        if constexpr (last_ktile) {
          tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_o_acc[i]));
        }
      };

      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        if (!M_SPLIT || i == 0) {
          mbarrier_wait_parity(smem_ptr_u32(&full_bar_q[M_SPLIT ? q_ph.get_stage() : i]), q_ph.get_phase());
        }
        bmm1(i, 0);
      }
      sh_v = sh_k;

      for (int k = 0; k + 1 < num_k_tiles; ++k) {
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
          bmm2(i, k == 0, std::false_type{});
          bmm1(i, k + 1);
        }
        sh_v = sh_k;

        spo_ph.advance();
      }

      #pragma unroll
      for (int i = 0; i < (M_SPLIT ? 1 : M_TILES_PER_CTA); ++i)
        tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar_q[M_SPLIT ? q_ph.get_stage() : i]));
      q_ph.advance();

      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
        bmm2(i, num_k_tiles == 1, std::true_type{});
      }

      spo_ph.advance();
      if constexpr (SHARE_KV) ++sh_item;

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
    griddepcontrol_wait();

    PhaseTracker<1> full_o_ph;
    [[maybe_unused]] PhaseTracker<1> full_o1_ph;
    if (elect_one_sync()) {
      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m)
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[m]));
    }
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem<Q_RASTER, M_SPLIT>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic1, magic2, q2k_num);
      const int bh = it.sample * num_heads + it.head;
      const int mtile[2] = { it.mtile0, it.mtile1 };

      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_epi[m]), (M_SPLIT && m == 1) ? full_o1_ph.get_phase() : full_o_ph.get_phase());

        if (elect_one_sync()) {
          if constexpr (M_SPLIT) {
            tma_store_4d(&tmap_o, 0, bh, mtile[0] * BLOCK, m,
                         smem_ptr_u32(sO0 + m * (M_TILE * SUB_COLS_BF16)));
          } else {
            tma_store_4d(&tmap_o, 0, bh, mtile[m] * BLOCK, 0,
                         smem_ptr_u32(reinterpret_cast<const uint8_t*>(sO_bufs[m])));
          }
          cp_async_bulk_commit_group();
        }
      }
      if constexpr (M_SPLIT) full_o1_ph.advance();

      if (elect_one_sync()) {
        if constexpr (M_SPLIT) {
          cp_async_bulk_wait_group_read<0>();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[0]));
        } else {
          cp_async_bulk_wait_group_read<1>();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[0]));
          cp_async_bulk_wait_group_read<0>();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[1]));
        }
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
    griddepcontrol_wait();

    if constexpr (USE_CLC) {
      int prod_stage = 0; uint32_t prod_phase = 1;
      int cons_stage = 0; uint32_t cons_phase = 0;
      while (true) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
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
        mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
      }
    }
  }
  else if (warp_id >= W_CORR0 && warp_id < W_MMA) {
    setmaxnreg_dec<80>();
    griddepcontrol_wait();

    const int corr_warp_id = warp_id - W_CORR0;
    const int corr_tid = corr_warp_id * 32 + lane;
    const int corr_row = corr_tid;
    [[maybe_unused]] PhaseTracker<1> alpha_ph;
    PhaseTracker<1> o_acc_ph;
    PhaseTracker<1> o_epi_empty_ph;

    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
      mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
    }

    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem<Q_RASTER, M_SPLIT>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic1, magic2, q2k_num);
      const int num_k_tiles = num_k_tiles_of<M_SPLIT>(it);

      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
      }
      if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();

      for (int k = 1; k < num_k_tiles; ++k) {
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
          else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());

          const float alpha = alpha_and_l_smem[(i * STAT_REGIONS + 0) * STATS + corr_tid];
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
          if constexpr (SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
        }
        if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();
      }

      if constexpr (M_SPLIT) {
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_acc[i]), o_acc_ph.get_phase());
          mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_l[i]), o_acc_ph.get_phase());
        }
        const float l0 = alpha_and_l_smem[(0 * STAT_REGIONS + 1) * STATS + corr_tid];
        const float m0 = alpha_and_l_smem[(0 * STAT_REGIONS + 2) * STATS + corr_tid];
        const float l1 = alpha_and_l_smem[(1 * STAT_REGIONS + 1) * STATS + corr_tid];
        const float m1 = alpha_and_l_smem[(1 * STAT_REGIONS + 2) * STATS + corr_tid];
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[0]));
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[1]));
        const float d = (m0 - m1) * scale_log2;
        const float beta_lo = ex2_approx_f32(-fabsf(d));
        const float b0 = (d >= 0.f) ? 1.f : beta_lo, b1 = (d >= 0.f) ? beta_lo : 1.f;
        const float l_tot = b0 * l0 + b1 * l1;
        const float inv = (l_tot > 0.f) ? rcp_approx_ftz_f32(l_tot) : 0.f;
        if (lse_out != nullptr) {
          const int q_row = it.mtile0 * BLOCK + corr_row;
          lse_out[((long)it.sample * num_heads + it.head) * (long)seqlen + q_row] =
              fmaxf(m0, m1) * scale_log2 + __log2f(l_tot > 0.f ? l_tot : 1.0f);
        }
        const float2 sc0 = f32x2_splat(b0 * inv), sc1 = f32x2_splat(b1 * inv);
        const uint32_t o0_addr = tmem_base + (uint32_t)(2 * S_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
        const uint32_t o1_addr = o0_addr + (uint32_t)O_COLS;
        #pragma unroll
        for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
          uint32_t o0_regs[16], o1_regs[16];
          tcgen05_ld_32x32b_x16(o0_addr + (uint32_t)c0, o0_regs);
          tcgen05_ld_32x32b_x16(o1_addr + (uint32_t)c0, o1_regs);
          if (c0 == 0) mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[0]), o_epi_empty_ph.get_phase());
          const float2* o0_f2 = reinterpret_cast<const float2*>(o0_regs);
          const float2* o1_f2 = reinterpret_cast<const float2*>(o1_regs);
          const int sidx = c0 / SUB_COLS_BF16;
          const int v_base = (c0 % SUB_COLS_BF16) / 8;
          __nv_bfloat16* so_sub = sO_bufs[0] + sidx * (M_TILE * SUB_COLS_BF16);
          #pragma unroll
          for (int vv = 0; vv < 2; ++vv) {
            const int v = v_base + vv;
            float2 r[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) r[q] = ffma2(o1_f2[vv * 4 + q], sc1, fmul2(o0_f2[vv * 4 + q], sc0));
            uint4 packed;
            packed.x = cvt_f32x2_to_bf16x2(r[0].x, r[0].y);
            packed.y = cvt_f32x2_to_bf16x2(r[1].x, r[1].y);
            packed.z = cvt_f32x2_to_bf16x2(r[2].x, r[2].y);
            packed.w = cvt_f32x2_to_bf16x2(r[3].x, r[3].y);
            *reinterpret_cast<uint4*>(&so_sub[corr_row * SUB_COLS_BF16 + (v ^ (corr_row & 7)) * 8]) = packed;
          }
          if (c0 == SUB_COLS_BF16 - 16) {
            fence_proxy_async_shared();
            mbarrier_arrive(smem_ptr_u32(&full_bar_o_epi[0]));
          }
        }
        tcgen05_fence_before_thread_sync();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[0]));
        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[1]));
        fence_proxy_async_shared();
        mbarrier_arrive(smem_ptr_u32(&full_bar_o_epi[1]));
      }
      else {
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_acc[i]), o_acc_ph.get_phase());
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_l[i]), o_acc_ph.get_phase());

        const float l = alpha_and_l_smem[(i * STAT_REGIONS + 1) * STATS + corr_tid];
        const float m = alpha_and_l_smem[(i * STAT_REGIONS + 2) * STATS + corr_tid];
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
        const float inv_l = (l > 0.f) ? rcp_approx_ftz_f32(l) : 0.f;
        if (lse_out != nullptr) {
          const int q_row = (i == 0 ? it.mtile0 : it.mtile1) * BLOCK + corr_row;
          lse_out[((long)it.sample * num_heads + it.head) * (long)seqlen + q_row] =
              m * scale_log2 + __log2f(l > 0.f ? l : 1.0f);
        }
        const float2 scale2 = f32x2_splat(inv_l);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
        #pragma unroll
        for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
          uint32_t o_regs[16];
          tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
          if (c0 == 0) mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[i]), o_epi_empty_ph.get_phase());
          const float2* o2 = reinterpret_cast<const float2*>(o_regs);
          const int s = c0 / SUB_COLS_BF16;
          const int v_base = (c0 % SUB_COLS_BF16) / 8;
          __nv_bfloat16* so_sub = sO_bufs[i] + s * (M_TILE * SUB_COLS_BF16);
          #pragma unroll
          for (int vv = 0; vv < 2; ++vv) {
            const int v = v_base + vv;
            const float2 r0 = fmul2(o2[vv * 4 + 0], scale2);
            const float2 r1 = fmul2(o2[vv * 4 + 1], scale2);
            const float2 r2 = fmul2(o2[vv * 4 + 2], scale2);
            const float2 r3 = fmul2(o2[vv * 4 + 3], scale2);
            uint4 packed;
            packed.x = cvt_f32x2_to_bf16x2(r0.x, r0.y);
            packed.y = cvt_f32x2_to_bf16x2(r1.x, r1.y);
            packed.z = cvt_f32x2_to_bf16x2(r2.x, r2.y);
            packed.w = cvt_f32x2_to_bf16x2(r3.x, r3.y);
            *reinterpret_cast<uint4*>(&so_sub[corr_row * SUB_COLS_BF16 + (v ^ (corr_row & 7)) * 8]) = packed;
          }
        }
        tcgen05_fence_before_thread_sync();

        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));

        fence_proxy_async_shared();

        mbarrier_arrive(smem_ptr_u32(&full_bar_o_epi[i]));
      }
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
    griddepcontrol_wait();

    const int warp_id_u = __shfl_sync(0xffffffffu, warp_id, 0);
    const int m_tile = warp_id_u < 4 ? 0 : 1;
    const int warp_in_group = warp_id_u & 3;
    const int sm_tid = warp_in_group * 32 + lane;
    const uint32_t alpha_slot_u32 = smem_ptr_u32(&alpha_and_l_smem[(m_tile * STAT_REGIONS + 0) * STATS + sm_tid]);
    const uint32_t s_tmem_addr = tmem_base + (uint32_t)(m_tile * S_COLS) + ((uint32_t)(warp_in_group * 32) << 16);
    PhaseTracker<1> spo_ph;
    PhaseTracker<1> scale_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;

    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem<Q_RASTER, M_SPLIT>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic1, magic2, q2k_num);
      const int num_k_tiles = num_k_tiles_of<M_SPLIT>(it);

      auto thr_window_load = [&](int w) -> int {
        const int cnt_mt = it.num_kv_blocks_mt[m_tile];
        if constexpr (SHARE_KV) {
          const int idx = min(w + lane, max_kv - 1);
          const int b0 = q2k_idx[it.global_mtile0 * max_kv + idx];
          const int b1 = q2k_idx[it.global_mtile1 * max_kv + idx];
          int sh_S_, r0, r1;
          share_reorder(lane, b0, b1, it.num_kv_blocks_mt[0] - w, it.num_kv_blocks_mt[1] - w, sh_S_, r0, r1);
          return (w + lane < cnt_mt) ? variable_block_sizes[m_tile == 0 ? r0 : r1] : 0;
        } else {
          const int j = M_SPLIT ? 2 * (w + lane) + m_tile : w + lane;
          const int id = q2k_idx[(it.global_mtile0 + (M_SPLIT ? 0 : m_tile)) * max_kv + min(j, max_kv - 1)];
          return (j < cnt_mt) ? variable_block_sizes[id] : 0;
        }
      };
      int thr_cache = thr_window_load(0);
      auto get_thr = [&](int k) -> int {
        if ((k & 31) == 0 && k > 0) thr_cache = thr_window_load(k);
        return __shfl_sync(0xffffffffu, thr_cache, k & 31);
      };

      float m_run = Q_PRESCALED ? 0.f : -INFINITY, l_run = 0.f;
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
      scale_empty_ph.advance();

      auto softmax_step = [&](auto is_first_c, auto masked_c, [[maybe_unused]] int thr) {
        constexpr bool IS_FIRST = decltype(is_first_c)::value;
        constexpr bool MASKED = decltype(masked_c)::value;
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_spo[m_tile]), spo_ph.get_phase());

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1030
        constexpr bool USE_LDRED = true;
#else
        constexpr bool USE_LDRED = false;
#endif
        uint32_t s_regs[S_COLS];
        [[maybe_unused]] float s_part[S_COLS / S_LD_COLS];
        #pragma unroll
        for (int c0 = 0; c0 < S_COLS; c0 += S_LD_COLS) {
          const uint32_t taddr = s_tmem_addr + (uint32_t)c0;
          if      constexpr (USE_LDRED)        tcgen05_ld_red_32x32b_max_f32<S_LD_COLS>(taddr, s_part[c0 / S_LD_COLS], *reinterpret_cast<uint32_t(*)[S_LD_COLS]>(&s_regs[c0]));
          else if constexpr (S_LD_COLS == 32)  tcgen05_ld_32x32b_x32 (taddr, *reinterpret_cast<uint32_t(*)[32]>(&s_regs[c0]));
          else if constexpr (S_LD_COLS == 64)  tcgen05_ld_32x32b_x64 (taddr, *reinterpret_cast<uint32_t(*)[64]>(&s_regs[c0]));
          else if constexpr (S_LD_COLS == 128) tcgen05_ld_32x32b_x128(taddr, *reinterpret_cast<uint32_t(*)[128]>(&s_regs[c0]));
        }

        [[maybe_unused]] float* scores = reinterpret_cast<float*>(s_regs);
        float2* scores2 = reinterpret_cast<float2*>(s_regs);
        tcgen05_fence_before_thread_sync();

        if constexpr (MASKED) {
          mask_s_row_r2p<false, S_COLS>(scores, 0, 0, thr);
          if constexpr (USE_LDRED) {
            #pragma unroll
            for (int p = 0; p < S_COLS / S_LD_COLS; ++p) {
              float mx = -INFINITY;
              #pragma unroll
              for (int j = 0; j < S_LD_COLS; ++j) mx = fmaxf(mx, scores[p * S_LD_COLS + j]);
              s_part[p] = mx;
            }
          }
        }

        float new_m, alpha = 0.0f;
        if constexpr (Q_PRESCALED) {
          static_assert(USE_LDRED && EXP2 == Exp2Mode::kF32, "fixed offset: ld.red partial maxes, f32 exp2");
          constexpr float FIX_HI = 32.f, FIX_LO = -96.f;
          float tile_max = -FLT_MAX;
          #pragma unroll
          for (int p = 0; p < S_COLS / S_LD_COLS; ++p) tile_max = fmaxf(tile_max, s_part[p]);
          if constexpr (IS_FIRST) {
            new_m = (tile_max > FIX_HI || tile_max < FIX_LO) ? tile_max : 0.f;
          } else {
            alpha = 1.0f;
            if (m_run == 0.f) {
              if (tile_max > FIX_HI) { new_m = tile_max; alpha = ex2_approx_f32(-tile_max); }
              else                   { new_m = 0.f; }
            } else {
              const float cand = fmaxf(m_run, tile_max);
              const float acc_scale_ = (m_run - cand) * scale_log2;
              if (acc_scale_ >= -(float)RESCALE_THRESHOLD) { new_m = m_run; }
              else { new_m = cand; alpha = ex2_approx_f32(acc_scale_); }
            }
            sts_f32(alpha_slot_u32, alpha);
          }
        } else {
          new_m = fmaxf(m_run, -FLT_MAX);
          if constexpr (USE_LDRED) {
            #pragma unroll
            for (int p = 0; p < S_COLS / S_LD_COLS; ++p) new_m = fmaxf(new_m, s_part[p]);
          } else {
            float rmax0 = new_m, rmax1 = -INFINITY, rmax2 = -INFINITY, rmax3 = -INFINITY;
            #pragma unroll
            for (int j = 0; j < S_COLS; j += 8) {
              rmax0 = fmaxf(fmaxf(rmax0, scores[j + 0]), scores[j + 1]);
              rmax1 = fmaxf(fmaxf(rmax1, scores[j + 2]), scores[j + 3]);
              rmax2 = fmaxf(fmaxf(rmax2, scores[j + 4]), scores[j + 5]);
              rmax3 = fmaxf(fmaxf(rmax3, scores[j + 6]), scores[j + 7]);
            }
            new_m = fmaxf(fmaxf(rmax0, rmax1), fmaxf(rmax2, rmax3));
          }
          if constexpr (!IS_FIRST) {
            const float acc_scale_ = (m_run - new_m) * scale_log2;
            if (acc_scale_ >= -(float)RESCALE_THRESHOLD) { new_m = m_run; alpha = 1.0f; }
            else                     { alpha = ex2_approx_f32(acc_scale_); }
            sts_f32(alpha_slot_u32, alpha);
          }
        }
        if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
        else mbarrier_arrive(smem_ptr_u32(&full_bar_alpha[m_tile]));

        const float2 scale2 = f32x2_splat(scale_log2);
        static_assert(EXP2 == Exp2Mode::kF32 || !DEFER_ROWSUM, "packed exp2 sums the row live");
        static_assert(EXP2 != Exp2Mode::kF16x2 || RESCALE_THRESHOLD + EXP2_F16_SHIFT < 16.0f,
                      "f16 P must stay below 65504");
        const float neg_m_scaled = -new_m * scale_log2;
        const float2 neg_m_scaled2 =
            f32x2_splat(EXP2 == Exp2Mode::kF16x2 ? neg_m_scaled + EXP2_F16_SHIFT : neg_m_scaled);
        uint32_t p_regs[S_COLS / 2];
        [[maybe_unused]] float2 lt2_live = make_float2(IS_FIRST ? 0.0f : l_run * alpha, 0.0f);
        [[maybe_unused]] float2 lt2_p[4] = {lt2_live, make_float2(0.f, 0.f), make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
        if constexpr (Q_PRESCALED) {
          if (!__all_sync(0xffffffffu, new_m == 0.f)) {
            #pragma unroll
            for (int c = 0; c < S_COLS / 2; ++c) scores2[c] = fadd2(scores2[c], neg_m_scaled2);
          }
        }
        #pragma unroll
        for (int c = 0; c < S_COLS / 2; ++c) {
          const float2 a2 = Q_PRESCALED ? scores2[c] : ffma2(scores2[c], scale2, neg_m_scaled2);
          if constexpr (EXP2 != Exp2Mode::kF32) {
            if constexpr (EXP2 == Exp2Mode::kBF16x2) {
              const uint32_t e = ex2_approx_bf16x2(cvt_f32x2_to_bf16x2(a2.x, a2.y));
              uint64_t l64;
              memcpy(&l64, &lt2_p[c & 3], sizeof(l64));
              l64 = add_f32x2_h2<F32Rnd::kRn, true>(e, l64);
              memcpy(&lt2_p[c & 3], &l64, sizeof(l64));
              p_regs[c] = e;
            } else {
              const uint32_t e = ex2_approx_f16x2(cvt_f32x2_to_f16x2(a2.x, a2.y));
              const uint64_t p64 = add_f32x2_h2<F32Rnd::kRn, false>(e, 0ull);
              float2 p2;
              memcpy(&p2, &p64, sizeof(p2));
              lt2_p[c & 3] = fadd2(lt2_p[c & 3], p2);
              p_regs[c] = cvt_f32x2_to_bf16x2(p2.x, p2.y);
            }
            continue;
          }
          scores2[c] = make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y));
          if constexpr (!DEFER_ROWSUM) lt2_live = fadd2(lt2_live, scores2[c]);
          p_regs[c] = cvt_f32x2_to_bf16x2(scores2[c].x, scores2[c].y);
        }
        if constexpr (EXP2 != Exp2Mode::kF32)
          lt2_live = fadd2(fadd2(lt2_p[0], lt2_p[1]), fadd2(lt2_p[2], lt2_p[3]));
        const uint32_t p_tmem_addr = s_tmem_addr;

        if constexpr (SPLIT_P) {
          st_p_cols<0, SPLIT_P_COL>(p_tmem_addr, p_regs);
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
          st_p_cols<SPLIT_P_COL, S_COLS / 2>(p_tmem_addr, p_regs);
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&full_bar_p_last[m_tile]));
        } else {
          tcgen05_st_32x32b_x32(p_tmem_addr,      *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x32(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[32]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
        }

        spo_ph.advance();
        float2 lt2 = lt2_live;
        if constexpr (DEFER_ROWSUM) {
          float2 lt2a = make_float2(IS_FIRST ? 0.0f : l_run * alpha, 0.0f), lt2b = make_float2(0.f, 0.f);
          float2 lt2c = make_float2(0.f, 0.f), lt2d = make_float2(0.f, 0.f);
          #pragma unroll
          for (int c = 0; c < S_COLS / 2; c += 4) {
            lt2a = fadd2(lt2a, scores2[c + 0]);
            lt2b = fadd2(lt2b, scores2[c + 1]);
            lt2c = fadd2(lt2c, scores2[c + 2]);
            lt2d = fadd2(lt2d, scores2[c + 3]);
          }
          lt2 = fadd2(fadd2(lt2a, lt2b), fadd2(lt2c, lt2d));
        }
        l_run = lt2.x + lt2.y; m_run = new_m;
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
        scale_empty_ph.advance();
      };
      auto run_step = [&](auto is_first_c, int k) {
        const int thr = get_thr(k);
        if (thr < S_COLS) softmax_step(is_first_c, std::true_type{}, thr);
        else              softmax_step(is_first_c, std::false_type{}, thr);
      };
      run_step(std::true_type{}, 0);
      for (int k = 1; k < num_k_tiles; ++k) run_step(std::false_type{}, k);

      alpha_and_l_smem[(m_tile * STAT_REGIONS + 1) * STATS + sm_tid] = l_run;
      alpha_and_l_smem[(m_tile * STAT_REGIONS + 2) * STATS + sm_tid] = m_run;
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
#endif
}

}

#endif
