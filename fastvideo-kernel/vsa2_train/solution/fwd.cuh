// fwd.cuh -- the fine branch's forward (eq. 12) and the gated merge (eq. 13-14): each query block's exact attention over
// the tokens of the key blocks it keeps, written as out = O_fine + gate * O_coarse (zero on padding rows), with each
// row's log-sum-exp for the backward.
//
// Adapted from FastVideo's ptx_blackwell block-sparse attention forward (hao-ai-lab/FastVideo, csrc/attention/
// block_sparse_kernel_blackwell.cuh at 7c2b44726, Apache License 2.0), by way of an earlier VSA2 forward kernel's
// fine.cuh, in its 128-token-block, [head][token][dim] configuration: warp-specialized, tcgen05 MMAs with S and O in
// TMEM, TMA loads and stores, cluster-launch-control scheduling. A CTA owns a pair of query blocks (two 128-row tiles,
// whose softmax and MMA interleave), each with its own list of key blocks; the pair runs max(count0, count1) key tiles
// and the shorter tile masks its extra ones whole.
// Changes from FastVideo's kernel:
//   - keys are masked by each key block's row mask (rowmask: a bit per row), since the padding sits where the 3D tile
//     puts it (and at the end of the last text block);
//   - the epilogue adds gate * coarse (the gate per row and channel) and zeroes padding rows;
//   - the log-sum-exp is written in natural units from the softmax warps;
//   - a work item is a pair of positions in a per-head order sorted by count (topk.cuh), heads one after another (a
//     head's K and V, 61 MB at s20, stay in L2; four heads interleaved streamed 7 GB from DRAM, which under the power
//     cap costs clock); an odd number of tiles gets a phantom of count 0, whose loads read out of bounds (TMA fills
//     zeros) and which writes nothing.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>

#include <cfloat>
#include <type_traits>

#include "common.cuh"
#include "primitives.cuh"

namespace vsa2t {
namespace fwd {

constexpr int BLOCK = 128;
constexpr int M_TILE = BLOCK;
constexpr int M_TILES_PER_CTA = 2;
constexpr int K_TILE = 128;
constexpr int HEAD_DIM = 128;

constexpr int SUB_COLS_BF16 = 64;
constexpr int SUB_COLS_BYTES = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
constexpr int Q_SUBTILES = HEAD_DIM / SUB_COLS_BF16;
constexpr int V_SUBTILES = K_TILE / SUB_COLS_BF16;
constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;
constexpr int K_SUB_COLS_BYTES = K_TILE * SUB_COLS_BYTES;
constexpr int Q_TILE_BYTES = Q_SUBTILES * Q_SUB_COLS_BYTES;

constexpr int KV_RING_SLOT_BYTES = 32 * 1024;
constexpr int NUM_KV_STAGES = 3;

constexpr int S_COLS = 128;
constexpr int K_ATOMS_PER_TILE = SUB_COLS_BF16 / 16;
// P is written to TMEM in two parts: the PV MMAs of the first three quarters start while the last quarter is stored.
constexpr int SPLIT_P_N = S_COLS / 4 * 3;
constexpr int SPLIT_P_ATOM = SPLIT_P_N / 16;
constexpr int SPLIT_P_COL = SPLIT_P_N / 2;
// Of every EX2_FREQ exponentials, EX2_RES are computed on the FMA pipe (FA4's cubic) to relieve the MUFU.
constexpr int EX2_FRG_PAIRS = 16;
constexpr int EX2_FRG_CNT = S_COLS / 32;
constexpr int EX2_FREQ = 16;
constexpr int EX2_RES = 4;

constexpr int O_COLS = HEAD_DIM;
constexpr int TMEM_TOTAL = 512;
constexpr int STATS = M_TILE;
constexpr int STAT_REGIONS = 2;
// O is rescaled whenever a row's max grows (FA4's lazy rescale, which keeps a running max up to 2^8 below the true
// one, is off): the row max's P must be exactly 1 in bf16, or a row that attends mostly to one key carries that key's
// rounding (2^-9 of its value). With the lazy rescale, `out` needed twice the atol of the task's bf16 emulation.
constexpr int RESCALE_THRESHOLD = 0;

constexpr int W_CORR0 = 8, W_MMA = 12, W_EPI = 13, W_LOAD = 14, W_SCHED = 15;
constexpr int N_WARPS = 16;
constexpr int CLC_STAGES = 2;

constexpr size_t kSmemBytes = (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * KV_RING_SLOT_BYTES +
                              (size_t)2 * M_TILE * HEAD_DIM * sizeof(__nv_bfloat16) + (2 * NUM_KV_STAGES + 22) * 8 +
                              (size_t)CLC_STAGES * (2 * 8 + 16) + 16 + 8 +
                              (size_t)2 * STAT_REGIONS * STATS * sizeof(float) + 256;

struct Args {
  int heads, num_blocks, P, Nv, tchunks, tlen, vtpad;
  long long seqlen;
  float scale_log2;
  const int* q2k_idx;          // [heads * N][N] each query block's key blocks
  const int* q2k_num;          // [heads * N]
  const int* vorder;           // [heads][vtpad] the tiles (vsa2t.h) in processing order, by count
  const uint32_t* rowmask;     // [N][4]
  const __nv_bfloat16* gate;   // [heads][S][128]
  const float* coarse;         // [heads][N][128]
  float* lse;                  // [heads][S]
  float* part_o;               // [heads][P * tchunks][128][128] a text chunk's O / l
  float* part_lse;             // [heads][P * tchunks][128] and its lse
};

extern __shared__ __align__(1024) uint8_t fwd_smem[];

union SmemDescPair {
  uint64_t u64;
  uint2 w;
};

__device__ __forceinline__ void mask_keep_word(float* s, uint32_t keep) {
#pragma unroll
  for (int i = 0; i < 32; ++i)
    if (!(keep & (1u << i))) s[i] = -INFINITY;
}

// A work item is a pair of tiles. A tile is a video query block over its list, a chunk of a text query block's key
// range (written as a partial, merged by fwd_merge_kernel), or the phantom (no keys, nothing written).
struct WorkItem {
  int head;
  int qb[2];            // the tile's query block (N for the phantom)
  int count[2];         // its key blocks
  int slot[2];          // the text partial it writes, -1 for an output tile
  const int* list[2];   // its key blocks' ids
  // The pair's trip count: the larger of the two tiles' counts, at least 1, so that every role runs the same number
  // of iterations; a tile past its own count runs against clamped (valid, in-bounds) key blocks, masked whole.
  int num_kv_blocks;
};

__device__ __forceinline__ WorkItem decode_workitem(int workitem_id, const Args& a) {
  WorkItem it;
  // Head-major: one head's K and V at a time is what L2 holds (the tiles are of similar counts, so the last head's
  // items leave no tail).
  const int pairs = a.vtpad / 2;
  it.head = workitem_id / pairs;
  const int pair = workitem_id - it.head * pairs;
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    const int vt = a.vorder[(size_t)it.head * a.vtpad + 2 * pair + m];
    int qb, start = 0, count, slot = -1;
    if (vt < a.Nv) {
      qb = a.P + vt;
      count = a.q2k_num[(size_t)it.head * a.num_blocks + qb];
    } else if (vt < a.Nv + a.P * a.tchunks) {
      slot = vt - a.Nv;
      qb = slot / a.tchunks;
      start = (slot - qb * a.tchunks) * a.tlen;
      count = min(a.tlen, a.num_blocks - start);
    } else {
      qb = a.num_blocks;
      count = 0;
    }
    it.qb[m] = qb;
    it.count[m] = count;
    it.slot[m] = slot;
    it.list[m] = a.q2k_idx + ((size_t)it.head * a.num_blocks + min(qb, a.num_blocks - 1)) * a.num_blocks + start;
  }
  it.num_kv_blocks = max(max(it.count[0], it.count[1]), 1);
  return it;
}

__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
    fwd_fine_kernel(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_k,
                    const __grid_constant__ CUtensorMap tmap_v, const __grid_constant__ CUtensorMap tmap_o, Args a) {
#if VSA2T_TCGEN05
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  uint8_t* sQ0 = fwd_smem;
  uint8_t* sQ1 = sQ0 + Q_TILE_BYTES;
  uint8_t* sQ[2] = {sQ0, sQ1};
  uint8_t* sKV = sQ1 + Q_TILE_BYTES;
  __nv_bfloat16* sO0 = reinterpret_cast<__nv_bfloat16*>(sKV + NUM_KV_STAGES * KV_RING_SLOT_BYTES);
  __nv_bfloat16* sO1 = sO0 + M_TILE * HEAD_DIM;
  __nv_bfloat16* sO_bufs[2] = {sO0, sO1};
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sO1) + M_TILE * HEAD_DIM * sizeof(__nv_bfloat16));
  uint64_t* empty_bar = full_bar + NUM_KV_STAGES;
  uint64_t* full_bar_q = empty_bar + NUM_KV_STAGES;
  uint64_t* empty_bar_q = full_bar_q + 2;
  uint64_t* full_bar_spo = empty_bar_q + 2;
  uint64_t* empty_bar_spo = full_bar_spo + 2;
  uint64_t* full_bar_o_acc = empty_bar_spo + 2;
  uint64_t* full_bar_alpha = full_bar_o_acc + 2;
  uint64_t* full_bar_l = full_bar_alpha + 2;
  uint64_t* full_bar_p_last = full_bar_l + 2;
  uint64_t* empty_bar_alpha_and_l = full_bar_p_last + 2;
  uint64_t* full_bar_o_epi = empty_bar_alpha_and_l + 2;
  uint64_t* empty_bar_o_epi = full_bar_o_epi + 2;
  uint64_t* clc_full = empty_bar_o_epi + 2;
  uint64_t* clc_empty = clc_full + CLC_STAGES;
  uint32_t* clc_response =
      reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(clc_empty + CLC_STAGES) + 15u) & ~uintptr_t(15u));
  uint32_t* tmem_slot = clc_response + CLC_STAGES * 4;
  float* alpha_and_l_smem = reinterpret_cast<float*>(tmem_slot + 2);

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

  if (warp_id == 0) {
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), TMEM_TOTAL);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  const uint32_t tmem_base = *tmem_slot;

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
#pragma unroll
    for (int s = 0; s < CLC_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&clc_full[s]), 1);
      mbarrier_init(smem_ptr_u32(&clc_empty[s]), N_WARPS);
    }
#pragma unroll
    for (int i = 0; i < CLC_STAGES * 4; ++i) clc_response[i] = 0;
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();

  if (warp_id == W_LOAD) {
    // ---- TMA loads: the Q pair, then the K / V blocks of each tile's list, interleaved ----
    setmaxnreg_dec<48>();

    EmptyPhaseTracker<NUM_KV_STAGES> kv_empty_ph;
    EmptyPhaseTracker<1> q_empty_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem(workitem_id, a);
      const int num_k_tiles = it.num_kv_blocks;

      int window_start[2] = {-(1 << 30), -(1 << 30)};
      int kv_block_id_cache[2] = {0, 0};
      auto get_kv_block_id = [&](int mtile_idx, int j) -> int {
        if (j >= window_start[mtile_idx] + 32) {
          window_start[mtile_idx] = j & ~31;
          // Positions past the tile's own count re-read its last entry; an empty tile loads block 0. Both are masked
          // whole by the softmax, so only the address matters.
          const int cnt = it.count[mtile_idx];
          const int idx = max(0, min(window_start[mtile_idx] + lane, cnt - 1));
          kv_block_id_cache[mtile_idx] = (cnt > 0) ? it.list[mtile_idx][idx] : 0;
        }
        return __shfl_sync(0xffffffffu, kv_block_id_cache[mtile_idx], j & 31);
      };

      auto load_kv = [&](const CUtensorMap* map, int mtile_idx, int ktile_idx) {
        const int kv_stage = kv_empty_ph.get_stage();
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        const int kv_tok = get_kv_block_id(mtile_idx, ktile_idx) * BLOCK;
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), KV_RING_SLOT_BYTES);
          tma_load_4d(smem_ptr_u32(sKV + kv_stage * KV_RING_SLOT_BYTES), map, smem_ptr_u32(&full_bar[kv_stage]), 0,
                      kv_tok, 0, it.head);
        }
      };

      load_kv(&tmap_k, 0, 0);
      load_kv(&tmap_k, 1, 0);

#pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
        const int tok0 = it.qb[m] * BLOCK;
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), Q_TILE_BYTES);
          tma_load_4d(smem_ptr_u32(sQ[m]), &tmap_q, smem_ptr_u32(&full_bar_q[m]), 0, it.head, tok0, 0);
        }
      }
      q_empty_ph.advance();
      for (int ktile_idx = 0; ktile_idx + 1 < num_k_tiles; ++ktile_idx) {
        load_kv(&tmap_v, 0, ktile_idx);
        load_kv(&tmap_k, 0, ktile_idx + 1);
        load_kv(&tmap_v, 1, ktile_idx);
        load_kv(&tmap_k, 1, ktile_idx + 1);
      }
      load_kv(&tmap_v, 0, num_k_tiles - 1);
      load_kv(&tmap_v, 1, num_k_tiles - 1);

      ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(clc_full, clc_empty, clc_response,
                                                                                     clc_stage, clc_phase,
                                                                                     elect_one_sync());
      clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
      if (!next.valid) break;
      workitem_id = next.n_tile;
    }
  } else if (warp_id == W_MMA) {
    // ---- MMA issue: S = Q K^T into TMEM, O += P V with P from TMEM ----
    setmaxnreg_dec<48>();

    const uint32_t lead = elect_one_sync() ? 1u : 0u;
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;

    const uint32_t idesc_qk = make_idesc_bf16_f32(M_TILE, K_TILE, false, false);
    const uint32_t idesc_pv = make_idesc_bf16_f32(M_TILE, HEAD_DIM, false, true);

    constexpr uint32_t DESC_LBO_MN = (uint32_t)(K_TILE * SUB_COLS_BF16 * 2);
    const uint64_t desc_v0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO_MN, SmemSwizzleBlackwell::B128);
    constexpr uint32_t PV_DESC_STEP = (uint32_t)((16 * SUB_COLS_BF16 * 2) >> 4);
    const uint64_t desc_q0 = build_smem_desc_blackwell(smem_ptr_u32(sQ0), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_kv0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);

    constexpr uint64_t KV_DESC_DELTA = KV_RING_SLOT_BYTES >> 4;
    constexpr uint64_t Q_SUB_DELTA = Q_SUB_COLS_BYTES >> 4;
    constexpr uint64_t Q_MTILE_DESC_DELTA = Q_TILE_BYTES >> 4;

    PhaseTracker<NUM_KV_STAGES> kv_ph;
    PhaseTracker<1> q_ph;
    PhaseTracker<1> spo_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem(workitem_id, a);
      const int num_k_tiles = it.num_kv_blocks;

      auto bmm1 = [&](int i) {
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint64_t da_base = desc_q0 + (uint64_t)i * Q_MTILE_DESC_DELTA;
        const int slot = kv_ph.get_stage();
        mbarrier_wait_parity(smem_ptr_u32(&full_bar[slot]), kv_ph.get_phase());
        kv_ph.advance();
#pragma unroll
        for (int s = 0; s < Q_SUBTILES; ++s) {
          const uint64_t da = da_base + (uint64_t)s * Q_SUB_DELTA;
          const uint64_t db = desc_kv0 + (uint64_t)slot * KV_DESC_DELTA + (uint64_t)s * (K_SUB_COLS_BYTES >> 4);
#pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki)
            tcgen05_mma_f16_ss_lead(lead, s_tmem_addr, da + 2 * ki, db + 2 * ki, idesc_qk, (s != 0) || (ki != 0));
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[slot]));
        tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_spo[i]));
      };

      auto bmm2 = [&](int i, bool first_ktile, auto last_c) {
        constexpr bool last_ktile = decltype(last_c)::value;
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
        const int slot = kv_ph.get_stage();
        mbarrier_wait_parity(smem_ptr_u32(&full_bar[slot]), kv_ph.get_phase());
        kv_ph.advance();
        SmemDescPair desc_bv;
        desc_bv.u64 = desc_v0;
        desc_bv.w.x += (uint32_t)(slot * (int)KV_DESC_DELTA);
#pragma unroll
        for (int p = 0; p < V_SUBTILES; ++p) {
#pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
            const int at = p * K_ATOMS_PER_TILE + ki;
            if (at == SPLIT_P_ATOM) mbarrier_wait_parity(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
            const bool accumulate = (!first_ktile) || (at != 0);
            tcgen05_mma_f16_ts_1sm_lead(lead, o_tmem_addr, s_tmem_addr + (uint32_t)(at * 8), desc_bv.u64, idesc_pv,
                                        accumulate);
            smem_desc_add_lo(desc_bv.u64, PV_DESC_STEP);
          }
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[slot]));
        if constexpr (last_ktile) tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_o_acc[i]));
      };

#pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity(smem_ptr_u32(&full_bar_q[i]), q_ph.get_phase());
        bmm1(i);
      }

      // LATE-ALTERNATION-INVARIANT: do not reorder this nest. `bmm2(i)` then `bmm1(i)`, per tile, inside the k loop,
      // is LOAD-BEARING LATENCY HIDING and not an incidental of the loop. The two tiles' softmax groups publish S
      // through their own barriers (full_bar_spo[2], empty_bar_spo[2], one pair per tile), and no wait couples the
      // two tiles: tile i's PV is in flight while tile 1-i's softmax runs. The same schedule has three parts and
      // all three belong to it -- the prologue's bmm1 pair, this interior loop, and the tail's bmm2 pair.
      // Measured by REMOVING the overlap (J7897, one container, 1132 MHz): fwd_fine 784.9 -> 1115.6 us at s8 and
      // 3469.4 -> 4984.8 us at s20 (+42 %, +44 %), at an unchanged REG:128 STACK:72 LOCAL:0, five of five still
      // match 1.0000, and bwd_fine unmoved at 1649.3 against a base of 1658.2. That build changed TWO things at
      // once -- the MMAs grouped by kind (both empty_bar_spo waits hoisted, then bmm2(0), bmm2(1), bmm1(0),
      // bmm1(1)) AND the loader re-ordered to V0,V1,K0,K1 so the ring slots still pair -- so the regression is the
      // pair's and the two were not isolated. A grouped rewrite has to move both, which is why this invariant
      // covers the loader's order above as much as this loop's. [I126]
      // ---- end invariant ----
      for (int k = 0; k + 1 < num_k_tiles; ++k) {
#pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
          bmm2(i, (k == 0), std::false_type{});
          bmm1(i);
        }
        spo_ph.advance();
      }

#pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar_q[i]));
      q_ph.advance();

#pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
        bmm2(i, (num_k_tiles == 1), std::true_type{});
      }
      spo_ph.advance();

      ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(clc_full, clc_empty, clc_response,
                                                                                     clc_stage, clc_phase,
                                                                                     elect_one_sync());
      clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
      if (!next.valid) break;
      workitem_id = next.n_tile;
    }
  } else if (warp_id == W_EPI) {
    // ---- TMA stores of the merged output tiles ----
    setmaxnreg_dec<48>();
    PhaseTracker<1> full_o_ph;
    if (elect_one_sync()) {
#pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[m]));
    }
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem(workitem_id, a);
#pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_epi[m]), full_o_ph.get_phase());
        if (elect_one_sync()) {
          // A text chunk's partial went to global memory from the correction warps; its group stays empty, so that
          // the waits below still pair group m with buffer m.
          if (it.slot[m] < 0)
            tma_store_4d(&tmap_o, 0, it.head, it.qb[m] * BLOCK, 0, smem_ptr_u32(reinterpret_cast<const uint8_t*>(sO_bufs[m])));
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
      ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(clc_full, clc_empty, clc_response,
                                                                                     clc_stage, clc_phase,
                                                                                     elect_one_sync());
      clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
      if (!next.valid) break;
      workitem_id = next.n_tile;
    }
    // The last stores must land before the CTA exits.
    if (elect_one_sync()) asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
  } else if (warp_id == W_SCHED) {
    // ---- Cluster launch control: claims the next work item for every role ----
    setmaxnreg_dec<48>();
    int prod_stage = 0;
    uint32_t prod_phase = 1;
    int cons_stage = 0;
    uint32_t cons_phase = 0;
    while (true) {
      if (lane == 0) mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
      __syncwarp();
      clc_arrive_expect_tx_cta(smem_ptr_u32(&clc_full[prod_stage]), 16);
      if (lane == 0) clc_try_cancel_async(smem_ptr_u32(&clc_response[prod_stage * 4]), smem_ptr_u32(&clc_full[prod_stage]));
      advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
      ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(clc_full, clc_empty, clc_response,
                                                                                     cons_stage, cons_phase,
                                                                                     elect_one_sync());
      clc_fetch_next_tile_advance<CLC_STAGES>(cons_stage, cons_phase);
      if (!next.valid) break;
    }
    for (int s = 0; s < CLC_STAGES; ++s) {
      if (lane == 0) mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
      __syncwarp();
      advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
    }
  } else if (warp_id >= W_CORR0 && warp_id < W_MMA) {
    // ---- Correction: rescales O when the running max moves; the epilogue merges and stages the output ----
    setmaxnreg_dec<80>();

    const int corr_warp_id = warp_id - W_CORR0;
    const int corr_tid = corr_warp_id * 32 + lane;
    const int corr_row = corr_tid;
    PhaseTracker<1> alpha_ph;
    PhaseTracker<1> o_acc_ph;
    PhaseTracker<1> o_epi_empty_ph;

#pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
      mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
    }

    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem(workitem_id, a);
      const int num_k_tiles = it.num_kv_blocks;

#pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
      }
      alpha_ph.advance();

      for (int k = 1; k < num_k_tiles; ++k) {
#pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
          const float alpha = alpha_and_l_smem[(i * STAT_REGIONS + 0) * STATS + corr_tid];
          mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
          const bool skip = __all_sync(0xffffffffu, alpha == 1.0f);
          if (!skip) {
            const uint32_t o_tmem_addr =
                tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
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
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
        }
        alpha_ph.advance();
      }

#pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_acc[i]), o_acc_ph.get_phase());
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_l[i]), o_acc_ph.get_phase());

        const float l = alpha_and_l_smem[(i * STAT_REGIONS + 1) * STATS + corr_tid];
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
        const float scale_own = (l > 0.f) ? rcp_approx_ftz_f32(l) : 0.f;

        // The merge: out = O / l + gate * coarse on token rows, zero on padding rows and the phantom block; a text
        // chunk writes O / l in fp32 instead.
        const int qblk = it.qb[i];
        const bool in_range = qblk < a.num_blocks;
        const int qb = in_range ? qblk : 0;
        const bool row_ok = in_range && mask_bit(a.rowmask, qb, corr_row);
        const float2 scale2 = f32x2_splat(scale_own);
        const float4* coarse4 = reinterpret_cast<const float4*>(a.coarse + ((size_t)it.head * a.num_blocks + qb) * HEAD_DIM);
        const uint4* gate_row = reinterpret_cast<const uint4*>(a.gate + ((size_t)it.head * a.seqlen + (size_t)qb * BLOCK + corr_row) * HEAD_DIM);

        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
        if (it.slot[i] >= 0) {
          float* dst = a.part_o + (((size_t)it.head * a.P * a.tchunks + it.slot[i]) * BLOCK + corr_row) * HEAD_DIM;
#pragma unroll
          for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
            uint32_t o_regs[16];
            tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
            tcgen05_wait_ld();
            const float2* o2 = reinterpret_cast<const float2*>(o_regs);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float2 lo = fmul2(o2[2 * e], scale2), hi = fmul2(o2[2 * e + 1], scale2);
              *reinterpret_cast<float4*>(dst + c0 + 4 * e) = make_float4(lo.x, lo.y, hi.x, hi.y);
            }
          }
          mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[i]), o_epi_empty_ph.get_phase());
        } else {
#pragma unroll
        for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
          uint32_t o_regs[16];
          tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
          if (c0 == 0) mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[i]), o_epi_empty_ph.get_phase());
          float4 cv[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) cv[e] = __ldg(coarse4 + c0 / 4 + e);
          uint4 gv[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) gv[e] = row_ok ? __ldg(gate_row + c0 / 8 + e) : make_uint4(0u, 0u, 0u, 0u);
          const float2* cf2 = reinterpret_cast<const float2*>(cv);
          const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(gv);
          tcgen05_wait_ld();
          const float2* o2 = reinterpret_cast<const float2*>(o_regs);
          const int s = c0 / SUB_COLS_BF16;
          const int v_base = (c0 % SUB_COLS_BF16) / 8;
          __nv_bfloat16* so_sub = sO_bufs[i] + s * (M_TILE * SUB_COLS_BF16);
#pragma unroll
          for (int vv = 0; vv < 2; ++vv) {
            const int v = v_base + vv;
            float2 r[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float2 gf = __bfloat1622float2(g2[vv * 4 + e]);
              r[e] = row_ok ? ffma2(o2[vv * 4 + e], scale2, fmul2(cf2[vv * 4 + e], gf)) : make_float2(0.f, 0.f);
            }
            uint4 packed;
            packed.x = cvt_f32x2_to_bf16x2(r[0].x, r[0].y);
            packed.y = cvt_f32x2_to_bf16x2(r[1].x, r[1].y);
            packed.z = cvt_f32x2_to_bf16x2(r[2].x, r[2].y);
            packed.w = cvt_f32x2_to_bf16x2(r[3].x, r[3].y);
            *reinterpret_cast<uint4*>(&so_sub[corr_row * SUB_COLS_BF16 + (v ^ (corr_row & 7)) * 8]) = packed;
          }
        }
        }

        tcgen05_fence_before_thread_sync();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
        fence_proxy_async_shared();
        mbarrier_arrive(smem_ptr_u32(&full_bar_o_epi[i]));
      }
      o_acc_ph.advance();
      o_epi_empty_ph.advance();

      ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(clc_full, clc_empty, clc_response,
                                                                                     clc_stage, clc_phase,
                                                                                     elect_one_sync());
      clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
      if (!next.valid) break;
      workitem_id = next.n_tile;
    }
  } else {
    // ---- Softmax: one thread per query row of tile m_tile ----
    setmaxnreg_inc<192>();

    const int warp_id_u = __shfl_sync(0xffffffffu, warp_id, 0);
    const int m_tile = warp_id_u < 4 ? 0 : 1;
    const int warp_in_group = warp_id_u & 3;
    const int sm_tid = warp_in_group * 32 + lane;
    const uint32_t alpha_slot_u32 = smem_ptr_u32(&alpha_and_l_smem[(m_tile * STAT_REGIONS + 0) * STATS + sm_tid]);
    const uint32_t s_tmem_addr = tmem_base + (uint32_t)(m_tile * S_COLS) + ((uint32_t)(warp_in_group * 32) << 16);
    PhaseTracker<1> spo_ph;
    PhaseTracker<1> scale_empty_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    const float scale_log2 = a.scale_log2;

    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem(workitem_id, a);
      const int num_k_tiles = it.num_kv_blocks;
      const int* list = it.list[m_tile];
      const int cnt = it.count[m_tile];

      float m_run = -INFINITY, l_run = 0.f;
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
      scale_empty_ph.advance();

      // The key blocks at list positions [window, window + 32), one per lane; -1 at or past this tile's count, which
      // masks the whole block.
      int kb_window = -(1 << 30);
      int kb_cache = -1;
      auto key_block = [&](int k) -> int {
        if (k >= kb_window + 32) {
          kb_window = k & ~31;
          const int kk = kb_window + lane;
          kb_cache = (kk < cnt) ? list[kk] : -1;
        }
        return __shfl_sync(0xffffffffu, kb_cache, k & 31);
      };

      auto softmax_step = [&](auto is_first_c, int k) {
        constexpr bool IS_FIRST = decltype(is_first_c)::value;
        const int kb = key_block(k);
        const uint4 keep = kb >= 0 ? __ldg(reinterpret_cast<const uint4*>(a.rowmask) + kb) : make_uint4(0u, 0u, 0u, 0u);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_spo[m_tile]), spo_ph.get_phase());

        uint32_t s_regs[S_COLS];
#pragma unroll
        for (int c0 = 0; c0 < S_COLS; c0 += 32)
          tcgen05_ld_32x32b_x32(s_tmem_addr + (uint32_t)c0, *reinterpret_cast<uint32_t(*)[32]>(&s_regs[c0]));

        float* scores = reinterpret_cast<float*>(s_regs);
        float2* scores2 = reinterpret_cast<float2*>(s_regs);
        tcgen05_wait_ld();
        tcgen05_fence_before_thread_sync();

        if ((keep.x & keep.y & keep.z & keep.w) != 0xffffffffu) {
          mask_keep_word(scores + 0, keep.x);
          mask_keep_word(scores + 32, keep.y);
          mask_keep_word(scores + 64, keep.z);
          mask_keep_word(scores + 96, keep.w);
        }

        float rmax0 = m_run, rmax1 = -INFINITY, rmax2 = -INFINITY, rmax3 = -INFINITY;
#pragma unroll
        for (int j = 0; j < S_COLS; j += 8) {
          rmax0 = fmaxf(fmaxf(rmax0, scores[j + 0]), scores[j + 1]);
          rmax1 = fmaxf(fmaxf(rmax1, scores[j + 2]), scores[j + 3]);
          rmax2 = fmaxf(fmaxf(rmax2, scores[j + 4]), scores[j + 5]);
          rmax3 = fmaxf(fmaxf(rmax3, scores[j + 6]), scores[j + 7]);
        }
        float new_m = fmaxf(fmaxf(rmax0, rmax1), fmaxf(rmax2, rmax3));
        new_m = fmaxf(new_m, -FLT_MAX);
        float alpha = 0.0f;
        if constexpr (!IS_FIRST) {
          const float acc_scale_ = (m_run - new_m) * scale_log2;
          if (acc_scale_ >= -(float)RESCALE_THRESHOLD) {
            new_m = m_run;
            alpha = 1.0f;
          } else {
            alpha = ex2_approx_f32(acc_scale_);
          }
          sts_f32(alpha_slot_u32, alpha);
        }
        mbarrier_arrive(smem_ptr_u32(&full_bar_alpha[m_tile]));

        const float2 scale2 = f32x2_splat(scale_log2);
        const float2 neg_m_scaled2 = f32x2_splat(-new_m * scale_log2);
        uint32_t p_regs[S_COLS / 2];
#pragma unroll
        for (int c = 0; c < S_COLS / 2; ++c) {
          const float2 a2 = ffma2(scores2[c], scale2, neg_m_scaled2);
          const int jj = c / EX2_FRG_PAIRS;
          const int kk = 2 * (c % EX2_FRG_PAIRS);
          const bool use_hw = (kk % EX2_FREQ < EX2_FREQ - EX2_RES) || (jj >= EX2_FRG_CNT - 1);
          scores2[c] = use_hw ? make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y)) : ex2_emu_f32x2(a2.x, a2.y);
          p_regs[c] = cvt_f32x2_to_bf16x2(scores2[c].x, scores2[c].y);
        }
        const uint32_t p_tmem_addr = s_tmem_addr;

        tcgen05_st_32x32b_x32(p_tmem_addr, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
        tcgen05_st_32x32b_x16(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[32]));
        tcgen05_wait_st();
        tcgen05_fence_before_thread_sync();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
        tcgen05_st_32x32b_x16(p_tmem_addr + SPLIT_P_COL, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[SPLIT_P_COL]));
        tcgen05_wait_st();
        tcgen05_fence_before_thread_sync();
        mbarrier_arrive(smem_ptr_u32(&full_bar_p_last[m_tile]));

        spo_ph.advance();
        float2 lt2a = make_float2(IS_FIRST ? 0.0f : l_run * alpha, 0.0f), lt2b = make_float2(0.f, 0.f);
        float2 lt2c = make_float2(0.f, 0.f), lt2d = make_float2(0.f, 0.f);
#pragma unroll
        for (int c = 0; c < S_COLS / 2; c += 4) {
          lt2a = fadd2(lt2a, scores2[c + 0]);
          lt2b = fadd2(lt2b, scores2[c + 1]);
          lt2c = fadd2(lt2c, scores2[c + 2]);
          lt2d = fadd2(lt2d, scores2[c + 3]);
        }
        const float2 lt2 = fadd2(fadd2(lt2a, lt2b), fadd2(lt2c, lt2d));
        l_run = lt2.x + lt2.y;
        m_run = new_m;
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
        scale_empty_ph.advance();
      };
      if (num_k_tiles > 0) softmax_step(std::true_type{}, 0);
      for (int k = 1; k < num_k_tiles; ++k) softmax_step(std::false_type{}, k);

      // The row's log-sum-exp in natural units: ln(l) + m * scale (a text chunk's into its partial).
      {
        const int qblk = it.qb[m_tile], slot = it.slot[m_tile];
        const float l_safe = l_run > 0.f ? l_run : 1.f;
        const float row_lse = (m_run * scale_log2 + lg2_approx_f32(l_safe)) * 0.69314718055994531f;
        if (slot >= 0) a.part_lse[((size_t)it.head * a.P * a.tchunks + slot) * BLOCK + sm_tid] = row_lse;
        else if (qblk < a.num_blocks) a.lse[(size_t)it.head * a.seqlen + (size_t)qblk * BLOCK + sm_tid] = row_lse;
      }

      alpha_and_l_smem[(m_tile * STAT_REGIONS + 1) * STATS + sm_tid] = l_run;
      mbarrier_arrive(smem_ptr_u32(&full_bar_l[m_tile]));

      ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(clc_full, clc_empty, clc_response,
                                                                                     clc_stage, clc_phase,
                                                                                     elect_one_sync());
      clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
      if (!next.valid) break;
      workitem_id = next.n_tile;
    }
  }
  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL);
#else
  __trap();  // built for an architecture without tcgen05 (primitives.cuh)
#endif
}

// The text rows: each text block's chunks merged by their lse, out = O + gate * coarse (zero on padding rows), and the
// row's lse. One warp per (head, text row), four dims per lane.
__global__ void __launch_bounds__(256) fwd_merge_kernel(Args a, __nv_bfloat16* __restrict__ out) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  const long long wid = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const long long rows = (long long)a.P * BLOCK;
  if (wid >= (long long)a.heads * rows) return;
  const int hd = (int)(wid / rows), r = (int)(wid - (long long)hd * rows);
  const int b = r / BLOCK, rr = r % BLOCK;
  const size_t slot0 = (size_t)hd * a.P * a.tchunks + (size_t)b * a.tchunks;
  float m = -INFINITY;
  for (int c = 0; c < a.tchunks; ++c) m = fmaxf(m, a.part_lse[(slot0 + c) * BLOCK + rr]);
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  float wsum = 0.f;
  for (int c = 0; c < a.tchunks; ++c) {
    const float wgt = __expf(a.part_lse[(slot0 + c) * BLOCK + rr] - m);
    const float4 o = *reinterpret_cast<const float4*>(a.part_o + ((slot0 + c) * BLOCK + rr) * HEAD_DIM + lane * 4);
    wsum += wgt;
    acc.x += wgt * o.x;
    acc.y += wgt * o.y;
    acc.z += wgt * o.z;
    acc.w += wgt * o.w;
  }
  const float inv = 1.f / wsum;
  const bool ok = mask_bit(a.rowmask, b, rr);
  const size_t at = ((size_t)hd * a.seqlen + r) * HEAD_DIM + lane * 4;
  const float4 cv = *reinterpret_cast<const float4*>(a.coarse + ((size_t)hd * a.num_blocks + b) * HEAD_DIM + lane * 4);
  const uint2 gv = *reinterpret_cast<const uint2*>(a.gate + at);
  const float2 g01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gv.x));
  const float2 g23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gv.y));
  uint2 res = make_uint2(0u, 0u);
  if (ok) {
    res.x = cvt_f32x2_to_bf16x2(acc.x * inv + g01.x * cv.x, acc.y * inv + g01.y * cv.y);
    res.y = cvt_f32x2_to_bf16x2(acc.z * inv + g23.x * cv.z, acc.w * inv + g23.y * cv.w);
  }
  *reinterpret_cast<uint2*>(out + at) = res;
  if (lane == 0) a.lse[(size_t)hd * a.seqlen + r] = m + __logf(wsum);
}

}  // namespace fwd
}  // namespace vsa2t
