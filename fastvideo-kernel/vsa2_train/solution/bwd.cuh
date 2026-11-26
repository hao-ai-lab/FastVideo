// bwd.cuh -- the fine branch's backward: dK and dV of each key block over the query blocks that kept it, dQ partials
// reduced into an fp32 accumulator; the coarse branch's gradients are added where each output is written.
//
// Adapted from FastVideo's ptx_blackwell block-sparse attention backward at 128-token blocks (hao-ai-lab/FastVideo,
// csrc/attention/block_sparse_bwd_kernel_blk128_blackwell.cuh at 7c2b44726, Apache License 2.0), in its [batch, head,
// token, dim] configuration, made persistent. A work item is one key block (the selection's transpose decides the
// work): the CTA keeps its K and V in shared memory and dK, dV in TMEM, and for each query block of its list computes
//   S^T = K Q^T, dP^T = V dO^T (TMEM), P^T = exp2(S^T * scale_log2 - lse2) and dS^T = P^T (dP^T - D) (bf16, back to
//   TMEM over S^T and dP^T), dV += P^T dO, dK += dS^T Q (A from TMEM), dQ = dS K (A from shared memory),
// and the dQ tile is drained by the epilogue warps into dqaccum with bulk asynchronous reductions.
// Warps: 0-3 dQ drain, 4-11 softmax (4 TMEM lane quarters x 2 column halves), 12 MMA, 13 load, 14 scheduler, 15 idle.
// Persistence: the grid still has one CTA per item, in the processing order, but a CTA does not exit after its item:
// near the end of it (when the load warp reaches the second-last query block of the list, CLAIM_LEAD) the scheduler
// warp claims the next not-yet-launched CTA
// with cluster launch control and publishes that CTA's item in a two-slot ring that every role reads in turn. The
// item boundary then costs what it costs inside a list -- the next item's first tiles load while this one drains --
// instead of a CTA's exit, launch and prologue (J5405: 8.2 % of the s8 kernel's SM time went to those, 3.5 % at s20).
// Across an item boundary: K and V are reloaded once the last item's dQ product (the last reader of sK, sV and
// sDST) completes (kv_free); its dV and dK tiles leave TMEM through sDST, and the next item's first dV and dK
// products wait until they have been read (dv_free, dk_free). The item's last dQ tile is drained as any other, but
// its reductions are pushed only once both dK stores are issued (egress_go): the bulk copies leave in issue order,
// and the next item's first dS tile waits for sDST, i.e. for the dK store, while nothing waits for the last dQ.
// Changes from FastVideo's kernel:
//   - a key row is masked by the key block's row mask (rowmask) instead of a valid-prefix length;
//   - a query row without a token has lse2 = +inf (bwd_rows_kernel), so its P is 0;
//   - dK and dV rows get the coarse branch's per-token share added, and padding rows are written as zeros;
//   - the items are (head, key block) pairs: every head's text key blocks first, then each head's video key
//     blocks in turn, heaviest first (topk.cuh), so that one head's Q, dO and dqaccum at a time live in L2 (heads
//     interleaved read 25 GB and wrote 13 GB of DRAM at s20, and the power that took cost ~15% of the clock);
//   - D = rowsum(dO * O_fine) comes from bwd_rows_kernel, which subtracts the gated coarse share from `out`;
//   - no programmatic dependent launch.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>

#include <type_traits>

#include "common.cuh"
#include "primitives.cuh"

namespace vsa2t {
namespace bwd {

constexpr int BLOCK = 128;
constexpr int KV_TILE = BLOCK;
constexpr int Q_TILE = BLOCK;
constexpr int HEAD_DIM = 128;
constexpr int SUB_COLS_BF16 = 64;
constexpr int SUB_COLS_BYTES = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
constexpr int HD_SUBTILES = HEAD_DIM / SUB_COLS_BF16;
constexpr int KV_SUB_COLS_BYTES = KV_TILE * SUB_COLS_BYTES;
constexpr int KV_TILE_BYTES = HD_SUBTILES * KV_SUB_COLS_BYTES;
constexpr int Q_SUB_COLS_BYTES = Q_TILE * SUB_COLS_BYTES;
constexpr int Q_TILE_BYTES = HD_SUBTILES * Q_SUB_COLS_BYTES;
constexpr int NUM_Q_STAGES = 2;
constexpr int DST_TILE_BYTES = KV_TILE * SUB_COLS_BYTES;
constexpr int DST_BYTES = 2 * DST_TILE_BYTES;
constexpr int MMA_K = 16;
constexpr int K_ATOMS_PER_SUBTILE = SUB_COLS_BF16 / MMA_K;
constexpr int K_ATOMS_PER_Q_HALF = SUB_COLS_BF16 / MMA_K;
constexpr int K_ATOMS_PER_KV_TILE = KV_TILE / MMA_K;
constexpr int BF16X2_COLS_PER_K16 = MMA_K / 2;

// dQ leaves TMEM in slices of 32 head-dim columns, staged in shared memory as [32 / 4][128 rows][4]: the "drain-native"
// layout of dqaccum, which bwd_post_kernel undoes.
constexpr int DQ_COLS = 32;
constexpr int DQ_ONE_PUSH_BYTES = Q_TILE * DQ_COLS * (int)sizeof(float);
constexpr int DQ_STAGE_BYTES = DQ_ONE_PUSH_BYTES;
constexpr int DQ_STAGE_BUFFERS = 2;
constexpr int DQ_BLOCK_ELEMS = Q_TILE * HEAD_DIM;

constexpr int N_WARPS = 16;
constexpr int W_EPI0 = 0, W_SOFTMAX0 = 4, W_MMA = 12, W_LOAD = 13, W_SCHED = 14;

// The persistent loop's item ring: two slots, each written by the scheduler warp and read by the 14 warps of the
// roles (0-13).
constexpr int ITEM_STAGES = 2;
constexpr int ITEM_READERS = 14;
// The load warp asks for the next item at this many query blocks before the end of its list: late enough that a CTA
// holds no item it is far from starting (an early claim keeps the last items from the SMs that free up first),
// early enough that the claim and the item's decode are done before its first tiles are wanted.
constexpr int CLAIM_LEAD = 2;

// Six barriers per stage -- full and empty for q, lse and delta -- and ten single ones: full/empty do, and full st,
// dpt, pt, dst, dv, dk with the empty dq pair; then the persistent loop's: full and empty per item slot, the claim
// request, the cluster launch control response, and kv_free, dv_free, dk_free, egress_go, egress_ack. After them,
// 16-byte aligned: the response, the item slots and tmem_slot, inside the tail that SMEM_TOTAL's last terms reserve.
constexpr int NUM_BARS = 6 * NUM_Q_STAGES + 10 + 2 * ITEM_STAGES + 2 + 5;
constexpr int BAR_OFFSET = 2 * KV_TILE_BYTES + NUM_Q_STAGES * Q_TILE_BYTES + Q_TILE_BYTES + DST_BYTES +
                           DQ_STAGE_BUFFERS * DQ_STAGE_BYTES + 2 * NUM_Q_STAGES * Q_TILE * (int)sizeof(float);
constexpr int CLC_OFFSET = (BAR_OFFSET + NUM_BARS * 8 + 15) / 16 * 16;
constexpr int ITEM_DESC_BYTES = 32;
constexpr int SMEM_TOTAL = CLC_OFFSET + 16 + ITEM_STAGES * ITEM_DESC_BYTES + 64;
static_assert(BAR_OFFSET % 16 == 0 && CLC_OFFSET % 16 == 0, "16-byte alignment of the barriers and the CLC response");
static_assert(SMEM_TOTAL <= 232448, "the fine backward's shared memory exceeds the 227 KB a CTA can have");

constexpr int ST_COLS = Q_TILE;
constexpr int ST_HALF_COLS = SUB_COLS_BF16;
constexpr int DV_COLS = HEAD_DIM;
constexpr int DK_COLS = HEAD_DIM;
constexpr int TMEM_TOTAL = ST_COLS + DV_COLS + ST_COLS + DK_COLS;
static_assert(ST_COLS == 2 * ST_HALF_COLS, "two q-halves per tile");
static_assert(TMEM_TOTAL == 512, "TMEM map must fill exactly 512 columns");

struct Args {
  int heads, num_blocks, P;
  long long seqlen;
  float scale_log2, sm_scale;
  float* dqaccum;              // [heads][S][128] fp32, drain-native per 128-row block
  const float* lse2;           // [heads][S] log2-unit lse, +inf on padding rows
  const float* delta;          // [heads][S]
  const int* k2q_idx;          // [heads * N][N]
  const int* k2q_num;          // [heads * N]
  const int* korder;           // [heads][N] key blocks in processing order
  const uint32_t* rowmask;     // [N][4]
  const float* dk_coarse;      // [heads][N][128] the coarse branch's dK share per token of the block
  const float* dv_coarse;      // [heads][N][128]
};

extern __shared__ __align__(1024) uint8_t bwd_smem[];

struct WorkItem {
  int head;
  int kv_block;
  const int* list;
  int count;
};

// One slot of the item ring: the item of CTA index x, decoded by the scheduler warp, with the key block's row mask.
struct __align__(16) ItemDesc {
  int valid, head, kv_block, count;
  uint32_t mask[4];
};
static_assert(sizeof(ItemDesc) == ITEM_DESC_BYTES, "item slot size");

// The block index of the CTA a clusterlaunchcontrol.try_cancel response cancelled (whose item this CTA now owns), or
// -1 when the request failed: every CTA of the grid has then started, and this one must not ask again (PTX ISA,
// clusterlaunchcontrol.try_cancel: a request after an observed failure is undefined).
__device__ __forceinline__ int clc_cancelled_block(uint32_t response_smem) {
  uint32_t ok, x;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b128 r;\n"
      "ld.shared::cta.b128 r, [%2];\n"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r;\n"
      "selp.u32 %1, 1, 0, p;\n"
      "mov.u32 %0, 0;\n"
      "@p clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, r;\n"
      "}\n"
      : "=r"(x), "=r"(ok)
      : "r"(response_smem)
      : "memory");
  return ok ? (int)x : -1;
}

__device__ __forceinline__ WorkItem decode_workitem(const Args& a, int x) {
  WorkItem it;
  // Every head's text key blocks first (the heaviest items: they start first, side by side), then each head's video
  // key blocks in turn, heaviest first: one head's Q, dO and dqaccum at a time is about what L2 holds, where four
  // heads interleaved would stream them from DRAM.
  const int text_items = a.P * a.heads;
  int pos;
  if (x < text_items) {
    pos = x / a.heads;
    it.head = x - pos * a.heads;
  } else {
    const int v = x - text_items, per_head = a.num_blocks - a.P;
    it.head = v / per_head;
    pos = a.P + (v - it.head * per_head);
  }
  it.kv_block = a.korder[(size_t)it.head * a.num_blocks + pos];
  const size_t item = (size_t)it.head * a.num_blocks + it.kv_block;
  it.list = a.k2q_idx + item * (size_t)a.num_blocks;
  it.count = a.k2q_num[item];
  return it;
}

__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
    bwd_fine_kernel(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_k,
                    const __grid_constant__ CUtensorMap tmap_v, const __grid_constant__ CUtensorMap tmap_do,
                    const __grid_constant__ CUtensorMap tmap_dk, const __grid_constant__ CUtensorMap tmap_dv, Args a) {
#if VSA2T_TCGEN05
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  const long long seqlen = a.seqlen;
  uint8_t* sK = bwd_smem;
  uint8_t* sV = sK + KV_TILE_BYTES;
  uint8_t* sQ[NUM_Q_STAGES] = {sV + KV_TILE_BYTES, sV + KV_TILE_BYTES + Q_TILE_BYTES};
  uint8_t* sDO = sQ[0] + NUM_Q_STAGES * Q_TILE_BYTES;
  __nv_bfloat16* sDST = reinterpret_cast<__nv_bfloat16*>(sDO + Q_TILE_BYTES);
  uint8_t* sDQ_STAGE_bytes = sDO + Q_TILE_BYTES + DST_BYTES;
  float* sDQ_STAGE[DQ_STAGE_BUFFERS] = {reinterpret_cast<float*>(sDQ_STAGE_bytes),
                                        reinterpret_cast<float*>(sDQ_STAGE_bytes + DQ_STAGE_BYTES)};
  float* sLSE = reinterpret_cast<float*>(sDQ_STAGE_bytes + DQ_STAGE_BUFFERS * DQ_STAGE_BYTES);
  float* sDelta = sLSE + NUM_Q_STAGES * Q_TILE;

  uint64_t* full_bar_q = reinterpret_cast<uint64_t*>(sDelta + NUM_Q_STAGES * Q_TILE);
  uint64_t* empty_bar_q = full_bar_q + NUM_Q_STAGES;
  uint64_t* full_bar_do = empty_bar_q + NUM_Q_STAGES;
  uint64_t* empty_bar_do = full_bar_do + 1;
  uint64_t* full_bar_lse = empty_bar_do + 1;
  uint64_t* empty_bar_lse = full_bar_lse + NUM_Q_STAGES;
  uint64_t* full_bar_delta = empty_bar_lse + NUM_Q_STAGES;
  uint64_t* empty_bar_delta = full_bar_delta + NUM_Q_STAGES;
  uint64_t* full_bar_st = empty_bar_delta + NUM_Q_STAGES;
  uint64_t* full_bar_dpt = full_bar_st + 1;
  uint64_t* full_bar_pt = full_bar_dpt + 1;
  uint64_t* full_bar_dst = full_bar_pt + 1;
  uint64_t* full_bar_dq = full_bar_dst + 1;
  uint64_t* empty_bar_dq = full_bar_dq + 1;
  uint64_t* full_bar_dv = empty_bar_dq + 1;
  uint64_t* full_bar_dk = full_bar_dv + 1;
  uint64_t* item_full = full_bar_dk + 1;
  uint64_t* item_empty = item_full + ITEM_STAGES;
  uint64_t* claim_bar = item_empty + ITEM_STAGES;
  uint64_t* clc_full = claim_bar + 1;
  uint64_t* kv_free = clc_full + 1;
  uint64_t* dv_free = kv_free + 1;
  uint64_t* dk_free = dv_free + 1;
  uint64_t* egress_go = dk_free + 1;
  uint64_t* egress_ack = egress_go + 1;
  uint32_t* clc_response = reinterpret_cast<uint32_t*>(bwd_smem + CLC_OFFSET);
  ItemDesc* item_desc = reinterpret_cast<ItemDesc*>(bwd_smem + CLC_OFFSET + 16);
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(bwd_smem + CLC_OFFSET + 16 + ITEM_STAGES * ITEM_DESC_BYTES);

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
#pragma unroll
    for (int s = 0; s < NUM_Q_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar_delta[s]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_delta[s]), 8);
    }
    mbarrier_init(smem_ptr_u32(full_bar_st), 1);
    mbarrier_init(smem_ptr_u32(full_bar_dpt), 1);
    mbarrier_init(smem_ptr_u32(full_bar_pt), 8);
    mbarrier_init(smem_ptr_u32(full_bar_dst), 8);
    mbarrier_init(smem_ptr_u32(full_bar_dq), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_dq), 4);
    mbarrier_init(smem_ptr_u32(full_bar_dv), 1);
    mbarrier_init(smem_ptr_u32(full_bar_dk), 1);
#pragma unroll
    for (int s = 0; s < ITEM_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&item_full[s]), 1);
      mbarrier_init(smem_ptr_u32(&item_empty[s]), ITEM_READERS);
    }
    mbarrier_init(smem_ptr_u32(claim_bar), 1);
    mbarrier_init(smem_ptr_u32(clc_full), 1);
    mbarrier_init(smem_ptr_u32(kv_free), 1);
    mbarrier_init(smem_ptr_u32(dv_free), 8);
    mbarrier_init(smem_ptr_u32(dk_free), 8);
    // One arrival per column half: the lane that issues that half's dK store (softmax, lane quarter 0).
    mbarrier_init(smem_ptr_u32(egress_go), 2);
    // The drain leader's acknowledgement of each egress_go phase it has consumed.
    mbarrier_init(smem_ptr_u32(egress_ack), 1);
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();

  // The item each role is on; every role takes the items from the ring in the same order.
  WorkItem it;
  auto take_item = [&](PhaseTracker<ITEM_STAGES>& ph, uint4& mask) -> bool {
    const int s = ph.get_stage();
    mbarrier_wait_parity_suspend(smem_ptr_u32(&item_full[s]), ph.get_phase());
    // Plain loads: the wait above is a compiler barrier (an asm with a memory clobber) and an acquire.
    const int4 d = *reinterpret_cast<const int4*>(&item_desc[s]);
    const uint4 m = *reinterpret_cast<const uint4*>(&item_desc[s].mask[0]);
    __syncwarp();
    if (elect_one_sync()) mbarrier_arrive(smem_ptr_u32(&item_empty[s]));
    ph.advance();
    it.head = d.y;
    it.kv_block = d.z;
    it.count = d.w;
    it.list = a.k2q_idx + ((size_t)d.y * a.num_blocks + d.z) * (size_t)a.num_blocks;
    mask = m;
    return d.x != 0;
  };

  auto load_tile = [&](uint8_t* dst, const CUtensorMap* map, uint64_t* full_bar, int token_begin) {
#pragma unroll
    for (int s = 0; s < HD_SUBTILES; ++s)
      tma_load_4d(smem_ptr_u32(dst + s * Q_SUB_COLS_BYTES), map, smem_ptr_u32(full_bar), 0, token_begin, s, it.head);
  };

  if (warp_id == W_LOAD) {
    setmaxnreg_dec<56>();

    EmptyPhaseTracker<NUM_Q_STAGES> q_empty_ph, lse_empty_ph;
    EmptyPhaseTracker<1> do_empty_ph;
    EmptyPhaseTracker<NUM_Q_STAGES> delta_empty_ph;
    PhaseTracker<ITEM_STAGES> item_ph;
    uint32_t kvf_ph = 0;
    bool kv_busy = false;  // an earlier item's products may still read sK and sV
    uint4 mask_unused;

    auto load_lse = [&](int qblock) {
      const int stage = lse_empty_ph.get_stage();
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_lse[stage]), lse_empty_ph.get_phase());
      lse_empty_ph.advance();
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_lse[stage]), Q_TILE * (int)sizeof(float));
        cpasync_bulk_load_mbarrier(smem_ptr_u32(sLSE + stage * Q_TILE),
                                   a.lse2 + (size_t)it.head * seqlen + (size_t)qblock * Q_TILE,
                                   Q_TILE * sizeof(float), smem_ptr_u32(&full_bar_lse[stage]));
      }
    };
    auto load_delta = [&](int qblock) {
      const int stage = delta_empty_ph.get_stage();
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_delta[stage]), delta_empty_ph.get_phase());
      delta_empty_ph.advance();
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_delta[stage]), Q_TILE * (int)sizeof(float));
        cpasync_bulk_load_mbarrier(smem_ptr_u32(sDelta + stage * Q_TILE),
                                   a.delta + (size_t)it.head * seqlen + (size_t)qblock * Q_TILE,
                                   Q_TILE * sizeof(float), smem_ptr_u32(&full_bar_delta[stage]));
      }
    };
    // Q (and dO) of a query block; the first of an item's list also expects the item's K (and V) on the same barrier,
    // whose loads the item's prologue below issues once the previous item has released sK and sV (kv_free).
    auto load_q = [&](int qblock, bool with_kv) {
      const int stage = q_empty_ph.get_stage();
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[stage]), q_empty_ph.get_phase());
      q_empty_ph.advance();
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[stage]), Q_TILE_BYTES + (with_kv ? KV_TILE_BYTES : 0));
        // sQ[0] + offset, not sQ[stage]: a runtime index into the pointer array puts it in local memory.
        load_tile(sQ[0] + stage * Q_TILE_BYTES, &tmap_q, &full_bar_q[stage], qblock * Q_TILE);
      }
      return stage;
    };
    auto load_do = [&](int qblock, bool with_kv) {
      mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_do), do_empty_ph.get_phase());
      do_empty_ph.advance();
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(smem_ptr_u32(full_bar_do), Q_TILE_BYTES + (with_kv ? KV_TILE_BYTES : 0));
        load_tile(sDO, &tmap_do, full_bar_do, qblock * Q_TILE);
      }
    };

    while (take_item(item_ph, mask_unused)) {
      const int claim_at = it.count > CLAIM_LEAD ? it.count - CLAIM_LEAD : 0;
      if (it.count == 0) {
        if (elect_one_sync()) mbarrier_arrive(smem_ptr_u32(claim_bar));
        continue;
      }
      const int q0 = it.list[0];
      const int stage0 = load_q(q0, true);
      load_lse(q0);
      load_do(q0, true);
      load_delta(q0);
      if (kv_busy) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(kv_free), kvf_ph);
        kvf_ph ^= 1;
      }
      if (elect_one_sync()) {
        load_tile(sK, &tmap_k, &full_bar_q[stage0], it.kv_block * KV_TILE);
        load_tile(sV, &tmap_v, full_bar_do, it.kv_block * KV_TILE);
      }
      kv_busy = true;
      if (claim_at == 0 && elect_one_sync()) mbarrier_arrive(smem_ptr_u32(claim_bar));
      for (int j = 1; j < it.count; ++j) {
        const int qblock = it.list[j];
        if (j == claim_at && elect_one_sync()) mbarrier_arrive(smem_ptr_u32(claim_bar));
        load_q(qblock, false);
        load_lse(qblock);
        load_do(qblock, false);
        load_delta(qblock);
      }
    }
    return;
  } else if (warp_id == W_MMA) {
    setmaxnreg_dec<56>();

    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), TMEM_TOTAL);
    bar_sync<10>(416);
    const uint32_t tmem_base = *tmem_slot;
    const uint32_t tmem_st = tmem_base;
    const uint32_t tmem_dv = tmem_st + ST_COLS;
    const uint32_t tmem_dpt = tmem_dv + DV_COLS;
    const uint32_t tmem_dk = tmem_dpt + ST_COLS;
    const uint32_t tmem_pt_bf16 = tmem_st, tmem_dst_bf16 = tmem_dpt, tmem_dq = tmem_dpt;
    const uint32_t lead = elect_one_sync() ? 1u : 0u;

    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    auto make_smem_desc = [](const void* smem, uint32_t leading_byte_offset) {
      return build_smem_desc_blackwell(smem_ptr_u32(smem), DESC_SBO, leading_byte_offset, SmemSwizzleBlackwell::B128);
    };
    const uint64_t desc_k = make_smem_desc(sK, DESC_LBO);
    const uint64_t desc_v = make_smem_desc(sV, DESC_LBO);
    const uint64_t desc_q0 = make_smem_desc(sQ[0], DESC_LBO);
    const uint64_t desc_do = make_smem_desc(sDO, DESC_LBO);
    const uint64_t desc_q0_mn = make_smem_desc(sQ[0], Q_SUB_COLS_BYTES);
    const uint64_t desc_do_mn = make_smem_desc(sDO, Q_SUB_COLS_BYTES);
    const uint64_t desc_k_mn = make_smem_desc(sK, KV_SUB_COLS_BYTES);
    const uint64_t desc_dst_mn = make_smem_desc(sDST, DST_TILE_BYTES);

    constexpr uint64_t K16_COLS_DELTA = (MMA_K * (int)sizeof(__nv_bfloat16)) >> 4;
    constexpr uint32_t K16_ROWS_DELTA = (MMA_K * SUB_COLS_BYTES) >> 4;
    constexpr uint64_t Q_SUB_DELTA = Q_SUB_COLS_BYTES >> 4;
    constexpr uint64_t KV_SUB_DELTA = KV_SUB_COLS_BYTES >> 4;
    constexpr uint64_t Q_STAGE_DELTA = Q_TILE_BYTES >> 4;

    const uint32_t idesc_st_dpt = make_idesc_bf16_f32(KV_TILE, Q_TILE, false, false);
    const uint32_t idesc_dv_dk = make_idesc_bf16_f32(KV_TILE, HEAD_DIM, false, true);
    const uint32_t idesc_dq = make_idesc_bf16_f32(Q_TILE, HEAD_DIM, true, true);

    PhaseTracker<NUM_Q_STAGES> q_full_ph;
    PhaseTracker<1> do_full_ph, pt_ph, dst_ph, dq_empty_ph;
    PhaseTracker<ITEM_STAGES> item_ph;
    uint32_t dvf_ph = 0, dkf_ph = 0;
    bool prev = false;  // an earlier item left its dV and dK in TMEM and its last dQ to be drained
    uint4 mask_unused;

    auto gemm12_st_dpt = [&](auto is_st_const, int stage) {
      constexpr bool is_st = decltype(is_st_const)::value;
      const uint32_t tmem_acc = is_st ? tmem_st : tmem_dpt;
      const uint64_t da_base = is_st ? desc_k : desc_v;
      // Every descriptor offset in this warp is a 32-bit add on the descriptor's low word (exact: the 14-bit address
      // field never carries into LBO). Do not turn these back into 64-bit adds: CUDA 13.0 ptxas can lower
      // `desc_q0 + stage * Q_STAGE_DELTA + ...` to a low-half-only add and drop the high word (SBO | version |
      // 128B swizzle), so that S(j+1) reads Q unswizzled from the second query block on (M14616: dq/dk/dv wrong,
      // deterministic, whenever a code change elsewhere in warpgroup 3 reshapes this loop's PTX).
      uint64_t db_base = is_st ? desc_q0 : desc_do;
      if constexpr (is_st) smem_desc_add_lo(db_base, (uint32_t)stage * (uint32_t)Q_STAGE_DELTA);
      uint64_t* commit_bar = is_st ? full_bar_st : full_bar_dpt;
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
          uint64_t da = da_base, db = db_base;
          smem_desc_add_lo(da, (uint32_t)(s * KV_SUB_DELTA + ki * K16_COLS_DELTA));
          smem_desc_add_lo(db, (uint32_t)(s * Q_SUB_DELTA + ki * K16_COLS_DELTA));
          tcgen05_mma_f16_ss_lead(lead, tmem_acc, da, db, idesc_st_dpt, accumulate);
        }
      }
      tcgen05_commit1_lead(lead, smem_ptr_u32(commit_bar));
    };

    auto gemm35_dv_dk = [&](auto is_dv_const, int stage, bool first) {
      constexpr bool is_dv = decltype(is_dv_const)::value;
      const uint32_t tmem_acc = is_dv ? tmem_dv : tmem_dk;
      const uint32_t tmem_a_base = is_dv ? tmem_pt_bf16 : tmem_dst_bf16;
      uint64_t db = is_dv ? desc_do_mn : desc_q0_mn;  // 32-bit low-word offset only: see gemm12_st_dpt
      if constexpr (!is_dv) smem_desc_add_lo(db, (uint32_t)stage * (uint32_t)Q_STAGE_DELTA);
      uint64_t* commit_bar = is_dv ? empty_bar_do : &empty_bar_q[stage];
#pragma unroll
      for (int q_half = 0; q_half < 2; ++q_half) {
#pragma unroll
        for (int ki = 0; ki < K_ATOMS_PER_Q_HALF; ++ki) {
          const uint32_t tmem_a = tmem_a_base + (uint32_t)(q_half * ST_HALF_COLS + ki * BF16X2_COLS_PER_K16);
          const bool accumulate = !(first && (q_half | ki) == 0);
          tcgen05_mma_f16_ts_1sm_lead(lead, tmem_acc, tmem_a, db, idesc_dv_dk, accumulate);
          smem_desc_add_lo(db, K16_ROWS_DELTA);
        }
      }
      tcgen05_commit1_lead(lead, smem_ptr_u32(commit_bar));
    };

    auto gemm4_dq = [&]() {
      uint64_t adst = desc_dst_mn;
      uint64_t bk = desc_k_mn;
#pragma unroll
      for (int ki = 0; ki < K_ATOMS_PER_KV_TILE; ++ki) {
        tcgen05_mma_f16_ss_lead(lead, tmem_dq, adst, bk, idesc_dq, ki != 0);
        smem_desc_add_lo(adst, K16_ROWS_DELTA);
        smem_desc_add_lo(bk, K16_ROWS_DELTA);
      }
      tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dq));
    };

    while (take_item(item_ph, mask_unused)) {
      if (it.count == 0) continue;
      gemm12_st_dpt(std::true_type{}, q_full_ph.get_stage());
      if (prev) {  // dP(0) overwrites the TMEM columns of the previous item's last dQ
        mbarrier_wait_parity(smem_ptr_u32(empty_bar_dq), dq_empty_ph.get_phase());
        dq_empty_ph.advance();
      }
      gemm12_st_dpt(std::false_type{}, 0);
      mbarrier_wait_parity(smem_ptr_u32(full_bar_pt), pt_ph.get_phase());
      pt_ph.advance();
      if (prev) {  // dV(0) restarts the accumulator the softmax warps are reading out
        mbarrier_wait_parity(smem_ptr_u32(dv_free), dvf_ph);
        dvf_ph ^= 1;
      }
      gemm35_dv_dk(std::true_type{}, 0, true);

      for (int j = 0; j < it.count; ++j) {
        const int stage = q_full_ph.get_stage();
        q_full_ph.advance();
        const int next_stage = q_full_ph.get_stage();
        const bool last = j + 1 == it.count;

        if (!last) gemm12_st_dpt(std::true_type{}, next_stage);
        if (last) tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dv));
        mbarrier_wait_parity(smem_ptr_u32(full_bar_dst), dst_ph.get_phase());
        dst_ph.advance();
        if (j == 0 && prev) {  // dK(0) restarts the dK accumulator
          mbarrier_wait_parity(smem_ptr_u32(dk_free), dkf_ph);
          dkf_ph ^= 1;
        }
        gemm35_dv_dk(std::false_type{}, stage, j == 0);
        if (last) tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dk));
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
      // The item's last dQ product is the last reader of sK, sV and sDST.
      tcgen05_commit1_lead(lead, smem_ptr_u32(kv_free));
      prev = true;
    }
    tcgen05_relinquish_alloc_permit<1>();
    bar_sync<10>(416);
    tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL);
    return;
  } else if (warp_id >= W_SOFTMAX0 && warp_id < W_MMA) {
    setmaxnreg_inc<160>();
    bar_sync<10>(416);
    const uint32_t tmem_base = *tmem_slot;
    const uint32_t tmem_st = tmem_base;
    const uint32_t tmem_dv = tmem_st + ST_COLS;
    const uint32_t tmem_dpt = tmem_dv + DV_COLS;
    const uint32_t tmem_dk = tmem_dpt + ST_COLS;
    const uint32_t tmem_pt_bf16 = tmem_st, tmem_dst_bf16 = tmem_dpt;

    const int softmax_warp_id = warp_id - W_SOFTMAX0;
    const int lane_group = softmax_warp_id & 3;
    const int col_half = softmax_warp_id >> 2;
    const int kv_row = lane_group * 32 + lane;

    const uint32_t tmem_lane_base = (uint32_t)(lane_group * 32) << 16;
    const uint32_t tmem_f32_offset = tmem_lane_base + (uint32_t)(col_half * ST_HALF_COLS);
    const uint32_t tmem_bf16x2_offset = tmem_f32_offset;

    constexpr int CHUNK_BF16 = 16 / (int)sizeof(__nv_bfloat16);
    constexpr int CHUNKS_PER_ROW = SUB_COLS_BF16 / CHUNK_BF16;

    __nv_bfloat16* sdst_row = sDST + (size_t)col_half * KV_TILE * SUB_COLS_BF16 + kv_row * SUB_COLS_BF16;

    PhaseTracker<1> st_ph, dpt_ph, dv_ph, dk_ph;
    PhaseTracker<NUM_Q_STAGES> lse_ph, delta_ph;
    PhaseTracker<ITEM_STAGES> item_ph;
    uint32_t kvf_ph = 0;
    uint32_t egress_n = 0;        // egress_go phases this thread's column half has completed (count != 0 items)
    bool store_inflight = false;  // a dV or dK tile's store may still be reading this column half of sDST
    bool kv_row_valid = false;
    uint4 kmask;

    // This column half's store out of sDST has been read: sDST may be written again.
    auto wait_store = [&]() {
      if (store_inflight) {
        if (lane_group == 0 && elect_one_sync()) cp_async_bulk_wait_group_read<0>();
        if (col_half == 0) bar_sync<12>(128);
        else bar_sync<13>(128);
        store_inflight = false;
      }
    };

    // This thread's 64 values of the coarse branch's share of a dK or dV row, loaded before the waits that precede
    // the tile's store so that their latency hides behind them.
    auto load_coarse = [&](const float* coarse_rows, float4 (&cr4)[ST_HALF_COLS / 4]) {
      const float4* crow4 = reinterpret_cast<const float4*>(
          coarse_rows + ((size_t)it.head * a.num_blocks + it.kv_block) * HEAD_DIM + col_half * ST_HALF_COLS);
#pragma unroll
      for (int i = 0; i < ST_HALF_COLS / 4; ++i) cr4[i] = kv_row_valid ? __ldg(crow4 + i) : make_float4(0.f, 0.f, 0.f, 0.f);
    };

    // dK / dV rows: the accumulator (times sm_scale for dK; none when no query block kept the block) plus the coarse
    // share, as bf16; zero on padding rows. Read out of TMEM into registers before sDST is free to take them, so
    // that the accumulator is released (dv_free, dk_free) and the conversion done while the previous reader of
    // sDST finishes.
    constexpr int ROW_CHUNKS = ST_HALF_COLS / CHUNK_BF16;
    auto convert_dv_dk_rows = [&](uint32_t tmem_acc, auto apply_sm_scale_const, const float4 (&cr4)[ST_HALF_COLS / 4],
                                  bool have_acc, uint4 (&rows)[ROW_CHUNKS]) {
      constexpr bool apply_sm_scale = decltype(apply_sm_scale_const)::value;
      const float* crow = reinterpret_cast<const float*>(cr4);
#pragma unroll
      for (int c0 = 0; c0 < ST_HALF_COLS; c0 += 32) {
        uint32_t acc_regs[32] = {};
        if (have_acc) {
          tcgen05_ld_32x32b_x32(tmem_acc + tmem_f32_offset + (uint32_t)c0, acc_regs);
          tcgen05_wait_ld();
          tcgen05_fence_before_thread_sync();
        }
        const float* acc = reinterpret_cast<const float*>(acc_regs);
#pragma unroll
        for (int v = 0; v < 32 / CHUNK_BF16; ++v) {
          float value[CHUNK_BF16];
#pragma unroll
          for (int e = 0; e < CHUNK_BF16; ++e) {
            const float x = apply_sm_scale ? acc[v * CHUNK_BF16 + e] * a.sm_scale : acc[v * CHUNK_BF16 + e];
            value[e] = kv_row_valid ? x + crow[c0 + v * CHUNK_BF16 + e] : 0.f;
          }
          uint4& packed = rows[c0 / CHUNK_BF16 + v];
          packed.x = cvt_f32x2_to_bf16x2(value[0], value[1]);
          packed.y = cvt_f32x2_to_bf16x2(value[2], value[3]);
          packed.z = cvt_f32x2_to_bf16x2(value[4], value[5]);
          packed.w = cvt_f32x2_to_bf16x2(value[6], value[7]);
        }
      }
    };

    // Stages this column half of the rows in sDST, whose previous reader must be done, and stores it. Returns true in
    // the one thread that issued the store.
    auto stage_store_tile = [&](const uint4 (&rows)[ROW_CHUNKS], const CUtensorMap* map) -> bool {
      uint8_t* bounce_subtile = reinterpret_cast<uint8_t*>(sDST) + col_half * KV_SUB_COLS_BYTES;
      __nv_bfloat16* stage_row = reinterpret_cast<__nv_bfloat16*>(bounce_subtile) + kv_row * SUB_COLS_BF16;
#pragma unroll
      for (int chunk = 0; chunk < ROW_CHUNKS; ++chunk)
        *reinterpret_cast<uint4*>(stage_row + ((chunk ^ (kv_row & 7)) * CHUNK_BF16)) = rows[chunk];
      fence_proxy_async_shared_cta();
      if (col_half == 0) bar_sync<12>(128);
      else bar_sync<13>(128);
      bool issuer = false;
      if (lane_group == 0 && elect_one_sync()) {
        tma_store_4d(map, 0, it.kv_block * KV_TILE, col_half, it.head, smem_ptr_u32(bounce_subtile));
        cp_async_bulk_commit_group();
        issuer = true;
      }
      store_inflight = true;
      return issuer;
    };

    while (take_item(item_ph, kmask)) {
      const uint32_t mword = lane_group == 0 ? kmask.x : lane_group == 1 ? kmask.y : lane_group == 2 ? kmask.z : kmask.w;
      kv_row_valid = ((mword >> lane) & 1u) != 0;

      for (int j = 0; j < it.count; ++j) {
        const int lse_stage = lse_ph.get_stage();
        const float2* lse2 = reinterpret_cast<const float2*>(sLSE + lse_stage * Q_TILE + col_half * ST_HALF_COLS);
        // delta2 uses lse_stage: the loader's delta_empty_ph and this warp's delta_ph both advance once per query
        // block from zero, so block j's copy always lands in, and is read from, stage lse_stage. Should either ever
        // be advanced on a different condition, each must read its own tracker's stage instead.
        const float2* delta2 = reinterpret_cast<const float2*>(sDelta + lse_stage * Q_TILE + col_half * ST_HALF_COLS);

        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_lse[lse_stage]), lse_ph.get_phase());
        lse_ph.advance();

        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_st), st_ph.get_phase());
        st_ph.advance();

        uint32_t st_regs[ST_HALF_COLS];
        uint32_t pt_bf16x2[ST_HALF_COLS / 2];
        float2* pt_fp32 = reinterpret_cast<float2*>(st_regs);
        tcgen05_ld_32x32b_x64(tmem_st + tmem_f32_offset, st_regs);
        tcgen05_wait_ld();
        tcgen05_fence_before_thread_sync();
        const float2 scale2 = f32x2_splat(a.scale_log2);
#pragma unroll
        for (int c = 0; c < ST_HALF_COLS / 2; ++c) {
          const float2 z = ffma2(pt_fp32[c], scale2, make_float2(-lse2[c].x, -lse2[c].y));
          float2 p = make_float2(ex2_approx_f32(z.x), ex2_approx_f32(z.y));
          if (!kv_row_valid) p = make_float2(0.f, 0.f);
          pt_fp32[c] = p;
          pt_bf16x2[c] = cvt_f32x2_to_bf16x2(p.x, p.y);
        }
        tcgen05_st_32x32b_x32(tmem_pt_bf16 + tmem_bf16x2_offset, pt_bf16x2);
        tcgen05_wait_st();
        tcgen05_fence_before_thread_sync();
        if (elect_one_sync()) {
          mbarrier_arrive(smem_ptr_u32(full_bar_pt));
          mbarrier_arrive(smem_ptr_u32(&empty_bar_lse[lse_stage]));
        }

        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_delta[lse_stage]), delta_ph.get_phase());
        delta_ph.advance();

        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dpt), dpt_ph.get_phase());
        dpt_ph.advance();

        uint32_t dpt_regs[ST_HALF_COLS];
        tcgen05_ld_32x32b_x64(tmem_dpt + tmem_f32_offset, dpt_regs);
        tcgen05_wait_ld();
        tcgen05_fence_before_thread_sync();
        const float2* dpt2 = reinterpret_cast<const float2*>(dpt_regs);
#pragma unroll
        for (int c = 0; c < ST_HALF_COLS / 2; ++c) {
          const float2 ds = fmul2(pt_fp32[c], fadd2(dpt2[c], make_float2(-delta2[c].x, -delta2[c].y)));
          st_regs[c] = cvt_f32x2_to_bf16x2(ds.x, ds.y);
        }
        uint32_t(&dst_bf16x2)[ST_HALF_COLS / 2] = reinterpret_cast<uint32_t(&)[ST_HALF_COLS / 2]>(st_regs);
        tcgen05_st_32x32b_x32(tmem_dst_bf16 + tmem_bf16x2_offset, dst_bf16x2);
        if (j == 0) wait_store();  // the previous item's dK tile may still be leaving through sDST
        const uint4* dst_chunks = reinterpret_cast<const uint4*>(dst_bf16x2);
#pragma unroll
        for (int v = 0; v < CHUNKS_PER_ROW; ++v)
          *reinterpret_cast<uint4*>(sdst_row + (v ^ (kv_row & 7)) * CHUNK_BF16) = dst_chunks[v];
        tcgen05_wait_st();
        tcgen05_fence_before_thread_sync();
        fence_proxy_async_shared_cta();
        if (elect_one_sync()) {
          mbarrier_arrive(smem_ptr_u32(full_bar_dst));
          mbarrier_arrive(smem_ptr_u32(&empty_bar_delta[lse_stage]));
        }
      }

      float4 cr4[ST_HALF_COLS / 4];
      uint4 rows[ROW_CHUNKS];
      if (it.count != 0) {
        // dV(last) completed before the item's last dK and dQ products were issued: its rows are read and converted
        // while those run. dK's rows (its coarse share is one row per block, a broadcast load) are converted while
        // the dV store reads sDST.
        load_coarse(a.dv_coarse, cr4);
        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dv), dv_ph.get_phase());
        dv_ph.advance();
        convert_dv_dk_rows(tmem_dv, std::false_type{}, cr4, true, rows);
        if (elect_one_sync()) mbarrier_arrive(smem_ptr_u32(dv_free));
        mbarrier_wait_parity_suspend(smem_ptr_u32(kv_free), kvf_ph);  // the last dQ product has read sDST
        kvf_ph ^= 1;
        stage_store_tile(rows, &tmap_dv);
        load_coarse(a.dk_coarse, cr4);

        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dk), dk_ph.get_phase());
        dk_ph.advance();
        convert_dv_dk_rows(tmem_dk, std::true_type{}, cr4, true, rows);
        if (elect_one_sync()) mbarrier_arrive(smem_ptr_u32(dk_free));
        wait_store();  // the dV store has read sDST
        // Once both halves' dK stores are issued, the drain pushes the last dQ tile's reductions behind them. A
        // parity wait cannot tell phase k from phase k + 2, so egress_go must not complete a phase before the drain
        // leader has consumed the previous one: the issuer first waits for that phase's egress_ack. (Without it, a
        // drain leader delayed past two items' egress would wait for a phase that never comes.) The drain consumes
        // each phase about an item before the next is ready, so this wait is normally already satisfied.
        if (stage_store_tile(rows, &tmap_dk)) {
          if (egress_n != 0) mbarrier_wait_parity_suspend(smem_ptr_u32(egress_ack), (egress_n - 1u) & 1u);
          mbarrier_arrive(smem_ptr_u32(egress_go));
        }
        ++egress_n;
      } else {
        // No query block kept this key block (a selection without its diagonal; the forward never makes one): the
        // coarse share alone, so that every row is still written. sDST needs no kv_free wait here: its last product
        // reader was the dQ product of the last count != 0 item, whose completion this warp waited for (kv_free) in
        // that item's epilogue above, and wait_store covers the stores out of sDST since then.
        load_coarse(a.dv_coarse, cr4);
        convert_dv_dk_rows(tmem_dv, std::false_type{}, cr4, false, rows);
        wait_store();
        stage_store_tile(rows, &tmap_dv);
        load_coarse(a.dk_coarse, cr4);
        convert_dv_dk_rows(tmem_dk, std::true_type{}, cr4, false, rows);
        wait_store();
        stage_store_tile(rows, &tmap_dk);
      }
    }
    if (lane_group == 0 && elect_one_sync()) cp_async_bulk_wait_group_read<0>();
    bar_sync<10>(416);
    return;
  } else if (warp_id < W_SOFTMAX0) {
    setmaxnreg_inc<136>();

    bar_sync<10>(416);
    const uint32_t tmem_base = *tmem_slot;
    const uint32_t tmem_dq = tmem_base + ST_COLS + DV_COLS;

    const int epi_warp_id = warp_id - W_EPI0;
    const uint32_t tmem_lane_base = (uint32_t)(epi_warp_id * 32) << 16;
    const int q_row = epi_warp_id * 32 + lane;
    const bool is_leader = epi_warp_id == 0 && lane == 0;

    PhaseTracker<1> dq_full_ph;
    PhaseTracker<ITEM_STAGES> item_ph;
    uint32_t egress_ph = 0;
    uint4 mask_unused;

    while (take_item(item_ph, mask_unused)) {
      float* dqaccum_head = a.dqaccum + (size_t)it.head * seqlen * HEAD_DIM;
      for (int j = 0; j < it.count; ++j) {
        float* dqaccum_block = dqaccum_head + (size_t)it.list[j] * DQ_BLOCK_ELEMS;
        const bool last_of_item = j == it.count - 1;

        mbarrier_wait_parity(smem_ptr_u32(full_bar_dq), dq_full_ph.get_phase());
        dq_full_ph.advance();

        uint32_t dq_regs[HEAD_DIM];
#pragma unroll
        for (int c = 0; c < HEAD_DIM / 64; ++c)
          tcgen05_ld_32x32b_x64(tmem_dq + tmem_lane_base + (uint32_t)(c * 64),
                                reinterpret_cast<uint32_t(&)[64]>(dq_regs[c * 64]));
        tcgen05_wait_ld();
        tcgen05_fence_before_thread_sync();
        // dQ's TMEM columns are free once read: the MMA's next dP (on an item's last tile, the NEXT item's dP(0), the
        // `if (prev)` gate) starts now, before the leader's egress_go wait below. A count-1 next item can then finish
        // its products and both egress_go arrivals while the leader still waits for this item's phase, so egress_go
        // could complete two phases and alias the parity (D75; witnessed by J6286). egress_ack bounds that at the
        // dK store's arrival instead of here (deferring this release past the wait cost 0.4 %, J6215).
        if (elect_one_sync()) mbarrier_arrive(smem_ptr_u32(empty_bar_dq));

#pragma unroll
        for (int hd_slice = 0; hd_slice < HEAD_DIM / DQ_COLS; ++hd_slice) {
          const int stage_buf = hd_slice & 1;
          const float4* dq_row4 = reinterpret_cast<const float4*>(dq_regs + hd_slice * DQ_COLS);
#pragma unroll
          for (int v4 = 0; v4 < DQ_COLS / 4; ++v4)
            *reinterpret_cast<float4*>(sDQ_STAGE[stage_buf] + v4 * Q_TILE * 4 + q_row * 4) = dq_row4[v4];
          fence_proxy_async_shared_cta();
          bar_sync<11>(128);
          if (is_leader) {
            if (hd_slice == 0 && last_of_item) {
              // The item's dV and dK stores leave first: the next item's first dS tile waits for sDST, i.e. for the
              // dK store, and nothing waits for this tile's reductions.
              mbarrier_wait_parity_suspend(smem_ptr_u32(egress_go), egress_ph);
              egress_ph ^= 1;
              mbarrier_arrive(smem_ptr_u32(egress_ack));  // the phase is consumed: the next one may complete
            }
            const size_t slice_offset = (size_t)hd_slice * Q_TILE * DQ_COLS;
            cpasync_reduce_bulk_add_f32(dqaccum_block + slice_offset, smem_ptr_u32(sDQ_STAGE[stage_buf]),
                                        DQ_ONE_PUSH_BYTES);
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<1>();
          }
          bar_sync<11>(128);
        }
      }
    }

    if (is_leader) cp_async_bulk_wait_group_read<0>();
    bar_sync<11>(128);
    bar_sync<10>(416);
    // The reductions must be complete (not only read) before the CTA exits.
    if (is_leader) asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
    return;
  } else if (warp_id == W_SCHED) {
    // Publishes this CTA's items in order: its own block's first, then, each time the load warp asks (CLAIM_LEAD
    // query blocks before the end of its list), the block of the next CTA that cluster launch control cancels. A
    // failed claim publishes the end of the list.
    // setmaxnreg must be executed with one value by every warp of its warpgroup (PTX ISA, setmaxnreg: "the same
    // setmaxnreg instruction must be executed by all warps in a warpgroup"): 56 is the value warps 12 and 13 take.
    setmaxnreg_dec<56>();
    EmptyPhaseTracker<ITEM_STAGES> item_empty_ph;
    uint32_t claim_ph = 0, clc_ph = 0;
    int x = (int)blockIdx.x;
    while (true) {
      const int s = item_empty_ph.get_stage();
      mbarrier_wait_parity_suspend(smem_ptr_u32(&item_empty[s]), item_empty_ph.get_phase());
      item_empty_ph.advance();
      if (elect_one_sync()) {
        int4 d = make_int4(0, 0, 0, 0);
        uint4 m = make_uint4(0u, 0u, 0u, 0u);
        if (x >= 0) {
          const WorkItem w = decode_workitem(a, x);
          d = make_int4(1, w.head, w.kv_block, w.count);
          m = __ldg(reinterpret_cast<const uint4*>(a.rowmask) + w.kv_block);
        }
        *reinterpret_cast<int4*>(&item_desc[s]) = d;
        *reinterpret_cast<uint4*>(&item_desc[s].mask[0]) = m;
        mbarrier_arrive(smem_ptr_u32(&item_full[s]));
      }
      __syncwarp();
      if (x < 0) break;
      mbarrier_wait_parity_suspend(smem_ptr_u32(claim_bar), claim_ph);
      claim_ph ^= 1;
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(smem_ptr_u32(clc_full), 16);
        clc_try_cancel_async(smem_ptr_u32(clc_response), smem_ptr_u32(clc_full));
      }
      __syncwarp();
      mbarrier_wait_parity_suspend(smem_ptr_u32(clc_full), clc_ph);
      clc_ph ^= 1;
      x = clc_cancelled_block(smem_ptr_u32(clc_response));
      // The next request's response overwrites this one through the async proxy.
      fence_proxy_async_shared_cta();
    }
    return;
  } else {
    // Warp 15 does no work, but it sits in warpgroup 3 with the MMA (12), load (13) and scheduler (14) warps, and
    // setmaxnreg must be executed with one value by every warp of its warpgroup (see above). 56 is the value the other
    // three take; with it the four increases of this CTA (4x8 + 8x32) equal its four decreases (4x72) and the pool
    // closes exactly at its 2,048 slots.
    setmaxnreg_dec<56>();
    return;
  }
#else
  __trap();  // built for an architecture without tcgen05 (primitives.cuh)
#endif
}

}  // namespace bwd
}  // namespace vsa2t
