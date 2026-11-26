// router.cuh -- the router's scores (eq. 5-7): each video query window's softmax over every key window of the video,
// pooled to tile pairs, on tcgen05 from the fp16 window means (prep.cuh).
//
// One sweep of the logits in the common case (MODE 2): a row's logits z (log2 units) are bounded by
// B = |q_w| max|k_w| * scale * log2(e) (Cauchy-Schwarz), so p = exp2(z - B) <= 1 needs no max pass. The sweep keeps,
// per query window and video key tile, u = sum over the tile's 16 windows of p (in global memory, [key tile][window]
// so that a warp's 32 rows store 128 contiguous bytes), and l = sum of all of them; then each CTA pools its rows:
// score[I][J] = sum over the windows of I of u / (256 l). A row whose bound is so loose that l < 2^-60 (its largest
// terms near fp32's underflow) flags its query tile, which the paper's two passes then redo (MODE 0: each row's lse;
// MODE 1: p = exp2(z - lse) pooled directly); for every other tile those two kernels return at once.
// A CTA owns one 128-window query tile (8 video tiles) of one head and sweeps every 128-window key tile; S = Qw Kw^T
// goes to TMEM, double-buffered, and 16 softmax warps read it: warp & 3 is the TMEM lane quarter (32 query windows,
// two video tiles), (warp - 4) / 4 the column part (32 key windows, two video tiles). A key window without a token is
// no key (its column is masked); a query window without one contributes nothing (its row is zero).
// Warps: 0 TMA, 1 MMA, 2-3 idle, 4-19 softmax.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "common.cuh"
#include "primitives.cuh"

namespace vsa2t {
namespace router {

constexpr int kStages = 3;                                 // key tiles in flight
constexpr int kTileBytes = kRows * kDim * 2;               // a 128 x 128 tile of the router's 16-bit operands, two 64-column subtiles
constexpr int kSubBytes = kRows * 64 * 2;                  // one subtile
// An S tile's columns in kParts parts, a softmax warp per (TMEM lane quarter, part): 32 key windows (two video
// tiles) a warp, so that a thread's S values leave registers to pipeline the exponentials' latency.
constexpr int kParts = 4, kCols = kRows / kParts, kTPP = kCols / kWindows;
constexpr int kSoftmaxWarps = 4 * kParts;
constexpr int kWarps = 4 + kSoftmaxWarps, W_LOAD = 0, W_MMA = 1, W_SOFTMAX = 4;
constexpr int kTmemCols = 256;                             // S, two buffers of 128 columns
constexpr int kBars = 1 + 2 * kStages + 4;
constexpr size_t kSmemBytes = 1024 + (size_t)(1 + kStages) * kTileBytes + kBars * 8 + 16 + 2 * kParts * kRows * 4;

struct Args {
  Geo g;
  const uint16_t* winmask;  // [Nv] a bit per window of each video tile
  const float* qnorm;       // [heads][Nv * 16]
  const float* knorm;       // [heads][Nv * 16]
  float* u;                 // [heads][rtiles * 8][upitch]: per video key tile and query window, the sum of p (MODE 2)
  long long upitch;         // windows rounded up to 32
  int* flags;               // [heads][rtiles]: 1 where MODE 2 left the query tile to the two passes
  float* rstat;             // [heads][Nv * 16]: each query window's lse in log2 units (+inf without a token)
  float* scores;            // [heads][Nv][Nv]
  float* lpart;             // [ksplit][heads][Nv * kWindows]: each key-range chunk's row sum of p (MODE 2)
  int ksplit;               // key-range chunks per (query tile, head); router_pool_kernel merges them
  float scale_log2;         // softmax scale * log2(e)
};

// The fill model.  A CTA pays a prologue -- Q's TMA, the pipeline fill, the first TMEM alloc -- before its
// sweep begins, measured on this kernel at a = 12,366 cycles against 1,477 per key tile, i.e. about
// kPrologueTiles key tiles (J7006 s8 / J7037 s20: T = a + 1,766 * nkt before the pool left the sweep).
// The measured value is 8.4; the model uses 10 because `a` is a FLOOR -- it excludes the fixed per-CTA tail
// (the parts' merge, the row sum's store, the TMEM dealloc) that a split also multiplies by k.
// Splitting the key range in k turns the sweep into more, better filled waves, so k pays only
// when the wave count it buys is worth the prologue it pays k times over.  Nothing here is tuned to a
// workload: it reads the grid and nkt, and k = 1 whenever no fill is left to buy.
constexpr int kPrologueTiles = 10;
constexpr int kPoolSlices = 1;
constexpr int kPoolThreads = 320, kPoolCtas = 2;  // 320 thr x 2 CTAs/SM <= 102 regs (it uses 96) -> one wave

inline int ksplit_for(int rtiles, int heads, int sms) {
  const long long g = (long long)rtiles * heads;
  if (rtiles < 2 || g >= 4LL * sms) return 1;  // one tile per chunk at the least, and nothing to buy beyond
  int best = 1;
  double bestc = 1e30;
  for (int k = 1; k <= 4 && k <= rtiles; ++k) {
    const long long waves = (g * k + sms - 1) / sms;
    const double c = (double)waves * ((double)kPrologueTiles + (double)rtiles / (double)k);
    if (c < bestc - 1e-9) {
      bestc = c;
      best = k;
    }
  }
  return best;
}

// MEASURED AND NOT SHIPPED: a per-head power-of-two pre-scale of the stored means (a max over the pooled
// means per head and tensor, then an in-place shrink so the largest sits below 2^13).  Its premise was that
// fp16's 65,504 ceiling is the cause of the ``--qk-scale`` failures, and the measurement says it is not: at
// the marked scales where `selection` and `lse` break, the stored means are ~2.55 * s, which is 41,779 at
// s = 16,384 -- BELOW the ceiling -- so the kernel computes a = 0, changes nothing, and the failures are
// identical with and without it.  The class it was written for is extreme-sharpness selection instability
// (a precision change of ANY kind moves the kept set where ties are dense), which a power-of-two scale
// leaves invariant by construction.  Kept here as a note so the next reader does not rebuild it: the range
// argument holds only above s ~ 25,688 measured on that ladder, and the failures arrive two binades earlier.


// 2^-60: a row of the one-pass sweep whose sum is below this redoes its tile in two passes.
constexpr float kMinSum = 8.673617379884035e-19f;

template <int MODE>
__global__ void __launch_bounds__(kWarps * 32, 1)
    router_kernel(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_k, Args a) {
#if VSA2T_TCGEN05
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  extern __shared__ uint8_t router_smem_raw[];
  uint8_t* base = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(router_smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* sQ = base;
  uint8_t* sK = base + kTileBytes;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sK + kStages * kTileBytes);
  uint64_t* full_q = bars;
  uint64_t* full_k = bars + 1;
  uint64_t* empty_k = full_k + kStages;
  uint64_t* s_full = empty_k + kStages;
  uint64_t* s_empty = s_full + 2;
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(bars + kBars);
  float* xm = reinterpret_cast<float*>(tmem_slot + 4);  // [kParts][128] each part's max, for the merge
  float* xl = xm + kParts * kRows;                       // [kParts][128] and its sum

  const Geo& g = a.g;
  const int ksplit = a.ksplit > 0 ? a.ksplit : 1;
  const int qt = blockIdx.x / ksplit, hd = blockIdx.y;
  const int ch = blockIdx.x - qt * ksplit;  // this CTA's slice of the key range
  const int nkt = g.rtiles;
  const int nkt_lo = (int)((long long)nkt * ch / ksplit), nkt_hi = (int)((long long)nkt * (ch + 1) / ksplit);
  const int nkt_c = nkt_hi - nkt_lo;  // the sweep's key tiles; MODE 0/1 always get the whole tile
  const long long windows = (long long)g.Nv * kWindows;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // The two-pass kernels redo only the tiles the sweep flagged.
  if (MODE != 2 && a.flags[(long long)hd * nkt + qt] == 0) return;
  __shared__ float kmax_part[kWarps];
  float kmax = 0.f;
  if (MODE == 2) {
    // The largest key norm of the head, for every row's bound.
    for (long long i = threadIdx.x; i < windows; i += blockDim.x) kmax = fmaxf(kmax, a.knorm[(long long)hd * windows + i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) kmax = fmaxf(kmax, __shfl_xor_sync(0xffffffffu, kmax, o));
    if (lane == 0) kmax_part[warp] = kmax;
  }

  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(full_q), 1);
    for (int s = 0; s < kStages; ++s) {
      mbarrier_init(smem_ptr_u32(&full_k[s]), 1);
      mbarrier_init(smem_ptr_u32(&empty_k[s]), 1);
    }
    for (int b = 0; b < 2; ++b) {
      mbarrier_init(smem_ptr_u32(&s_full[b]), 1);
      mbarrier_init(smem_ptr_u32(&s_empty[b]), kSoftmaxWarps);
    }
  }
  if (warp == W_MMA) {
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), kTmemCols);
    tcgen05_relinquish_alloc_permit<1>();
  }
  fence_mbarrier_init_release_cluster();
  tcgen05_fence_before_thread_sync();
  __syncthreads();
  tcgen05_fence_after_thread_sync();
  const uint32_t tmem_base = *tmem_slot;
  if (MODE == 2) {
#pragma unroll
    for (int i = 0; i < kWarps; ++i) kmax = fmaxf(kmax, kmax_part[i]);
  }

  if (warp == W_LOAD) {
    if (elect_one_sync()) {
      mbarrier_arrive_expect_tx(smem_ptr_u32(full_q), kTileBytes);
      tma_load_4d(smem_ptr_u32(sQ), &tmap_q, smem_ptr_u32(full_q), 0, qt * kRows, 0, hd);
      for (int i = 0; i < nkt_c; ++i) {
        const int kt = nkt_lo + i;
        const int st = i % kStages;
        mbarrier_wait_parity(smem_ptr_u32(&empty_k[st]), ((i / kStages) & 1) ^ 1);
        mbarrier_arrive_expect_tx(smem_ptr_u32(&full_k[st]), kTileBytes);
        tma_load_4d(smem_ptr_u32(sK + st * kTileBytes), &tmap_k, smem_ptr_u32(&full_k[st]), 0, kt * kRows, 0, hd);
      }
    }
  } else if (warp == W_MMA) {
    const uint32_t lead = elect_one_sync();
    const uint32_t idesc = make_idesc_f16_f32(kRows, kRows, false, false);
    const uint64_t desc_q = build_smem_desc_blackwell(smem_ptr_u32(sQ), 1024, 16, SmemSwizzleBlackwell::B128);
    const uint64_t desc_k = build_smem_desc_blackwell(smem_ptr_u32(sK), 1024, 16, SmemSwizzleBlackwell::B128);
    mbarrier_wait_parity(smem_ptr_u32(full_q), 0);
    for (int i = 0; i < nkt_c; ++i) {
      const int st = i % kStages, buf = i & 1;
      mbarrier_wait_parity(smem_ptr_u32(&full_k[st]), (i / kStages) & 1);
      mbarrier_wait_parity(smem_ptr_u32(&s_empty[buf]), ((i >> 1) & 1) ^ 1);
      tcgen05_fence_after_thread_sync();
      const uint32_t s_tmem = tmem_base + (uint32_t)(buf * kRows);
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const uint64_t da = desc_q + (uint64_t)((s * kSubBytes) >> 4);
        const uint64_t db = desc_k + (uint64_t)((st * kTileBytes + s * kSubBytes) >> 4);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
          tcgen05_mma_f16_ss_lead(lead, s_tmem, da + 2 * ki, db + 2 * ki, idesc, (s | ki) != 0);
      }
      tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_k[st]));
      tcgen05_commit1_lead(lead, smem_ptr_u32(&s_full[buf]));
    }
    if (lead) {
      // Producer tail: this CTA must stay alive until the issuer's last tcgen05.commit on each used
      // empty_k slot has landed -- an unconsumed arrive-on may otherwise hit released shared memory.
      // The same shape is required by flashinfer (gemm/kernels/tgv_gemm_cute_ext.py drains bar_empty
      // for this reason, in the presence of a second consumed barrier on the same MMA). Unused slots
      // take no commit; a used slot's loader consumes every generation but the last.
      const int used = nkt_c < kStages ? nkt_c : kStages;
      for (int st = 0; st < used; ++st)
        mbarrier_wait_parity(smem_ptr_u32(&empty_k[st]), ((nkt_c - 1 - st) / kStages) & 1);
    }
  } else if (warp >= W_SOFTMAX) {
    const int sw = warp - W_SOFTMAX, quarter = sw & 3, part = sw >> 2;
    const int row = quarter * 32 + lane;
    const long long qrow = (long long)qt * kRows + row;  // the query window
    const bool row_ok = (win_word(a.winmask, g.Nv, (long long)qt * 8 + (row >> 4)) >> (row & 15)) & 1u;
    const float c = a.scale_log2;
    const uint32_t lane_addr = tmem_base + ((uint32_t)(quarter * 32) << 16) + (uint32_t)(part * kCols);
    constexpr int kSync = kSoftmaxWarps * 32;
    // MODE 1: exp2(z - lse2) = 0 on a row without a token. MODE 2: the row's bound (one more than Cauchy-Schwarz's,
    // for the products' rounding), its sum, and the rows' u in [key tile][window] order.
    float lse2 = INFINITY;
    if (MODE == 1 && row_ok) lse2 = a.rstat[(long long)hd * windows + qrow];
    if (MODE == 2) lse2 = row_ok ? fmaf(a.qnorm[(long long)hd * windows + qrow] * kmax, c, 1.f) : INFINITY;
    float* urow = a.u + (long long)hd * nkt * 8 * a.upitch + qrow;
    float m = -INFINITY, l = 0.f;  // MODE 0: the running max and sum; MODE 2: the sum
    const int qtile = qt * 8 + quarter * 2 + (lane >> 4);  // lanes 0-15 and 16-31: two video query tiles

    for (int i = 0; i < nkt_c; ++i) {
      const int kt = nkt_lo + i;
      const int buf = i & 1;
      mbarrier_wait_parity(smem_ptr_u32(&s_full[buf]), (i >> 1) & 1);
      tcgen05_fence_after_thread_sync();
      uint32_t sr[kCols];
      tcgen05_ld_32x32b_x32(lane_addr + (uint32_t)(buf * kRows), *reinterpret_cast<uint32_t(*)[32]>(&sr[0]));
      tcgen05_wait_ld();
      tcgen05_fence_before_thread_sync();
      __syncwarp();
      if (lane == 0) mbarrier_arrive(smem_ptr_u32(&s_empty[buf]));

      const uint32_t wmask = win_word(a.winmask, g.Nv, (long long)kt * 8 + part * kTPP);
      float* s = reinterpret_cast<float*>(sr);
      const bool full = wmask == 0xffffffffu;
      if (MODE == 0) {
        float tmax = -INFINITY;
#pragma unroll
        for (int j = 0; j < kCols; ++j) {
          s[j] = (full || ((wmask >> j) & 1u)) ? s[j] * c : -INFINITY;
          tmax = fmaxf(tmax, s[j]);
        }
        if (tmax > m) {
          l *= ex2_approx_f32(m - tmax);
          m = tmax;
        }
        if (m > -INFINITY) {
          float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
          for (int j = 0; j < kCols; j += 2) {
            acc0 += ex2_approx_f32(s[j] - m);
            acc1 += ex2_approx_f32(s[j + 1] - m);
          }
          l += acc0 + acc1;
        }
      } else {
        // MODE 1 and 2: p = exp2(z - lse2) summed per video key tile (16 windows).
        if (!full) {
#pragma unroll
          for (int j = 0; j < kCols; ++j) s[j] = ((wmask >> j) & 1u) ? s[j] : -INFINITY;
        }
        float acc[kTPP];
#pragma unroll
        for (int q = 0; q < kTPP; ++q) {
          float t = 0.f, t2 = 0.f;
#pragma unroll
          for (int j = q * 16; j < q * 16 + 16; j += 2) {
            t += ex2_approx_f32(fmaf(s[j], c, -lse2));
            t2 += ex2_approx_f32(fmaf(s[j + 1], c, -lse2));
          }
          acc[q] = t + t2;
        }
        if (MODE == 2) {
#pragma unroll
          for (int q = 0; q < kTPP; ++q) {
            l += acc[q];
            if (qrow < windows) urow[(long long)(kt * 8 + part * kTPP + q) * a.upitch] = acc[q];
          }
        } else {
          // The sum over the 16 query windows of each video tile: lanes 0-15 and 16-31.
#pragma unroll
          for (int o = 8; o > 0; o >>= 1)
#pragma unroll
            for (int q = 0; q < kTPP; ++q) acc[q] += __shfl_xor_sync(0xffffffffu, acc[q], o);
          if ((lane & 15) == 0 && qtile < g.Nv) {
            float* dst = a.scores + ((long long)hd * g.Nv + qtile) * g.Nv;
#pragma unroll
            for (int q = 0; q < kTPP; ++q) {
              const int ktile = kt * 8 + part * kTPP + q;
              if (ktile < g.Nv) dst[ktile] = acc[q] * (1.f / 256.f);
            }
          }
        }
      }
    }

    if (MODE == 0) {
      // Merge the parts of each row: part 0 takes the others'.
      xm[part * kRows + row] = m;
      xl[part * kRows + row] = l;
      bar_sync<1>(kSync);
      if (part == 0 && qrow < windows) {
        float mm = -INFINITY;
#pragma unroll
        for (int pp = 0; pp < kParts; ++pp) mm = fmaxf(mm, xm[pp * kRows + row]);
        float out = INFINITY;
        if (row_ok && mm > -INFINITY) {
          float lt = 0.f;
#pragma unroll
          for (int pp = 0; pp < kParts; ++pp) {
            const float mp = xm[pp * kRows + row];
            if (mp > -INFINITY) lt += xl[pp * kRows + row] * ex2_approx_f32(mp - mm);
          }
          out = mm + lg2_approx_f32(lt);
        }
        a.rstat[(long long)hd * windows + qrow] = out;
      }
    } else if (MODE == 2) {
      // The row's sum over the parts (one reference), then the tile's verdict.
      xl[part * kRows + row] = l;
      bar_sync<1>(kSync);
      float total = 0.f;
#pragma unroll
      for (int pp = 0; pp < kParts; ++pp) total += xl[pp * kRows + row];
      if (part == 0 && qrow < windows)
        a.lpart[((long long)ch * g.heads + hd) * windows + qrow] = row_ok ? total : 0.f;  // this chunk's l
    }
  }

  tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (warp == W_MMA) {
    tcgen05_fence_after_thread_sync();
    tcgen05_dealloc<1>(tmem_base, kTmemCols);
  }
#else
  __trap();  // built for an architecture without tcgen05 (primitives.cuh)
#endif
}

// The router's pool, over every chunk's row sum: it decides the query tiles the one-pass sweep could not
// resolve and turns u into score[I][J] = (sum over I's 16 windows of u) / (256 l).  It is a kernel of its
// own for two reasons.  A CTA that sweeps a slice of the key range sees only that slice's l, so the merge
// has to happen after the sweep; and the pool is then 640 threads over 8 columns each instead of 128
// columns of a serial epilogue in the sweep's last warp (J7051: the pool was 20.8 % of that stage).
__global__ void __launch_bounds__(kPoolThreads, kPoolCtas)
    router_pool_kernel(Args a) {
#if VSA2T_TCGEN05
  griddepcontrol_wait();
  const Geo& g = a.g;
  const int qt = blockIdx.x, hd = blockIdx.y;
  const int nkt = g.rtiles;
  const int ksplit = a.ksplit > 0 ? a.ksplit : 1;
  const long long windows = (long long)g.Nv * kWindows;
  const long long qbase = (long long)qt * kRows;
  const int tid = (int)threadIdx.x;
  __shared__ float iv[kRows];  // the row's 1/(256 l); zero on a row without a token
  __shared__ int any_bad;
  if (tid == 0) any_bad = 0;
  __syncthreads();
  for (int row = tid; row < kRows; row += (int)blockDim.x) {
    const bool ok = (win_word(a.winmask, g.Nv, (long long)qt * 8 + (row >> 4)) >> (row & 15)) & 1u;
    const long long qr = qbase + row;
    float t = 0.f;
    if (ok && qr < windows) {
      for (int c = 0; c < ksplit; ++c) t += a.lpart[((long long)c * g.heads + hd) * windows + qr];
      if (!(t >= kMinSum)) atomicOr(&any_bad, 1);
    }
    iv[row] = (ok && qr < windows && t >= kMinSum) ? 1.f / (256.f * t) : 0.f;
  }
  __syncthreads();
  if (tid == 0 && blockIdx.z == 0) a.flags[(long long)hd * nkt + qt] = any_bad ? 1 : 0;
  // One thread per (video query tile, video key tile): a tile's 16 query windows are contiguous in u, so
  // the sum is four 128-bit loads and sixteen multiplies, with no cross-lane traffic and no re-read.
  const int ntile = nkt * 8;
  const int qtm = tid & 7, lane8 = tid >> 3;
  const int crow = qtm * 16;
  const long long grow = qbase + crow;
  const bool vok = ((a.upitch & 3) == 0) && (grow + 16 <= windows);
  const long long ubase = (long long)hd * nkt * 8 * a.upitch;
  const int qtil = qt * 8 + qtm;
  // The key tiles are cut across blockIdx.z: the per-thread loop is short (five tiles at s8), so one CTA's
  // 20 warps do not keep enough loads in flight to cover the L2 round trip, and 28.8 us for a 40 MB stream
  // said so.  The row scales are recomputed per slice, which is 128 floats.
  // The loop is short (five tiles a thread at s8) and its store fences the next iteration's loads, so a
  // thread had four loads in flight and the CTA ran at 14 GB/s: 28.8 us for a 40 MB stream that should be
  // an L2 read.  Four key tiles are now loaded up front, sixteen loads in flight before any fma.
  const int kz = (int)blockIdx.z, nkz = (int)gridDim.z;
  const int kt_lo = (int)((long long)ntile * kz / nkz), kt_hi = (int)((long long)ntile * (kz + 1) / nkz);
  const int kstride = kPoolThreads / 8;
  if (qtil < g.Nv) {
    const float* __restrict__ uu = a.u;
    float* __restrict__ sc = a.scores;
    float* dbase = sc + ((long long)hd * g.Nv + qtil) * g.Nv;
    int ktile = kt_lo + lane8;
    const int kend = kt_hi < g.Nv ? kt_hi : g.Nv;
    if (vok) {
      for (; ktile + 3 * kstride < kend; ktile += 4 * kstride) {
        const float* p0 = uu + ubase + (long long)ktile * a.upitch + grow;
        const float* p1 = p0 + (long long)kstride * a.upitch;
        const float* p2 = p1 + (long long)kstride * a.upitch;
        const float* p3 = p2 + (long long)kstride * a.upitch;
        const float4 x0 = *reinterpret_cast<const float4*>(p0), x1 = *reinterpret_cast<const float4*>(p0 + 4);
        const float4 x2 = *reinterpret_cast<const float4*>(p0 + 8), x3 = *reinterpret_cast<const float4*>(p0 + 12);
        const float4 y0 = *reinterpret_cast<const float4*>(p1), y1 = *reinterpret_cast<const float4*>(p1 + 4);
        const float4 y2 = *reinterpret_cast<const float4*>(p1 + 8), y3 = *reinterpret_cast<const float4*>(p1 + 12);
        const float4 z0 = *reinterpret_cast<const float4*>(p2), z1 = *reinterpret_cast<const float4*>(p2 + 4);
        const float4 z2 = *reinterpret_cast<const float4*>(p2 + 8), z3 = *reinterpret_cast<const float4*>(p2 + 12);
        const float4 w0 = *reinterpret_cast<const float4*>(p3), w1 = *reinterpret_cast<const float4*>(p3 + 4);
        const float4 w2 = *reinterpret_cast<const float4*>(p3 + 8), w3 = *reinterpret_cast<const float4*>(p3 + 12);
        const float i0 = iv[crow], i1 = iv[crow + 1], i2 = iv[crow + 2], i3 = iv[crow + 3];
        const float i4 = iv[crow + 4], i5 = iv[crow + 5], i6 = iv[crow + 6], i7 = iv[crow + 7];
        const float i8 = iv[crow + 8], i9 = iv[crow + 9], i10 = iv[crow + 10], i11 = iv[crow + 11];
        const float i12 = iv[crow + 12], i13 = iv[crow + 13], i14 = iv[crow + 14], i15 = iv[crow + 15];
        dbase[ktile] = ((x0.x * i0 + x0.y * i1 + x0.z * i2 + x0.w * i3) + (x1.x * i4 + x1.y * i5 + x1.z * i6 + x1.w * i7)) +
                        ((x2.x * i8 + x2.y * i9 + x2.z * i10 + x2.w * i11) + (x3.x * i12 + x3.y * i13 + x3.z * i14 + x3.w * i15));
        dbase[ktile + kstride] =
            ((y0.x * i0 + y0.y * i1 + y0.z * i2 + y0.w * i3) + (y1.x * i4 + y1.y * i5 + y1.z * i6 + y1.w * i7)) +
            ((y2.x * i8 + y2.y * i9 + y2.z * i10 + y2.w * i11) + (y3.x * i12 + y3.y * i13 + y3.z * i14 + y3.w * i15));
        dbase[ktile + 2 * kstride] =
            ((z0.x * i0 + z0.y * i1 + z0.z * i2 + z0.w * i3) + (z1.x * i4 + z1.y * i5 + z1.z * i6 + z1.w * i7)) +
            ((z2.x * i8 + z2.y * i9 + z2.z * i10 + z2.w * i11) + (z3.x * i12 + z3.y * i13 + z3.z * i14 + z3.w * i15));
        dbase[ktile + 3 * kstride] =
            ((w0.x * i0 + w0.y * i1 + w0.z * i2 + w0.w * i3) + (w1.x * i4 + w1.y * i5 + w1.z * i6 + w1.w * i7)) +
            ((w2.x * i8 + w2.y * i9 + w2.z * i10 + w2.w * i11) + (w3.x * i12 + w3.y * i13 + w3.z * i14 + w3.w * i15));
      }
    }
    for (; ktile < kend; ktile += kstride) {
      const float* src = uu + ubase + (long long)ktile * a.upitch + grow;
      float s;
      if (vok) {
        const float4 a0 = *reinterpret_cast<const float4*>(src);
        const float4 a1 = *reinterpret_cast<const float4*>(src + 4);
        const float4 a2 = *reinterpret_cast<const float4*>(src + 8);
        const float4 a3 = *reinterpret_cast<const float4*>(src + 12);
        const float p0 = a0.x * iv[crow] + a0.y * iv[crow + 1] + a0.z * iv[crow + 2] + a0.w * iv[crow + 3];
        const float p1 = a1.x * iv[crow + 4] + a1.y * iv[crow + 5] + a1.z * iv[crow + 6] + a1.w * iv[crow + 7];
        const float p2 = a2.x * iv[crow + 8] + a2.y * iv[crow + 9] + a2.z * iv[crow + 10] + a2.w * iv[crow + 11];
        const float p3 = a3.x * iv[crow + 12] + a3.y * iv[crow + 13] + a3.z * iv[crow + 14] + a3.w * iv[crow + 15];
        s = (p0 + p1) + (p2 + p3);
      } else {
        s = 0.f;
        for (int j = 0; j < 16 && grow + j < windows; ++j) s = fmaf(src[j], iv[crow + j], s);
      }
      dbase[ktile] = s;
    }
  }
#else
  __trap();  // built for an architecture without tcgen05 (primitives.cuh)
#endif
}

}  // namespace router
}  // namespace vsa2t
