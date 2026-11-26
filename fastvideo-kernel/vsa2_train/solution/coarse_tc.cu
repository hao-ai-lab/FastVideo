// coarse_tc.cu -- the definitions of coarse_tc.cuh's kernels, in a translation unit of their own (glue.cuh says why a
// stage's kernels stay out of vsa2t.cu's).
#include "coarse_tc.cuh"

#include <cmath>

#include "primitives.cuh"

namespace vsa2t {
namespace coarse_tc {
namespace {

// The eight component products (i, j) with i + j <= 3, smallest first: (1,2) (2,1) (2,0) (0,2) (1,1) (1,0) (0,1) (0,0).
__device__ constexpr int kPa[kProducts] = {1, 2, 2, 0, 1, 1, 0, 0};
__device__ constexpr int kPb[kProducts] = {2, 1, 0, 2, 1, 0, 1, 0};

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ float ex2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
// A cluster-wide barrier with release / acquire at cluster scope: what a CTA wrote before it (global memory
// included) is visible to every CTA of the cluster after it.
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
// A bulk copy (the TMA engine), global -> this CTA's shared memory, completing on an mbarrier.
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}

// Rows [r0, r0 + nrows) of a head's six planes (prep.cuh SplitOut: already in the tile's 128-byte-swizzled form, r0
// a multiple of 8) into a tile of `ra` rows per plane: six bulk copies on `bar`, of 768 nrows bytes in all.
__device__ __forceinline__ void load_tile(uint32_t tile, int ra, const __nv_bfloat16* planes, int rpad, int r0,
                                          int nrows, uint32_t bar) {
#pragma unroll
  for (int p = 0; p < kPlanes; ++p)
    bulk_g2s(tile + (uint32_t)(p * ra * 128), planes + ((size_t)p * rpad + r0) * 64, (uint32_t)nrows * 128u, bar);
}

__device__ __forceinline__ uint64_t kmajor_desc(uint32_t addr) {
  return build_smem_desc_blackwell(addr, 1024, 16, SmemSwizzleBlackwell::B128);
}
// Step kd (16 dims) of a K-major tile of `rows` rows per plane: plane kd / 4 of the component, 32 bytes per step.
__device__ __forceinline__ uint64_t kstep(int kd, int rows) {
  return (uint64_t)(((kd >> 2) * rows * 128) >> 4) + 2u * (uint64_t)(kd & 3);
}

// Shared memory past the tiles (kMiscBytes), in floats.
constexpr int kMsKs = 0;       // [128] the pass's K row factors 2^-lift (0 past its keys)
constexpr int kMsKb = 128;     // [128] 0, or -inf past the pass's keys
constexpr int kMsVf = 256;     // [128] P's column fold: the V row's 2^-lift (0 past the keys)
constexpr int kMsMax = 512;    // [2][128] the two column halves' row maxima
constexpr int kMsL = 1024;     // [2][128] ... and their row sums
constexpr int kMsBars = 1296;  // 6 mbarriers, then the tensor memory's address
static_assert(kMsBars * 4 + 6 * 8 + 16 <= (int)kMiscBytes, "the misc area");
// The merge's weights, [rows a rank owns][kMaxSplit], go over the pass's tables once the passes are done.
static_assert(kTile * kMaxSplit <= kMsL, "the merge's weights fit below the row sums");

// The forward's tensor memory: S at columns [0, 128), P's three components at [128 + 64 k, + 64), O at [384, 512).
constexpr uint32_t kTmP = 128, kTmO = 384;

__device__ __forceinline__ int chunk_pos(int q, int r) { return (q & ~7) | ((q ^ r) & 7); }

__device__ __forceinline__ void coarse_fwd(const FwdArgs& a) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = su32(smem_raw);
  const uint32_t sbase = (raw + 1023u) & ~1023u;  // the 128-byte swizzle's atoms are 1,024-byte aligned
  uint8_t* smem = smem_raw + (sbase - raw);
  const uint32_t sQ = sbase, sKV = sbase + kQBytes;
  float* misc = reinterpret_cast<float*>(smem + kQBytes + kKVBytes);
  float* sKs = misc + kMsKs;
  float* sKb = misc + kMsKb;
  float* sVf = misc + kMsVf;
  float* sMax = misc + kMsMax;
  float* sLh = misc + kMsL;
  uint64_t* bars = reinterpret_cast<uint64_t*>(misc + kMsBars);
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(bars + 6);

  // Warps w and w + 4 share tensor-memory lanes 32 (w % 4) .. + 31, i.e. 32 rows of the tile, and take alternate
  // 32-key chunks of each pass.
  const int tid = (int)threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int grp = warp & 3, half = warp >> 2, row = 32 * grp + lane;
  const int ks = a.ksplit, split = (int)blockIdx.x % ks, tile = (int)blockIdx.x / ks, hd = (int)blockIdx.y;
  const int N = a.N;
  const int per = split_keys(N, ks), kb = min(N, split * per), ke = min(N, kb + per), nkeys = ke - kb;
  const int npass = (nkeys + kPassKeys - 1) / kPassKeys;
  // At most 64 keys: K and V side by side, all loaded at once. Otherwise each pass's V waits for its S product to
  // have read K's tile.
  const bool dual = nkeys <= 64;
  const int ra = dual ? 64 : kPassKeys;
  const uint32_t sK = sKV, sV = dual ? sKV + (uint32_t)(kPlanes * 64 * 128) : sKV;
  const size_t hrow = (size_t)hd * N;
  const int q0 = tile * kTile, qrow = q0 + row;
  const uint32_t bar_ld = su32(bars), bar_v = su32(bars + 1), bar_s = su32(bars + 2), bar_p = su32(bars + 3),
                 bar_o = su32(bars + 4);
  const int rpc = (kTile + ks - 1) / ks, rb = split * rpc, re = min(kTile, rb + rpc);

  if (tid == 0) {
    mbarrier_init(bar_ld, 1);        // the pass's tiles landed (bulk copies)
    mbarrier_init(bar_v, 1);         // V's tile landed, when it follows K's
    mbarrier_init(bar_s, 1);         // S's product done (tcgen05.commit)
    mbarrier_init(bar_p, kThreads);  // P written to tensor memory (every thread)
    mbarrier_init(bar_o, 1);         // the PV product done (tcgen05.commit)
    fence_mbarrier_init_release_cluster();
  }
  if (warp == 4) {
    tcgen05_alloc<1>(su32(tmem_slot), 512);
    tcgen05_relinquish_alloc_permit<1>();
  }

  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();

  const size_t hplane = (size_t)hd * kPlanes * a.rpad * 64;
  const __nv_bfloat16* qp = a.qp + hplane;
  const __nv_bfloat16* kp = a.kp + hplane;
  const __nv_bfloat16* vp = a.vp + hplane;
  const int nk0 = min(nkeys, kPassKeys), nkp0 = (nk0 + 31) & ~31;
  if (tid == 0) {  // the thread that initialised the barriers
    const uint32_t bytes = 768u * (uint32_t)kTile + (npass > 0 ? 768u * (uint32_t)nkp0 * (dual ? 2u : 1u) : 0u);
    mbarrier_arrive_expect_tx(bar_ld, bytes);
    load_tile(sQ, kTile, qp, a.rpad, q0, kTile, bar_ld);
    if (npass > 0) {
      load_tile(sK, ra, kp, a.rpad, kb, nkp0, bar_ld);
      if (dual) load_tile(sV, ra, vp, a.rpad, kb, nkp0, bar_ld);
    }
  }
  // The row's logit factor: its lift back off (2^-l <= 1), times scale * log2(e).
  const float rowf = mul_nf(unlift(qrow < N ? a.qe[hrow + qrow] : 0), a.scale_log2);
  __syncthreads();  // the tensor memory's address
  tcgen05_fence_after_thread_sync();
  const uint32_t tmem = *tmem_slot;
  const uint32_t trow = tmem + ((uint32_t)(32 * grp) << 16);
  float m_run = -INFINITY, l_run = 0.f;

  for (int p = 0; p < npass; ++p) {
    const int k0 = kb + p * kPassKeys, nk = min(kPassKeys, ke - k0), nkp = (nk + 31) & ~31;
    if (p > 0) {
      // The previous pass's PV product has read V's tile and P: the tile and S's columns are free.
      mbarrier_wait_parity(bar_o, (uint32_t)((p - 1) & 1));
      if (tid == 0) {
        mbarrier_arrive_expect_tx(bar_ld, 768u * (uint32_t)nkp);
        load_tile(sK, ra, kp, a.rpad, k0, nkp, bar_ld);
      }
    }
    if (tid < kPassKeys) {
      const bool ok = tid < nk;
      sKs[tid] = ok ? unlift(a.ke[hrow + k0 + tid]) : 0.f;
      sKb[tid] = ok ? 0.f : -INFINITY;
      sVf[tid] = ok ? unlift(a.ve[hrow + k0 + tid]) : 0.f;
    }
    __syncthreads();

    if (warp == 4) {
      if (elect_one_sync()) {
        // S = Q K^T: eight component products, each over 8 steps of 16 dims.
        mbarrier_wait_parity(bar_ld, (uint32_t)(p & 1));
        tcgen05_fence_after_thread_sync();
        const uint32_t idesc = make_idesc_bf16_f32(kTile, nkp, false, false);
#pragma unroll
        for (int pr = 0; pr < kProducts; ++pr) {
          const uint64_t dq = kmajor_desc(sQ + (uint32_t)(2 * kPa[pr] * kTile * 128));
          const uint64_t dk = kmajor_desc(sK + (uint32_t)(2 * kPb[pr] * ra * 128));
#pragma unroll
          for (int kd = 0; kd < kDim / 16; ++kd)
            tcgen05_mma_f16_ss_lead(1u, tmem, dq + kstep(kd, kTile), dk + kstep(kd, ra), idesc, pr > 0 || kd > 0);
        }
        tcgen05_commit1_lead(1u, bar_s);
        if (!dual) {
          // V's tile over K's once S's product has read it; it lands during the softmax.
          mbarrier_wait_parity(bar_s, (uint32_t)(p & 1));
          mbarrier_arrive_expect_tx(bar_v, 768u * (uint32_t)nkp);
          load_tile(sV, ra, vp, a.rpad, k0, nkp, bar_v);
        }
      }
      __syncwarp();
    }
    mbarrier_wait_parity(bar_s, (uint32_t)(p & 1));
    tcgen05_fence_after_thread_sync();
    const int nch = nkp >> 5;
    // Pass 1: the logits in log2 units, x = S * 2^-(lq + lk) * scale_log2 (-inf past the keys), back in place; the
    // row's largest. Two chunks per tensor-memory wait.
    float mx = -INFINITY;
    for (int c = half; c < nch; c += 4) {
      const bool two = c + 2 < nch;
      uint32_t r[2][32];
      tcgen05_ld_32x32b_x32(trow + (uint32_t)(32 * c), r[0]);
      if (two) tcgen05_ld_32x32b_x32(trow + (uint32_t)(32 * (c + 2)), r[1]);
      tcgen05_wait_ld();
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        if (h == 1 && !two) break;
        const int cc = c + 2 * h;
#pragma unroll
        for (int j = 0; j < 32; j += 4) {
          const float4 k4 = *reinterpret_cast<const float4*>(sKs + 32 * cc + j);
          const float4 b4 = *reinterpret_cast<const float4*>(sKb + 32 * cc + j);
          const float x0 = fma_nf(__uint_as_float(r[h][j]), mul_nf(rowf, k4.x), b4.x);
          const float x1 = fma_nf(__uint_as_float(r[h][j + 1]), mul_nf(rowf, k4.y), b4.y);
          const float x2 = fma_nf(__uint_as_float(r[h][j + 2]), mul_nf(rowf, k4.z), b4.z);
          const float x3 = fma_nf(__uint_as_float(r[h][j + 3]), mul_nf(rowf, k4.w), b4.w);
          mx = fmaxf(mx, fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)));
          r[h][j] = __float_as_uint(x0);
          r[h][j + 1] = __float_as_uint(x1);
          r[h][j + 2] = __float_as_uint(x2);
          r[h][j + 3] = __float_as_uint(x3);
        }
        tcgen05_st_32x32b_x32(trow + (uint32_t)(32 * cc), r[h]);
      }
    }
    sMax[half * kTile + row] = mx;
    named_bar(1 + grp, 64);
    const float m_new = fmaxf(m_run, fmaxf(sMax[row], sMax[kTile + row]));
    // A later pass raised a row's max: the partial O (this half's 64 dims) and l are rescaled. The tensor-memory
    // loads and stores are warp-collective (.sync.aligned), so the whole warp takes the branch when any row needs it,
    // the others with alpha = 1 (exact).
    if (__any_sync(0xffffffffu, p > 0 && m_new > m_run)) {
      const float alpha = m_new > m_run ? ex2_nf(m_run - m_new) : 1.f;
      l_run *= alpha;
#pragma unroll 1
      for (int c = 0; c < 2; ++c) {
        uint32_t r[32];
        tcgen05_ld_32x32b_x32(trow + kTmO + (uint32_t)(64 * half + 32 * c), r);
        tcgen05_wait_ld();
#pragma unroll
        for (int j = 0; j < 32; ++j) r[j] = __float_as_uint(mul_nf(__uint_as_float(r[j]), alpha));
        tcgen05_st_32x32b_x32(trow + kTmO + (uint32_t)(64 * half + 32 * c), r);
      }
    }
    m_run = m_new;
    tcgen05_wait_st();
    // Pass 2: P = exp2(x - m), its fp32 sum, and P times the V row's fold as three bf16 components (16 columns each
    // per 32-key chunk).
    for (int c = half; c < nch; c += 2) {
      uint32_t r[32], pc[kComps][16];
      tcgen05_ld_32x32b_x32(trow + (uint32_t)(32 * c), r);
      tcgen05_wait_ld();
#pragma unroll
      for (int jj = 0; jj < 16; ++jj) {
        const float p0 = ex2_nf(__uint_as_float(r[2 * jj]) - m_run);
        const float p1 = ex2_nf(__uint_as_float(r[2 * jj + 1]) - m_run);
        l_run += p0 + p1;
        const float2 f2 = *reinterpret_cast<const float2*>(sVf + 32 * c + 2 * jj);
        uint32_t w[kComps];
        split3_x2(mul_nf(p0, f2.x), mul_nf(p1, f2.y), w);
        pc[0][jj] = w[0];
        pc[1][jj] = w[1];
        pc[2][jj] = w[2];
      }
#pragma unroll
      for (int k = 0; k < kComps; ++k) tcgen05_st_32x32b_x16(trow + kTmP + (uint32_t)(64 * k + 16 * c), pc[k]);
    }
    tcgen05_wait_st();
    tcgen05_fence_before_thread_sync();
    mbarrier_arrive(bar_p);
    if (warp == 4) {
      if (elect_one_sync()) {
        // O += P V: eight component products over steps of 16 keys, P from tensor memory, V from its tile (MN-major).
        if (!dual) mbarrier_wait_parity(bar_v, (uint32_t)(p & 1));
        mbarrier_wait_parity(bar_p, (uint32_t)(p & 1));
        tcgen05_fence_after_thread_sync();
        const uint32_t idesc = make_idesc_bf16_f32(kTile, kDim, false, true);
#pragma unroll
        for (int pr = 0; pr < kProducts; ++pr) {
          const uint64_t dv = build_smem_desc_blackwell(sV + (uint32_t)(2 * kPb[pr] * ra * 128), 1024,
                                                        (uint32_t)ra * 128u, SmemSwizzleBlackwell::B128);
          const uint32_t pa = tmem + kTmP + (uint32_t)(64 * kPa[pr]);
          for (int u = 0; u < nkp / 16; ++u)
            tcgen05_mma_f16_ts_1sm_lead(1u, tmem + kTmO, pa + (uint32_t)(8 * u), dv + (uint64_t)u * 128u, idesc,
                                        p > 0 || pr > 0 || u > 0);
        }
        tcgen05_commit1_lead(1u, bar_o);
      }
      __syncwarp();
    }
  }

  // The merge, through global memory (L2): every split writes its partial (O and its rows' m and l) to its own slot;
  // after the cluster's barrier each rank combines the rows it owns, rows [r * rpc, (r + 1) * rpc) for rank r, with
  // the weights exp2(m - M) / L.
  if (npass > 0) {
    mbarrier_wait_parity(bar_o, (uint32_t)((npass - 1) & 1));
    tcgen05_fence_after_thread_sync();
  }
  const size_t slot = ((size_t)hd * a.tiles + tile) * kMaxSplit;
  sLh[half * kTile + row] = l_run;
  named_bar(1 + grp, 64);
  if (half == 0) {
    float* pst = a.pstat + (slot + split) * 2 * kTile;
    pst[row] = m_run;
    pst[kTile + row] = sLh[row] + sLh[kTile + row];
  }
  // The partial through a swizzled stage over the Q tile, so that its rows leave in whole lines.
  float* sStage = reinterpret_cast<float*>(smem);  // [128][128]
#pragma unroll 1
  for (int c = 0; c < 2; ++c) {
    uint32_t r[32];
    if (npass > 0) {
      tcgen05_ld_32x32b_x32(trow + kTmO + (uint32_t)(64 * half + 32 * c), r);
      tcgen05_wait_ld();
    } else {
#pragma unroll
      for (int j = 0; j < 32; ++j) r[j] = 0u;
    }
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      const int q = (64 * half + 32 * c + j) >> 2;
      *reinterpret_cast<float4*>(sStage + row * kDim + 4 * chunk_pos(q, row)) =
          make_float4(__uint_as_float(r[j]), __uint_as_float(r[j + 1]), __uint_as_float(r[j + 2]),
                      __uint_as_float(r[j + 3]));
    }
  }
  tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (warp == 4) {
    tcgen05_fence_after_thread_sync();
    tcgen05_dealloc<1>(tmem, 512);
  }
  {
    float* part = a.part + (slot + split) * kTile * kDim;
#pragma unroll 4
    for (int it = tid; it < kTile * (kDim / 4); it += kThreads) {
      const int r = it >> 5, q = it & 31;
      *reinterpret_cast<float4*>(part + r * kDim + 4 * q) =
          *reinterpret_cast<const float4*>(sStage + r * kDim + 4 * chunk_pos(q, r));
    }
  }
  cluster_sync_all();  // release / acquire at cluster scope: every split's partial is visible to the cluster

  float* sW = sKs;  // [rpc][kMaxSplit] the owned rows' weights, over the pass's tables
  if (tid < re - rb) {
    const int r = rb + tid;
    float mr[kMaxSplit], lr[kMaxSplit];
#pragma unroll
    for (int s = 0; s < kMaxSplit; ++s) {
      mr[s] = s < ks ? __ldcg(a.pstat + (slot + s) * 2 * kTile + r) : -INFINITY;
      lr[s] = s < ks ? __ldcg(a.pstat + (slot + s) * 2 * kTile + kTile + r) : 0.f;
    }
    float M = -INFINITY;
#pragma unroll
    for (int s = 0; s < kMaxSplit; ++s) M = fmaxf(M, mr[s]);
    float e[kMaxSplit], L = 0.f;
#pragma unroll
    for (int s = 0; s < kMaxSplit; ++s) {
      e[s] = s < ks ? ex2(mr[s] - M) : 0.f;  // an empty split's m is -inf: weight 0
      L += e[s] * lr[s];
    }
    const float inv = 1.f / L;
#pragma unroll
    for (int s = 0; s < kMaxSplit; ++s) sW[tid * kMaxSplit + s] = e[s] * inv;
    if (q0 + r < N) a.lse2[(size_t)hd * a.npad + q0 + r] = M + __log2f(L);
  }
  __syncthreads();
  const int items = (re - rb) * (kDim / 4);
  for (int it = tid; it < items; it += kThreads) {
    const int lr_ = it >> 5, q = it & 31, orow = q0 + rb + lr_;
    float4 v[kMaxSplit];
#pragma unroll
    for (int s = 0; s < kMaxSplit; ++s)
      v[s] = s < ks ? __ldcg(reinterpret_cast<const float4*>(a.part + ((slot + s) * kTile + rb + lr_) * kDim) + q)
                    : make_float4(0.f, 0.f, 0.f, 0.f);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < kMaxSplit; ++s) {
      const float ws = sW[lr_ * kMaxSplit + s];
      acc.x = fma_nf(ws, v[s].x, acc.x);
      acc.y = fma_nf(ws, v[s].y, acc.y);
      acc.z = fma_nf(ws, v[s].z, acc.z);
      acc.w = fma_nf(ws, v[s].w, acc.w);
    }
    if (orow < N) *reinterpret_cast<float4*>(a.oc + ((size_t)hd * N + orow) * kDim + 4 * q) = acc;
  }
}

// Phase timestamps for thread 0 of chosen CTAs (a diagnostic build defines COARSE_TC_STAMPS; no other build does).
#ifdef COARSE_TC_STAMPS
#define CTS_DECL                     \
  unsigned long long cts_0, cts_[16]; \
  int cts_n = 0;                      \
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(cts_0)::"memory");
#define CTS()                                                          \
  do {                                                                 \
    unsigned long long t_;                                             \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)::"memory");    \
    if (cts_n < 16) cts_[cts_n++] = t_ - cts_0;                        \
  } while (0)
#define CTS_PRINT(cond)                                                                                              \
  do {                                                                                                               \
    for (int i_ = cts_n; i_ < 16; ++i_) cts_[i_] = 0;                                                                \
    if (threadIdx.x == 0 && (cond))                                                                                  \
      printf("CTS r%d s%d t%d np%d | %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n", \
             role, split, tile, npass, cts_[0], cts_[1], cts_[2], cts_[3], cts_[4], cts_[5], cts_[6], cts_[7], cts_[8],   \
             cts_[9], cts_[10], cts_[11], cts_[12], cts_[13], cts_[14], cts_[15]);                                   \
  } while (0)
#else
#define CTS_DECL
#define CTS()
#define CTS_PRINT(cond)
#endif

// The coarse backward's shared memory past the tiles (kMiscBytes), in floats: the pass's column tables.
constexpr int kGcS = 0;       // [64] S's column factor 2^-lift (0 past the columns)
constexpr int kGcB = 64;      // [64] dQ: 0, or -inf past the keys; dK/dV: lse2_i (+inf past the queries)
constexpr int kGcV = 128;     // [64] dP's column factor: 2^-lv_j (dQ) or 2^-lo_i (dK)
constexpr int kGcD = 192;     // [64] dK: Dc_i
constexpr int kGcF = 256;     // [64] the second product's column fold: 2^-lk_j (dQ), 2^-lq_i (dK), 2^-lo_i (dV)
constexpr int kGcBars = 320;  // 6 mbarriers, then the tensor memory's address
static_assert(kGcBars * 4 + 6 * 8 + 16 <= (int)kMiscBytes, "the coarse backward's misc area");

// The backward's tensor memory: the rows' second operand (dQ role: dO, dK role: V) as three components at [0, 192),
// S at [192, 256), dP at [256, 320), the accumulator at [384, 512). The second product's A operand (dS or P times
// its column fold) of a 32-column half h goes over that half's own S and dP columns: component 0 at S's 32 h,
// component 1 16 columns on, component 2 at dP's 32 h.
constexpr uint32_t kTmR2 = 0, kTmS = 192, kTmDP = 256, kTmAcc = 384;
__device__ __forceinline__ uint32_t a_col(int h, int k) {
  return k < 2 ? kTmS + (uint32_t)(32 * h + 16 * k) : kTmDP + (uint32_t)(32 * h);
}

__device__ __forceinline__ void coarse_grad(const GradArgs& a) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = su32(smem_raw);
  const uint32_t sbase = (raw + 1023u) & ~1023u;
  uint8_t* smem = smem_raw + (sbase - raw);
  const uint32_t sR1 = sbase, sC1 = sbase + kRBytes, sC2 = sC1 + kCBytes;
  float* misc = reinterpret_cast<float*>(smem + kRBytes + 2 * kCBytes);
  float* cS = misc + kGcS;
  float* cB = misc + kGcB;
  float* cV = misc + kGcV;
  float* cD = misc + kGcD;
  float* cF = misc + kGcF;
  uint64_t* bars = reinterpret_cast<uint64_t*>(misc + kGcBars);
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(bars + 6);

  // The role: CTAs [0, tiles * ksplit) of a head are the dQ role (rows = queries, columns = keys), the next as many
  // the dK role and the last the dV role (rows = keys, columns = queries); a cluster is `ksplit` CTAs of one role and
  // one tile.
  const int tid = (int)threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int grp = warp & 3, half = warp >> 2, row = 32 * grp + lane;
  const int ks = a.ksplit, N = a.N, hd = (int)blockIdx.y;
  const int per_role = a.tiles * ks;
  const int role = (int)blockIdx.x / per_role;
  const int idx = (int)blockIdx.x - role * per_role, split = idx % ks, tile = idx / ks;
  const bool has_dp = role != 2;
  const int per = split_keys(N, ks), cb = min(N, split * per), ce = min(N, cb + per), ncols = ce - cb;
  // Passes of at most four 16-column steps, as even as the steps allow (so that no pass is a short tail): pass p
  // starts at column cb + 16 (p psteps + min(p, pextra)) and takes psteps steps, one more when p < pextra.
  const int nsteps = (ncols + 15) >> 4;
  const int npass = (nsteps + 3) >> 2;
  const int psteps = npass > 0 ? nsteps / npass : 0, pextra = npass > 0 ? nsteps % npass : 0;
  auto pass_k0 = [&](int p) { return cb + 16 * (p * psteps + min(p, pextra)); };
  auto pass_nkp = [&](int p) {
    const int k0 = pass_k0(p);
    return (min(16 * (psteps + (p < pextra ? 1 : 0)), ce - k0) + 15) & ~15;
  };
  const int r0 = tile * kTile, rrow = r0 + row;
  const bool rvalid = rrow < N;
  const size_t hrow = (size_t)hd * N, hpad = (size_t)hd * a.npad, hplane = (size_t)hd * kPlanes * a.rpad * 64;
  const __nv_bfloat16* r1p = (role == 0 ? a.qp : a.kp) + hplane;  // S's rows (shared memory)
  const __nv_bfloat16* r2p = (role == 0 ? a.op : a.vp) + hplane;  // dP's rows (tensor memory)
  const __nv_bfloat16* c1p = (role == 0 ? a.kp : a.qp) + hplane;  // S's columns, and the dQ / dK product's B
  const __nv_bfloat16* c2p = (role == 0 ? a.vp : a.op) + hplane;  // dP's columns, and the dV product's B
  const uint32_t sB = role == 2 ? sC2 : sC1;                     // the second product's B (MN-major)
  // The column tile that is free once S and dP are done (dQ, dK: dP's columns; dV: S's), loaded a pass ahead, and
  // the one the pass's last product reads too, loaded once it is done.
  const uint32_t sE = role == 2 ? sC1 : sC2, sL = role == 2 ? sC2 : sC1;
  const __nv_bfloat16* ep = role == 2 ? c1p : c2p;
  const __nv_bfloat16* lp = role == 2 ? c2p : c1p;
  const uint32_t bar_r = su32(bars), bar_ce = su32(bars + 1), bar_s = su32(bars + 2), bar_p = su32(bars + 3),
                 bar_o = su32(bars + 4), bar_cl = su32(bars + 5);
  CTS_DECL

  if (tid == 0) {
    mbarrier_init(bar_r, 1);         // the rows' tile landed
    mbarrier_init(bar_ce, 1);        // the pass's early-free column tile landed
    mbarrier_init(bar_cl, 1);        // ... and its other column tile
    mbarrier_init(bar_s, 1);         // S and dP done (tcgen05.commit)
    mbarrier_init(bar_p, kThreads);  // the second product's A written to tensor memory
    mbarrier_init(bar_o, 1);         // the pass's products done: its tiles and columns are free
    fence_mbarrier_init_release_cluster();
  }
  if (warp == 4) {
    tcgen05_alloc<1>(su32(tmem_slot), 512);
    tcgen05_relinquish_alloc_permit<1>();
  }

  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  CTS();

  if (tid == 0) {
    mbarrier_arrive_expect_tx(bar_r, 768u * (uint32_t)kTile);
    load_tile(sR1, kTile, r1p, a.rpad, r0, kTile, bar_r);
    if (npass > 0) {  // the first pass's column tiles
      const int nkp0 = pass_nkp(0);
      mbarrier_arrive_expect_tx(bar_ce, 768u * (uint32_t)nkp0);
      load_tile(sE, kGradCols, ep, a.rpad, cb, nkp0, bar_ce);
      mbarrier_arrive_expect_tx(bar_cl, 768u * (uint32_t)nkp0);
      load_tile(sL, kGradCols, lp, a.rpad, cb, nkp0, bar_cl);
    }
  }

  // The row's factors. dQ: x = S 2^-(lq_i + lk_j) scale_log2 - lse2_i, dP = dP' 2^-(lo_i + lv_j), dS = P (dP - Dc_i),
  // A = dS 2^-lk_j. dK / dV: the same with the roles of the rows and the columns exchanged; dV's A = P 2^-lo_i.
  float rowf, rl2 = 0.f, ro = 0.f, rdc = 0.f;
  if (role == 0) {
    rowf = mul_nf(unlift(rvalid ? a.qe[hrow + rrow] : 0), a.scale_log2);
    rl2 = rvalid ? a.lse2[hpad + rrow] : INFINITY;
    ro = rvalid ? unlift(a.oe[hrow + rrow]) : 0.f;
    rdc = rvalid ? a.dc[hpad + rrow] : 0.f;
  } else {
    rowf = mul_nf(unlift(rvalid ? a.ke[hrow + rrow] : 0), a.scale_log2);
    if (role == 1) ro = rvalid ? unlift(a.ve[hrow + rrow]) : 0.f;
  }

  // A pass's column tables: threads 0..63 fetch their column's entries one pass ahead, into registers, so that the
  // global loads run during the previous pass.
  float tS = 0.f, tB = 0.f, tV = 0.f, tD = 0.f, tF = 0.f;
  auto fetch_col = [&](int k0) {
    const int c = k0 + tid;
    const bool ok = c < ce;
    if (role == 0) {
      const float fk = ok ? unlift(a.ke[hrow + c]) : 0.f;
      tS = fk;
      tB = ok ? 0.f : -INFINITY;
      tV = ok ? unlift(a.ve[hrow + c]) : 0.f;
      tF = fk;
    } else {
      const float fq = ok ? unlift(a.qe[hrow + c]) : 0.f;
      tS = fq;
      tB = ok ? -a.lse2[hpad + c] : -INFINITY;  // the logit's bias: minus the query's lse2
      if (role == 1) {
        tV = ok ? unlift(a.oe[hrow + c]) : 0.f;
        tD = ok ? a.dc[hpad + c] : 0.f;
        tF = fq;
      } else {
        tF = ok ? unlift(a.oe[hrow + c]) : 0.f;
      }
    }
  };
  if (tid < kGradCols && npass > 0) fetch_col(cb);
  __syncthreads();  // the tensor memory's address
  tcgen05_fence_after_thread_sync();
  const uint32_t tmem = *tmem_slot;
  const uint32_t trow = tmem + ((uint32_t)(32 * grp) << 16);
  if (has_dp) {
    // dP's rows into tensor memory, as the product's A: this thread's row, its half's 64 dims of each component
    // (columns 64 k + 32 half: two bf16 per column, the lower dim in the low half, as the planes hold them).
    // All 24 loads in flight at once, then the three stores.
    uint4 v[kComps][8];
#pragma unroll
    for (int k = 0; k < kComps; ++k) {
      const uint4* src = reinterpret_cast<const uint4*>(r2p + ((size_t)(2 * k + half) * a.rpad + rrow) * 64);
#pragma unroll
      for (int c = 0; c < 8; ++c) v[k][c] = __ldg(src + (c ^ (rrow & 7)));
    }
#pragma unroll
    for (int k = 0; k < kComps; ++k) {
      uint32_t w[32];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        w[4 * c] = v[k][c].x;
        w[4 * c + 1] = v[k][c].y;
        w[4 * c + 2] = v[k][c].z;
        w[4 * c + 3] = v[k][c].w;
      }
      tcgen05_st_32x32b_x32(trow + kTmR2 + (uint32_t)(64 * k + 32 * half), w);
    }
    tcgen05_wait_st();
  }
  tcgen05_fence_before_thread_sync();  // ordered before the MMA warp's reads by the pass loop's __syncthreads
  CTS();

  for (int p = 0; p < npass; ++p) {
    const int k0 = pass_k0(p), nkp = pass_nkp(p);
    if (p > 0) {
      // The previous pass's products have read its late column tile and the A operand: both are free.
      mbarrier_wait_parity(bar_o, (uint32_t)((p - 1) & 1));
      tcgen05_fence_after_thread_sync();
      CTS();
      if (tid == 0) {
        mbarrier_arrive_expect_tx(bar_cl, 768u * (uint32_t)nkp);
        load_tile(sL, kGradCols, lp, a.rpad, k0, nkp, bar_cl);
      }
    }
    if (tid < kGradCols) {
      cS[tid] = tS;
      cB[tid] = tB;
      cV[tid] = tV;
      cD[tid] = tD;
      cF[tid] = tF;
    }
    __syncthreads();
    if (tid < kGradCols && p + 1 < npass) fetch_col(pass_k0(p + 1));

    if (warp == 4) {
      if (elect_one_sync()) {
        // S = R1 C1^T (both from shared memory) and dP = R2 C2^T (R2 from tensor memory): eight component products
        // each, over 8 steps of 16 dims. dP first: its column tile was loaded a pass ahead, so it runs while the
        // late tile (S's, in the dQ and dK roles) lands. In the dV role S's tile is the early one.
        if (p == 0) mbarrier_wait_parity(bar_r, 0u);
        mbarrier_wait_parity(bar_ce, (uint32_t)(p & 1));
        tcgen05_fence_after_thread_sync();
        const uint32_t idesc = make_idesc_bf16_f32(kTile, nkp, false, false);
        if (has_dp) {
#pragma unroll
          for (int pr = 0; pr < kProducts; ++pr) {
            const uint64_t dc2 = kmajor_desc(sC2 + (uint32_t)(2 * kPb[pr] * kGradCols * 128));
#pragma unroll
            for (int kd = 0; kd < kDim / 16; ++kd)
              tcgen05_mma_f16_ts_1sm_lead(1u, tmem + kTmDP, tmem + kTmR2 + (uint32_t)(64 * kPa[pr] + 8 * kd),
                                          dc2 + kstep(kd, kGradCols), idesc, pr > 0 || kd > 0);
          }
          mbarrier_wait_parity(bar_cl, (uint32_t)(p & 1));
          tcgen05_fence_after_thread_sync();
        }
#pragma unroll
        for (int pr = 0; pr < kProducts; ++pr) {
          const uint64_t dr = kmajor_desc(sR1 + (uint32_t)(2 * kPa[pr] * kTile * 128));
          const uint64_t dc1 = kmajor_desc(sC1 + (uint32_t)(2 * kPb[pr] * kGradCols * 128));
#pragma unroll
          for (int kd = 0; kd < kDim / 16; ++kd)
            tcgen05_mma_f16_ss_lead(1u, tmem + kTmS, dr + kstep(kd, kTile), dc1 + kstep(kd, kGradCols), idesc,
                                    pr > 0 || kd > 0);
        }
        tcgen05_commit1_lead(1u, bar_s);
      }
      __syncwarp();
    }
    mbarrier_wait_parity(bar_s, (uint32_t)(p & 1));
    tcgen05_fence_after_thread_sync();
    CTS();
    if (tid == 0 && p + 1 < npass) {
      // S and dP have read the early tile: the next pass's goes over it during this pass's softmax and last product.
      const int k1 = pass_k0(p + 1), nkp1 = pass_nkp(p + 1);
      mbarrier_arrive_expect_tx(bar_ce, 768u * (uint32_t)nkp1);
      load_tile(sE, kGradCols, ep, a.rpad, k1, nkp1, bar_ce);
    }
    // This half's 32 columns: P, dS, and the second product's A as three components over its own S and dP columns.
    if (32 * half < nkp) {
      const int c0 = 32 * half;
      uint32_t s[32], d[32];
      tcgen05_ld_32x32b_x32(trow + kTmS + (uint32_t)c0, s);
      if (has_dp) tcgen05_ld_32x32b_x32(trow + kTmDP + (uint32_t)c0, d);
      tcgen05_wait_ld();
      uint32_t ac[kComps][16];
      const float2 rowf2 = make_float2(rowf, rowf), ro2 = make_float2(ro, ro), rl22 = make_float2(rl2, rl2),
                   rdc2 = make_float2(rdc, rdc);
      // Column pairs on packed fp32 (the same operations and rounding per element as on scalars).
#pragma unroll
      for (int jj = 0; jj < 16; ++jj) {
        const int j = c0 + 2 * jj;
        const float2 sc = *reinterpret_cast<const float2*>(cS + j);
        const float2 bc = *reinterpret_cast<const float2*>(cB + j);
        const float2 fc = *reinterpret_cast<const float2*>(cF + j);
        const float2 sv = make_float2(__uint_as_float(s[2 * jj]), __uint_as_float(s[2 * jj + 1]));
        // The logit in log2 units less lse2: dQ, the row's lse2 (bc is 0, or -inf past the keys); dK, dV, the
        // column's (bc is minus it).
        const float2 x = fma2_nf(sv, mul2_nf(rowf2, sc), role == 0 ? sub2_nf(bc, rl22) : bc);
        float2 p = make_float2(ex2_nf(x.x), ex2_nf(x.y));
        if (role != 0 && !rvalid) p = make_float2(0.f, 0.f);
        float2 y;
        if (role == 2) {
          y = mul2_nf(p, fc);
        } else {
          const float2 vc = *reinterpret_cast<const float2*>(cV + j);
          const float2 dd = role == 0 ? rdc2 : *reinterpret_cast<const float2*>(cD + j);
          const float2 dv = make_float2(__uint_as_float(d[2 * jj]), __uint_as_float(d[2 * jj + 1]));
          y = mul2_nf(mul2_nf(p, sub2_nf(mul2_nf(dv, mul2_nf(ro2, vc)), dd)), fc);
        }
        uint32_t w[kComps];
        split3_f2(y, w);
        ac[0][jj] = w[0];
        ac[1][jj] = w[1];
        ac[2][jj] = w[2];
      }
#pragma unroll
      for (int k = 0; k < kComps; ++k) tcgen05_st_32x32b_x16(trow + a_col(half, k), ac[k]);
    }
    tcgen05_wait_st();
    tcgen05_fence_before_thread_sync();
    mbarrier_arrive(bar_p);
    CTS();
    if (warp == 4) {
      if (elect_one_sync()) {
        // dQ += dS K, dK += dS^T Q or dV += P^T dO: eight component products over steps of 16 columns, A from
        // tensor memory, B the column tile (MN-major).
        mbarrier_wait_parity(bar_p, (uint32_t)(p & 1));
        if (!has_dp) mbarrier_wait_parity(bar_cl, (uint32_t)(p & 1));  // dV's B is its late tile
        tcgen05_fence_after_thread_sync();
        const uint32_t idesc = make_idesc_bf16_f32(kTile, kDim, false, true);
#pragma unroll
        for (int pr = 0; pr < kProducts; ++pr) {
          const uint64_t db = build_smem_desc_blackwell(sB + (uint32_t)(2 * kPb[pr] * kGradCols * 128), 1024,
                                                        (uint32_t)kGradCols * 128u, SmemSwizzleBlackwell::B128);
          for (int u = 0; u < nkp / 16; ++u)
            tcgen05_mma_f16_ts_1sm_lead(1u, tmem + kTmAcc, tmem + a_col(u >> 1, kPa[pr]) + (uint32_t)(8 * (u & 1)),
                                        db + (uint64_t)u * 128u, idesc, p > 0 || pr > 0 || u > 0);
        }
        tcgen05_commit1_lead(1u, bar_o);
      }
      __syncwarp();
    }
  }

  // The partial, times scale (dQ, dK) and the rows' 1 / count, through a swizzled stage over the rows' tile, added to
  // dqc, dkc or dvc (zeroed by bwd_rows) with vector reductions, 512 B per warp instruction. No cluster: the splits
  // meet in L2, so a split's CTAs need not be co-resident. (fp32 adds in any order: within 2^-24 of the sum's
  // magnitude per add, like any fixed order; a subnormal sum may flush, an absolute 2^-126 against dq's 1e-2.)
  if (npass > 0) {
    mbarrier_wait_parity(bar_o, (uint32_t)((npass - 1) & 1));
    tcgen05_fence_after_thread_sync();
    CTS();
    const float rc = rvalid ? a.rcnt[rrow] : 0.f;
    const float uo = role == 2 ? rc : a.scale * rc;
    float* sStage = reinterpret_cast<float*>(smem);  // [128][128]
#pragma unroll 1
    for (int c = 0; c < 2; ++c) {
      uint32_t r[32];
      tcgen05_ld_32x32b_x32(trow + kTmAcc + (uint32_t)(64 * half + 32 * c), r);
      tcgen05_wait_ld();
#pragma unroll
      for (int j = 0; j < 32; j += 4) {
        const int q = (64 * half + 32 * c + j) >> 2;
        *reinterpret_cast<float4*>(sStage + row * kDim + 4 * chunk_pos(q, row)) =
            make_float4(uo * __uint_as_float(r[j]), uo * __uint_as_float(r[j + 1]), uo * __uint_as_float(r[j + 2]),
                        uo * __uint_as_float(r[j + 3]));
      }
    }
    tcgen05_fence_before_thread_sync();
    __syncthreads();
    if (warp == 4) {
      tcgen05_fence_after_thread_sync();
      tcgen05_dealloc<1>(tmem, 512);
    }
    float* out = (role == 0 ? a.dqc : (role == 1 ? a.dkc : a.dvc)) + (hrow + r0) * kDim;
    const int items = min(kTile, N - r0) * (kDim / 4);
#pragma unroll 4
    for (int it = tid; it < items; it += kThreads) {
      const int r = it >> 5, q = it & 31;
      const float4 v = *reinterpret_cast<const float4*>(sStage + r * kDim + 4 * chunk_pos(q, r));
      asm volatile("red.global.v4.f32.add [%0], {%1, %2, %3, %4};\n" ::"l"(out + r * kDim + 4 * q), "f"(v.x),
                   "f"(v.y), "f"(v.z), "f"(v.w)
                   : "memory");
    }
  } else {
    // A split without columns (grad_split makes none) adds nothing, once its rows' load has landed.
    mbarrier_wait_parity(bar_r, 0u);
    tcgen05_fence_before_thread_sync();
    __syncthreads();
    if (warp == 4) {
      tcgen05_fence_after_thread_sync();
      tcgen05_dealloc<1>(tmem, 512);
    }
  }
  CTS();
  CTS_PRINT(blockIdx.y == 0 && idx == 0);
}

}  // namespace

__global__ void __launch_bounds__(kThreads, 1) fwd_coarse_tc_kernel(FwdArgs a) { coarse_fwd(a); }
__global__ void __launch_bounds__(kThreads, 1) bwd_coarse_tc_kernel(FwdArgs a) { coarse_fwd(a); }
__global__ void __launch_bounds__(kThreads, 1) bwd_coarse_grad_tc_kernel(GradArgs a) { coarse_grad(a); }

}  // namespace coarse_tc
}  // namespace vsa2t
