// glue.cu -- the definitions of the backward's two row-wise passes. See glue.cuh for what they compute and for why
// they are in their own translation unit rather than inlined into `vsa2t.cu` with every other stage.
#include "glue.cuh"

#include "coarse_tc.cuh"
#include "primitives.cuh"

namespace vsa2t {
namespace glue {

__global__ void __launch_bounds__(kRowsThreads) bwd_rows_kernel(
    Geo g, const __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ dout,
    const __nv_bfloat16* __restrict__ gate, const float* __restrict__ lse, const float* __restrict__ oc,
    const uint32_t* __restrict__ rowmask, float* __restrict__ delta, float* __restrict__ lse2,
    __nv_bfloat16* __restrict__ dgate, __nv_bfloat16* __restrict__ dco_planes, int* __restrict__ dco_lift, int rpad,
    float* __restrict__ dc, float* __restrict__ dqaccum, float* __restrict__ coarse_zero) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  const int b = (int)blockIdx.x, hd = (int)blockIdx.y;
  const int tid = (int)threadIdx.x, c = tid & 15, rg = tid >> 4;
  // The block's rows of dqc, dkc and dvc, which the coarse backward adds to: 3 x 32 float4.
  if (tid < 3 * (kHeadDim / 4)) {
    const int arr = tid / (kHeadDim / 4), q = tid % (kHeadDim / 4);
    *reinterpret_cast<float4*>(coarse_zero + (((size_t)arr * g.heads + hd) * g.N + b) * kHeadDim + 4 * q) =
        make_float4(0.f, 0.f, 0.f, 0.f);
  }
  // The block's coarse output: its 8 dims of this thread, read once (the same for the block's 128 rows).
  const float* ocb = oc + ((size_t)hd * g.N + b) * kHeadDim + c * kDimsPerThread;
  const float4 o0 = *reinterpret_cast<const float4*>(ocb), o1 = *reinterpret_cast<const float4*>(ocb + 4);
  const float ocv[8] = {o0.x, o0.y, o0.z, o0.w, o1.x, o1.y, o1.z, o1.w};
  const float* lseb = lse + (size_t)hd * g.S + (size_t)b * kBlock;
  float* deltab = delta + (size_t)hd * g.S + (size_t)b * kBlock;
  float* lse2b = lse2 + (size_t)hd * g.S + (size_t)b * kBlock;
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;

  size_t at = ((size_t)hd * g.S + (size_t)b * kBlock + rg) * kHeadDim + c * kDimsPerThread;
  const size_t step = (size_t)kRowGroups * kHeadDim;
#pragma unroll
  for (int it = 0; it < kRowsPerGroup; ++it, at += step) {
    const int rr = rg + kRowGroups * it;
    const bool ok = mask_bit(rowmask, b, rr);
    uint4 dg = make_uint4(0u, 0u, 0u, 0u);
    float d = 0.f;
    if (ok) {
      const uint4 ov = *reinterpret_cast<const uint4*>(out + at);
      const uint4 dv = *reinterpret_cast<const uint4*>(dout + at);
      const uint4 gv = *reinterpret_cast<const uint4*>(gate + at);
      uint32_t gp[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 o2 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&ov)[e]);
        const float2 d2 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&dv)[e]);
        const float2 g2 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&gv)[e]);
        d += d2.x * (o2.x - g2.x * ocv[2 * e]) + d2.y * (o2.y - g2.y * ocv[2 * e + 1]);
        // The coarse output's gradient pools gate * dout over the block, keeping subnormals (coarse_tc.cuh *_nf).
        acc[2 * e] = coarse_tc::fma_nf(g2.x, d2.x, acc[2 * e]);
        acc[2 * e + 1] = coarse_tc::fma_nf(g2.y, d2.y, acc[2 * e + 1]);
        const __nv_bfloat162 y = __floats2bfloat162_rn(ocv[2 * e] * d2.x, ocv[2 * e + 1] * d2.y);
        gp[e] = *reinterpret_cast<const uint32_t*>(&y);
      }
      dg = make_uint4(gp[0], gp[1], gp[2], gp[3]);
    }
    *reinterpret_cast<uint4*>(dgate + at) = dg;
    // The accumulator's zeroing is NOT guarded by `ok`, and that is a measured result rather than an oversight
    // (J876: guarding it reads `lowest match 0.9992 in dq` at s8). The reason is a layout mismatch: this loop
    // writes row-major (`row*kHeadDim + dim`), while bwd_post reads the *drain-native* order
    // (`[slice of 32 columns][v4][row][4]`, bwd.cuh), a permutation of the same 128-row block. So skipping the
    // row-major positions of a padding row leaves some of a *valid* row's drain positions untouched -- the two
    // keys disagree about which element belongs to which row. Zeroing all of it is what makes the keys
    // interchangeable; to skip the padding rows' stores the loop would have to be re-keyed to the drain layout.
    *reinterpret_cast<float4*>(dqaccum + at) = make_float4(0.f, 0.f, 0.f, 0.f);
    *reinterpret_cast<float4*>(dqaccum + at + 4) = make_float4(0.f, 0.f, 0.f, 0.f);
    // The row's dot product: the 16 threads of the group hold its 128 dims.
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
    if (c == 0) {
      deltab[rr] = ok ? d : 0.f;
      lse2b[rr] = ok ? lseb[rr] * 1.4426950408889634f : INFINITY;
    }
  }

  // dco: this thread's 8 dims of its 4 rows, then the 32 row groups in shared memory (thread = dim).
  __shared__ float red[kRowGroups][kHeadDim];
#pragma unroll
  for (int e = 0; e < 8; ++e) red[rg][c * kDimsPerThread + e] = acc[e];
  __syncthreads();
  // dco as the coarse backward's operand: three bf16 components under the row's lift, laid out as its tile planes
  // (the same form as the means', prep.cuh block_means; coarse_tc.cuh), and Dc = rowsum(dco * Oc) of the fp32 sum:
  // [heads][npad], npad = N rounded up to 32. Sums that keep subnormals, and integer maxima.
  __shared__ float dsum[kHeadDim / 32];
  __shared__ unsigned wmax[kHeadDim / 32];
  __nv_bfloat16* const pl = dco_planes + (size_t)hd * coarse_tc::kPlanes * rpad * 64;
  float tot = 0.f;
  if (tid < kHeadDim) {
#pragma unroll
    for (int i = 0; i < kRowGroups; ++i) tot = coarse_tc::add_nf(tot, red[i][tid]);
    float pd = coarse_tc::mul_nf(tot, oc[((size_t)hd * g.N + b) * kHeadDim + tid]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) pd = coarse_tc::add_nf(pd, __shfl_xor_sync(0xffffffffu, pd, o));
    const unsigned m = __reduce_max_sync(0xffffffffu, __float_as_uint(tot) & 0x7fffffffu);
    if ((tid & 31) == 0) {
      dsum[tid >> 5] = pd;
      wmax[tid >> 5] = m;
    }
  }
  __syncthreads();
  if (tid < kHeadDim) {
    const int l = coarse_tc::lift_exp(__uint_as_float(max(max(wmax[0], wmax[1]), max(wmax[2], wmax[3]))));
    uint32_t w[3];
    coarse_tc::split3_x2(coarse_tc::lift(tot, l), 0.f, w);
    const int e = tid & 63, pos = ((((e >> 3) ^ b) & 7) << 3) | (e & 7);
#pragma unroll
    for (int k = 0; k < 3; ++k)
      pl[((size_t)(2 * k + (tid >> 6)) * rpad + b) * 64 + pos] = __ushort_as_bfloat16((unsigned short)(w[k] & 0xffffu));
    if (tid == 0) {
      dco_lift[(size_t)hd * g.N + b] = l;
      dc[(size_t)hd * ((g.N + 31) / 32 * 32) + b] =
          coarse_tc::add_nf(coarse_tc::add_nf(dsum[0], dsum[1]), coarse_tc::add_nf(dsum[2], dsum[3]));
    }
  } else if (tid < kHeadDim + coarse_tc::kPlanes * 16) {
    // The planes' padding rows N + b, N + b + N, ... below rpad: 6 planes x 16 pieces of 8 bytes per row.
    const int u = tid - kHeadDim, pp = u >> 4, c = u & 15;
    for (int r = g.N + b; r < rpad; r += g.N)
      *reinterpret_cast<uint2*>(pl + ((size_t)pp * rpad + r) * 64 + c * 4) = make_uint2(0u, 0u);
  }
}

// (kPostThreads, kPostCtas): 128 threads, and the second argument is a register BUDGET for ptxas, not a residency
// cap. Measured on this kernel at 128 threads per CTA (`cuobjdump -res-usage`, job cited per row):
//   128-row tile, fp32 stage              REG:162  J1964   3 CTAs = 12 warps/SM   38.2 us s8 / 77.5 s20
//   128-row tile, bf16 tile              REG:168  J6889   = the (128,3) budget exactly, still 3 CTAs   34.656 / 72.992
//    64-row tile, bf16 tile (this)     REG:120  J7050   4 CTAs = 16 warps/SM   28.384 / 59.520
// A fourth CTA at 128 threads needs <= 128 registers per thread: 162 x 128 x 4 = 82,944 and 168 x 128 x 4 = 86,016
// both exceed 65,536, while 120 x 128 x 4 = 61,440 fits (and 4 x 17,408 B = 69,632 B fits the 228 KB). So the halved
// tile is what admitted the fourth CTA, and the CTA count, the tile and the residency move together here.
// Higher residency at the 128-row tile was tried once and lost -- 48 warps over 6 CTAs at REG:40, 39.1 us s8 / 82.1
// s20 (J6865) -- but that arm changed warps per CTA as well as residency and predates the halved tile, so it does not
// price the 4-CTA / 64-row point.
// J6844: a transpose-free stream of these bytes at 12 warps/SM takes 37.1 us median, so the pre-split kernel's s8 time
// was already the floor OF ITS OCCUPANCY.
//
// THE TILE: bf16 with the conversion on the way IN, which deletes the dq_packed[kHeadDim/2] accumulator (64 of main's
// 162 registers) -- a thread holds 4 dims of a row, not 128. It was 32,768 B over 128 rows (J6889: 34.656 s8 / 72.992
// s20) and it is 16,384 B over the 64 rows one CTA now owns, because the row split below halved it.
// THE PIPELINE: two halves, and the middle stage loads half B into smem WHILE it stores half A out, so both memory
// directions are in flight -- main had a read-only phase and a write-only phase behind one barrier (J6902: 32.512 us s8
// / 67.872 s20, against 34.656 / 72.992 without it, on the 128-row tile). Within the tile the halves are disjoint and
// each one's rows are contiguous inside a drain group, which is what makes the middle stage safe and cheap.
// WHAT THE ROW SPLIT COSTS THE READ: a drain
// group is ONE float4 per row (`bwd.cuh`'s [32/4][128 rows][4]), so a group is 128 float4 = 2,048 B. A 64-row CTA now
// reads 64 of those 128 float4 = a 1,024 B run per group at a 2,048 B stride, i.e. 32 fragments of 1 KB, where the
// pre-split 128-row CTA read the whole 64 KB contiguously. THE COALESCING IS UNCHANGED AND NEEDS NO ASSUMPTION ABOUT
// SCHEDULING: within a run the 64 rows are consecutive float4 and consecutive threads take consecutive rows of one
// 4-dim group, so a warp's 32 threads cover 512 B contiguous, and 1 KB is eight whole 128 B cache lines -- every line
// the CTA asks for is fully used, at any residency, in any order. That the sibling CTA (blockIdx.x ^ 1) reads the
// complementary 1,024 B of each run is why the STREAM is whole end to end, which is the weaker secondary point: it
// says nothing about this CTA's efficiency and does not assume the two are ever resident together.
// THE ROW SPLIT: 28.384 us s8 / 59.520 s20 (J7050, all five workloads in one green run), -12.0 % / -12.2 % against
// 32.256 / 67.808 (J6917 -- the same 128-row tile, one job over all five workloads; it printed timings only, so what
// identifies the baseline kernel is the column pair, not a tag in that job).
__global__ void __launch_bounds__(kPostThreads, kPostCtas)
    bwd_post_kernel(const float* __restrict__ dqaccum, const float* __restrict__ dq_coarse,
                    const uint32_t* __restrict__ rowmask, __nv_bfloat16* __restrict__ dq, int num_blocks,
                    long long seqlen, float sm_scale) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  // ROW SPLIT: one CTA owns HALF a block, so the grid is N * (kBlock/kPostRows) in x -- 2 x g.N CTAs at the pinned
  // kPostRows = 64, i.e. twice main's -- and each CTA's tile is kPostRows rows. The hypothesis is w26's: the achievable
  // stream rate is a function of the CTA COUNT, not of per-CTA work, and a 1,576-CTA grid sits at 3.46 waves, the worst
  // fill cell of that curve. The half and the block below are DERIVED from kPostRows rather than written as `>> 1` and
  // `& 1`, so the launch's `g.N * (kBlock/kPostRows)` and this split cannot drift apart; the expressions constant-fold
  // to the shifts at the pinned value.
  constexpr int kTilesPerBlock = kBlock / kPostRows;
  __shared__ __align__(16) __nv_bfloat16 post_tile[kPostRows * kHeadDim];  // 16,384 B, bf16, swizzled
  const int tile = (int)blockIdx.x, head = (int)blockIdx.y;
  const int q_block = tile / kTilesPerBlock, row_base = (tile % kTilesPerBlock) * kPostRows;
  const int thread = (int)threadIdx.x;
  constexpr int kQuadPerSlice = kDqCols / 4;                      // 8 groups of 4 dims per 32-dim slice
  constexpr int kGroups = (kHeadDim / kDqCols) * kQuadPerSlice;   // 32 (slice, v4) groups
  constexpr int kHalfRows = kPostRows / 2;
  constexpr int kQuadsHalf = kGroups * kHalfRows;
  constexpr int kRowsPerPass = kPostThreads / 16;
  // The two macros' loop bounds are exact divisions, and a change to kPostThreads or kHeadDim must fail HERE rather
  // than silently drop a quarter of the tile: the loader assumes kQuadsHalf % kPostThreads == 0, the store assumes
  // kHalfRows % kRowsPerPass == 0, and 16 threads per row assumes kPostThreads % 16 == 0.
  static_assert(kPostThreads % 16 == 0, "16 threads per row");
  static_assert(kQuadsHalf % kPostThreads == 0, "the load's quad count must divide the block");
  static_assert(kHalfRows % kRowsPerPass == 0, "the store's row count must divide the passes");
  // AND the store's row coverage, which the three above do NOT bound: the store writes 8 dims per lane over 16 lanes,
  // so its row width is exactly 16 * 8. With kHeadDim = 256 every assert above still passes -- at the shipped
  // kPostRows = 64 that is kGroups 64, kQuadsHalf 64 x 32 = 2,048 (2048 % 128 == 0) and kHalfRows 32 (32 % 8 == 0) --
  // while POST_STORE_HALF covers dims 0-127 only: the same silent half-row loss this block exists to stop, one level
  // out. kHeadDim = 64 is the mirror (lanes write past the row).
  static_assert(kHeadDim == 16 * 8, "16 lanes of 8 dims must fill one row exactly");
  // AND the coverage of the block itself, which none of the above bounds. Because BOTH the launch and the split derive
  // from kPostRows (kTilesPerBlock = kBlock / kPostRows, row_base = (tile % kTilesPerBlock) * kPostRows), the only way
  // a row is lost is a kPostRows that does not DIVIDE kBlock: at 48, kTilesPerBlock truncates to 2, row_base can only
  // be 0 or 48, and rows 96-127 of every block are never written -- into an empty_like buffer, with the build green
  // and every assertion above silent (J7067: 48 fails on this line with this message, 32 and 128 build).
  // SHIPPING POLICY, not an invariant: this kernel ships kPostRows = 64, one CTA per half block. At kBlock = 128 and
  // kPostThreads = 128 the values that BUILD are 16, 32, 64 and 128; the smaller divisors (1, 2, 4, 8) are each
  // stopped at build time by an assert in this kernel (some above this line, some below). WHICH assert stops which is
  // deliberately NOT restated here: it is a property of kPostThreads as well, so the list would go stale the moment
  // either knob moves, and a wrong reason is worse than none. Change a knob and read the compile error.
  // So this assertion bounds what is WRONG (the tail rows), never what is chosen.
  static_assert(kBlock % kPostRows == 0, "a kPostRows that does not divide the block leaves the tail rows unwritten");
  static_assert(kPostRows % 2 == 0, "the two halves must split the CTA's rows evenly");
  const int row_in_pass = thread >> 4;
  const int dimension_begin = (thread & 15) * 8;

  const float4* src = reinterpret_cast<const float4*>(dqaccum + ((size_t)head * seqlen + (size_t)q_block * kBlock) * kHeadDim);
  const float* crow = dq_coarse + ((size_t)head * num_blocks + q_block) * kHeadDim;

  // Consecutive threads take consecutive rows of ONE 4-dim group: a warp covers 512 contiguous bytes and shares one
  // `c`, which makes the coarse row a broadcast rather than a load.
#define POST_LOAD_HALF(H)                                                                          \
  _Pragma("unroll") for (int it = 0; it < kQuadsHalf / kPostThreads; ++it) {                        \
    const int i = it * kPostThreads + thread;                                                      \
    const int g = i / kHalfRows, row = (H) * kHalfRows + i % kHalfRows;                             \
    const int c = (g / kQuadPerSlice) * kDqCols + (g % kQuadPerSlice) * 4;                          \
    const float4 value = *reinterpret_cast<const float4*>(src + g * kBlock + row_base + row);       \
    const float4 cv = __ldg(reinterpret_cast<const float4*>(crow + c));                             \
    const bool ok = mask_bit(rowmask, q_block, row_base + row);                                     \
    const int hh = c >> 6, chunk = ((c & 63) >> 3) ^ (row & 7);                                     \
    const int pos = row * kHeadDim + hh * 64 + chunk * 8 + (c & 7);                                 \
    *reinterpret_cast<uint2*>(post_tile + pos) =                                                    \
        ok ? make_uint2(cvt_f32x2_to_bf16x2(value.x * sm_scale + cv.x, value.y * sm_scale + cv.y),    \
                        cvt_f32x2_to_bf16x2(value.z * sm_scale + cv.z, value.w * sm_scale + cv.w))   \
           : make_uint2(0u, 0u);                                                                    \
  }

  // Out row-major, 8 dims (16 B) per thread: 16 threads hold one row, so one pass covers kPostThreads/16 rows.
#define POST_STORE_HALF(H)                                                                          \
  _Pragma("unroll") for (int rp = 0; rp < kHalfRows / kRowsPerPass; ++rp) {                          \
    const int row = (H) * kHalfRows + rp * kRowsPerPass + row_in_pass;                               \
    const int hh = dimension_begin >> 6, chunk = ((dimension_begin & 63) >> 3) ^ (row & 7);          \
    const uint4 value = *reinterpret_cast<const uint4*>(post_tile + row * kHeadDim + hh * 64 + chunk * 8); \
    const size_t element =                                                                           \
        ((size_t)head * seqlen + (size_t)q_block * kBlock + row_base + row) * kHeadDim + dimension_begin; \
    *reinterpret_cast<uint4*>(dq + element) = value;                                                  \
  }

  POST_LOAD_HALF(0)
  __syncthreads();
  // The middle stage needs NO barrier between its two halves because they touch DISJOINT smem: POST_LOAD_HALF(1)
  // writes only elements [kHalfRows*kHeadDim, kPostRows*kHeadDim) and POST_STORE_HALF(0) reads only
  // [0, kHalfRows*kHeadDim) -- half of a tile that is kPostRows*kHeadDim elements. Both reads are published by the
  // __syncthreads() either side. A change to the half split (three quarters, or a split on columns) makes these ranges
  // overlap and turns this stage into a race.
  POST_LOAD_HALF(1)   // the middle stage: half B in ...
  POST_STORE_HALF(0)  // ... and half A out, in the same stage
  __syncthreads();
  POST_STORE_HALF(1)
#undef POST_LOAD_HALF
#undef POST_STORE_HALF
}

}  // namespace glue
}  // namespace vsa2t
