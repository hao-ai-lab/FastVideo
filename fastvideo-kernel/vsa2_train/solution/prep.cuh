// prep.cuh -- one read of a block's rows: the block means of the coarse branch (fp32) and, in the forward, the window
// means of the router (fp16, to keep the top-K stable -- issue I23).
//
// A CTA owns one block of one head, 256 threads: thread (group, 8 dims), group = tid / 16 is router window `group` of
// a video tile (its 8 slots, eq. 5) or rows 8 group .. 8 group + 7 of a text block. The block mean is the sum of the
// 16 groups' sums over the block's tokens (shared memory).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "coarse_tc.cuh"
#include "common.cuh"

namespace vsa2t {
namespace prep {

constexpr int kThreads = 256;

// The 8 slots of group `grp` of block b, and which of them are tokens.
struct Group {
  int slot[8];
  uint32_t ok;  // bit j: slot[j] is a token
};

// The 8 slots of group `grp` of block b, and which of them are tokens. The block's tile coordinate (and, for a text
// block, its first token's index) comes in precomputed, because prep's original form called `row_valid` once per
// slot and `row_valid` calls `tile_coord`, whose two integer divisions are emulated instructions (about twenty each).
// Eight of them per thread, for a coordinate that is the same for every thread of the CTA, is what kept this pass at
// 3.5 TB/s: on prep's own load pattern, with everything else identical, the per-slot form measures 4.79 TB/s against
// 7.47 without it, and computing the coordinate once per thread measures 7.48 (lab/probes/prep_stream.cu, rows A/A8/A9).
// The validity test itself is three comparisons, so it is the divisions that cost, not the geometry.
__device__ __forceinline__ Group group_slots_at(const Geo& g, int b, int grp, const TileCoord& c, long long text_begin) {
  Group s;
  s.ok = 0;
  const bool video = b >= g.P;
  const int t0 = c.ti * 4, h0 = c.hi * 4, w0 = c.wi * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int slot;
    if (video) {
      const int lt = 2 * (grp >> 3) + (j >> 2), lh = 2 * ((grp >> 2) & 1) + ((j >> 1) & 1), lw = 2 * (grp & 3) + (j & 1);
      slot = lt * 32 + lh * 8 + lw;
      const bool in = t0 + (slot >> 5) < g.t && h0 + ((slot >> 3) & 3) < g.h && w0 + (slot & 7) < g.w;
      s.ok |= in ? (1u << j) : 0u;
    } else {
      slot = grp * 8 + j;
      s.ok |= (text_begin + slot < g.text) ? (1u << j) : 0u;
    }
    s.slot[j] = slot;
  }
  return s;
}

// The block's mean (sum / count, or the sum itself when count is 0) from the 16 groups' sums in red[16][128], written
// by threads 0..127 (one dim each).
// Each block mean (fp32) is written as three bf16 components with exponents of their own (coarse_tc.cuh split3),
// times a lift 2^l that only a row whose largest |entry| is below 2^-20 takes (lift_exp): mean * 2^l = b0 + b1 + b2,
// exact for every entry with |mean * 2^l| >= 2^-110. The coarse branch's tensor cores take the components as 16-bit
// operands and the eight products b_i * c_j with i + j <= 3 (exact to 2^-32).
// The components are laid out as the coarse kernels' shared-memory tiles, so that a tile is six bulk copies: per head
// six planes (b0 dims 0-63, b0 64-127, b1 0-63, b1 64-127, b2 0-63, b2 64-127) of `npad` rows of 128 bytes, row r's
// 16-byte chunk c at position c ^ (r & 7) (the MMA's 128-byte swizzle), and rows N..npad-1 zero.
struct SplitOut {
  __nv_bfloat16* planes;  // [heads][6][npad][64]
  int* e;                 // [heads][N] the rows' lifts l >= 0
  int npad;               // coarse_tc::plane_rows(N)
};
// All of the CTA's threads call it (one barrier, for the rows' maxima); threads 0..127 own one dim each, threads
// 128..255 zero block b's share of the padding rows, N + b, N + b + N, ... below npad.
__device__ __forceinline__ void block_means(const float (*red)[16][kDim], float scale, size_t row,
                                            const SplitOut (&split)[3], float (*wmax)[4], int N, int hd, int b) {
  const int t = (int)threadIdx.x;
  float v[3] = {0.f, 0.f, 0.f};
  if (t < kDim) {
#pragma unroll
    for (int x = 0; x < 3; ++x) {
      // Sums and the scale that keep subnormals (coarse_tc.cuh *_nf): a subnormal mean against a large counterpart
      // in the coarse branch's products is not negligible.
      float tot = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) tot = coarse_tc::add_nf(tot, red[x][i][t]);
      v[x] = coarse_tc::mul_nf(tot, scale);
      // The warp's largest |v|: a non-negative float orders as its bits, so one redux instead of five shuffles (and
      // integer operations throughout, which no flush-to-zero mode touches).
      const unsigned m = __reduce_max_sync(0xffffffffu, __float_as_uint(v[x]) & 0x7fffffffu);
      if ((t & 31) == 0) wmax[x][t >> 5] = __uint_as_float(m);
    }
  }
  __syncthreads();
  if (t < kDim) {
    const int e = t & 63, pos = ((((e >> 3) ^ b) & 7) << 3) | (e & 7);
#pragma unroll
    for (int x = 0; x < 3; ++x) {
      const unsigned mb = max(max(__float_as_uint(wmax[x][0]), __float_as_uint(wmax[x][1])),
                              max(__float_as_uint(wmax[x][2]), __float_as_uint(wmax[x][3])));
      const int re = coarse_tc::lift_exp(__uint_as_float(mb));
      uint32_t w[3];
      coarse_tc::split3_x2(coarse_tc::lift(v[x], re), 0.f, w);  // the lift is exact: a power of 2, no overflow
      const int np = split[x].npad;
      __nv_bfloat16* const pl = split[x].planes + (size_t)hd * 6 * np * 64;
#pragma unroll
      for (int k = 0; k < 3; ++k)
        pl[((size_t)(2 * k + (t >> 6)) * np + b) * 64 + pos] = __ushort_as_bfloat16((unsigned short)(w[k] & 0xffffu));
      if (t == 0) split[x].e[row] = re;
    }
  } else {
    // The planes' padding rows: 64 bf16 (128 bytes) of each of the 3 x 6 planes per row, 8 bytes per thread.
    const int u = t - kDim;  // 0..127
    for (int r = N + b; r < split[0].npad; r += N) {
#pragma unroll
      for (int x = 0; x < 3; ++x) {
        __nv_bfloat16* const pl = split[x].planes + (size_t)hd * 6 * split[x].npad * 64;
        if (u < 6 * 16) {
          const int p = u >> 4, c = u & 15;  // plane, 8-byte piece of the row
          *reinterpret_cast<uint2*>(pl + ((size_t)p * split[x].npad + r) * 64 + c * 4) = make_uint2(0u, 0u);
        }
      }
    }
  }
}

// The block's row mask, its window bits (video tiles) and this CTA's share of the zeroing -- **without shared
// memory, atomics or a barrier**, because both masks are pure functions of the tile coordinate the kernel already
// holds: row r of the block is valid iff its token lies inside the latent (video) or before the text's end (text),
// and a 2x2x2 window holds a token iff its lowest corner does, since the three extents only ever cut the tile from
// the high side. So 128 threads compute one row bit each and their four warps' ballots are the block's four words,
// and warp 4's lanes 0..15 do the sixteen window bits the same way. The version this replaces had each group's
// `dc == 0` thread issue eight shared `atomicOr`s into five hot words and then a `__syncthreads()` that held all
// 256 threads until they drained; the clear of the router's scratch still needs nothing.
__device__ __forceinline__ void tables_and_masks(const Geo& g, int b, int hd, const TileCoord& c, long long text_begin,
                                                 uint32_t* rowmask, uint16_t* winmask, uint32_t* zero,
                                                 long long zero_words) {
  const int t = (int)threadIdx.x;
  const long long ctas = (long long)gridDim.x * gridDim.y, cta = (long long)blockIdx.y * gridDim.x + blockIdx.x;
  for (long long i = cta * blockDim.x + t; i < zero_words; i += ctas * blockDim.x) zero[i] = 0u;
  if (hd != 0) return;
  if (t < kRows) {
    const bool ok = (b >= g.P) ? (c.ti * 4 + (t >> 5) < g.t && c.hi * 4 + ((t >> 3) & 3) < g.h &&
                                  c.wi * 8 + (t & 7) < g.w)
                              : (text_begin + t < g.text);
    const uint32_t m = __ballot_sync(0xffffffffu, ok);
    if ((t & 31) == 0) rowmask[(size_t)b * 4 + (t >> 5)] = m;
  } else if (t < kRows + 32 && winmask != nullptr && b >= g.P) {
    const int w = t - kRows;  // warp 4: lane 0..15 is the tile's window 0..15
    const bool ok = w < kWindows && c.ti * 4 + 2 * (w >> 3) < g.t && c.hi * 4 + 2 * ((w >> 2) & 1) < g.h &&
                    c.wi * 8 + 2 * (w & 3) < g.w;
    const uint32_t m = __ballot_sync(0xffffffffu, ok);
    if ((t & 31) == 0) winmask[b - g.P] = (uint16_t)m;
  }
}

struct FwdArgs {
  Geo g;
  const __nv_bfloat16* q;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  SplitOut sq, sk, sv;  // the block means, as three bf16 components and a row lift (block_means)
  __half* qw;           // [heads][Nv * 16][128] window means of the video tiles (0 for a window without a token)
  __half* kw;
  float* qnorm;         // [heads][Nv * 16] the norm of each (fp16) query window mean, 0 without a token
  float* knorm;         // [heads][Nv * 16] likewise for the keys
  uint32_t* rowmask;    // [N][4] (common.cuh), written by head 0's CTAs
  uint16_t* winmask;    // [Nv]
  uint32_t* zero;       // zeroed by the grid: the top-K's and the order's histograms
  long long zero_words;
};

// One CTA per (block, head): 256 threads = 16 window groups x 16 threads holding 8 dims of the head each. The three
// tensors share the group's eight row addresses, so one thread accumulates all three (12 float2 accumulators) and
// each address calculation feeds three loads. What this pass needed was not a different load path but the geometry
// hoisted out of the slot loop: ncu at s8 (J192) put prep at 3.5 TB/s of DRAM, 55% issue, its 2 resident CTAs
// stalled on ALU dependencies (Stall Wait 1.57, Long Scoreboard 0.73), and lab/probes/prep_stream.cu showed the same
// pattern reaching 7.5 TB/s once the per-slot `tile_coord` -- two emulated integer divisions, sixteen times per
// thread, for a coordinate identical across the CTA -- became one coordinate per thread. Two rewrites that attacked
// the load path instead bought nothing: staging the block with cp.async (96 KB, still 2 CTAs) measured no change at
// all (J256: 40.7/36.2 us at s8, 89.4/82.7 at s20 against 39.6/36.6 and 84.9/88.5), and one tensor per CTA over the
// grid's z was worse by 22% at s8 and 52% at s20 (J345).

// The rows of the group, summed in fp32, with the 8 addresses **constant offsets** from the group's first row.
// Group::slots are a fixed pattern: a text block's slot j is grp * 8 + j (delta j), a video tile's is
// 2lt * 32 + 2lh * 8 + 2lw + (lt bit)*32 + (lh bit)*8 + (lw bit) (delta 0,1,8,9,32,33,40,41), both independent of the
// group and the block. Walking them from one base pointer makes every load `LDG [R+imm]` instead of a 64-bit
// multiply-add per load, which is what ncu's ALU pipe (44.5% against 29.6% DRAM, J192) is spending its time on.
// What is pinned here and what is not -- stated exactly, because the assertions below are weaker than they read:
// they compare this table against the two `constexpr` copies that follow, and **nothing in device code reads any of
// the three**, since a namespace-scope `constexpr` array is not device-visible without `--expt-relaxed-constexpr`.
// So `group_slots_at` spells the tile formula inline and `group_sum_c` spells the delta inline, and the geometry
// lives in four places: those two expressions, this table, and the two copies. The assertions catch a change to the
// copies and cannot catch a change to the inline expressions; a change to the tile, the window or the slot order has
// to touch all of them. (Making it automatic is one edit away -- mark `video_slot` and `slot_delta`
// `__host__ __device__ constexpr` and call them from both sites -- and it is deliberately not in this change, which
// moves no device code: two seats have verified this tree numerically and a pin is not worth invalidating that for.)
constexpr int kSlotDelta[8] = {0, 1, 8, 9, 32, 33, 40, 41};

// group_slots' video-tile formula, spelled out so the assertions below read as the same derivation.
constexpr int video_slot(int grp, int j) {
  const int lt = 2 * (grp >> 3) + (j >> 2), lh = 2 * ((grp >> 2) & 1) + ((j >> 1) & 1), lw = 2 * (grp & 3) + (j & 1);
  return lt * 32 + lh * 8 + lw;
}
// The expression the loader below uses (host-side copy, for the assertions only: the device code spells it inline).
constexpr int slot_delta(int j) { return (j >> 2) * 32 + ((j >> 1) & 1) * 8 + (j & 1); }
constexpr bool slot_deltas_hold() {
  for (int j = 0; j < 8; ++j)
    if (slot_delta(j) != kSlotDelta[j]) return false;
  for (int grp = 0; grp < kWindows; ++grp)
    for (int j = 0; j < 8; ++j)
      if (video_slot(grp, j) - video_slot(grp, 0) != slot_delta(j)) return false;
  return true;  // a text block's slots are grp * 8 + j, whose delta is j: the loop index itself
}
static_assert(kRows == 128 && kDim == 128 && kWindows == 16, "the tiles, the windows and the head dim are fixed");
static_assert(slot_deltas_hold(), "kSlotDelta must be group_slots' slot[j] - slot[0] for a video tile");

// A padding row is zero in q, k and v (the spec's layout guarantees it) and adding 0.0f to an fp32 accumulator
// changes nothing, so a caller that does not need the loads skipped can pass `ok = ~0u` and ptxas folds the predicate
// away: measured at bwd_prep, that is 9% off that pass at s8 and 4% at s20 (J1053 against J1097, clock-normalised),
// while the same change costs fwd_prep 9-10%, which has other work to overlap them with. So the predicate stays and
// the choice is made per kernel at the call site.
template <bool VIDEO>
__device__ __forceinline__ void group_sum_c(const uint4* __restrict__ base, unsigned ok, float (&acc)[8]) {
  float2 a[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) a[e] = make_float2(0.f, 0.f);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int dlt = VIDEO ? ((j >> 2) * 32 + ((j >> 1) & 1) * 8 + (j & 1)) : j;
    if (ok & (1u << j)) {
      const uint4 v = __ldg(base + dlt * (kDim / 8));
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
      for (int e = 0; e < 4; ++e) a[e] = fadd2(a[e], __bfloat1622float2(p[e]));
    }
  }
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    acc[2 * e] = a[e].x;
    acc[2 * e + 1] = a[e].y;
  }
}

// Forward: block means of q, k, v; window means of q, k on video tiles. Grid (N, heads); thread (group, 8 dims).
__global__ void __launch_bounds__(kThreads) fwd_prep_kernel(FwdArgs a) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  __shared__ float red[3][16][kDim];
  const Geo& g = a.g;
  const int b = blockIdx.x, hd = blockIdx.y;
  const int dc = threadIdx.x & 15, grp = threadIdx.x >> 4;
  // One tile coordinate for the block, not one per slot (see group_slots_at).
  // Zero-initialised: `group_slots_at` derives t0/h0/w0 from it before the branch, so a text block would otherwise
  // multiply three indeterminate ints (the values are only used inside the video branch, but the form is undefined).
  TileCoord c{};
  long long text_begin = 0;
  if (b >= g.P) c = tile_coord(g, b - g.P); else text_begin = (long long)b * kRows;
  const Group s = group_slots_at(g, b, grp, c, text_begin);
  const int n = __popc(s.ok);
  const size_t row0 = ((size_t)hd * g.S + (size_t)b * kRows) * kDim + (size_t)s.slot[0] * kDim + dc * 8;
  float sum[3][8];
  if (b >= g.P) {
    group_sum_c<true>(reinterpret_cast<const uint4*>(a.q + row0), s.ok, sum[0]);
    group_sum_c<true>(reinterpret_cast<const uint4*>(a.k + row0), s.ok, sum[1]);
    group_sum_c<true>(reinterpret_cast<const uint4*>(a.v + row0), s.ok, sum[2]);
  } else {
    group_sum_c<false>(reinterpret_cast<const uint4*>(a.q + row0), s.ok, sum[0]);
    group_sum_c<false>(reinterpret_cast<const uint4*>(a.k + row0), s.ok, sum[1]);
    group_sum_c<false>(reinterpret_cast<const uint4*>(a.v + row0), s.ok, sum[2]);
  }
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    if (b >= g.P) {
      const float inv = n > 0 ? 1.f / (float)n : 0.f;
      uint4 packed;
      uint32_t* pw = reinterpret_cast<uint32_t*>(&packed);
      float sq = 0.f;  // the squared norm of the rounded mean, which the router's logits are products of
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const __half2 h2 = __floats2half2_rn(sum[x][2 * e] * inv, sum[x][2 * e + 1] * inv);
        pw[e] = *reinterpret_cast<const uint32_t*>(&h2);
        const float2 f = __half22float2(h2);
        sq += f.x * f.x + f.y * f.y;
      }
      const size_t win = ((size_t)hd * g.Nv + (b - g.P)) * kWindows + grp;
      reinterpret_cast<uint4*>((x == 0 ? a.qw : a.kw) + win * kDim)[dc] = packed;
      // The group's 16 threads are a half-warp.
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
      if (dc == 0) (x == 0 ? a.qnorm : a.knorm)[win] = sqrtf(sq);
    }
  }
  tables_and_masks(g, b, hd, c, text_begin, a.rowmask, a.winmask, a.zero, a.zero_words);
#pragma unroll
  for (int x = 0; x < 3; ++x) {
#pragma unroll
    for (int e = 0; e < 8; ++e) red[x][grp][dc * 8 + e] = sum[x][e];
  }
  __syncthreads();
  __shared__ float wmax[3][4];
  const float inv = 1.f / (float)block_count(g, b);
  const size_t row = (size_t)hd * g.N + b;
  const SplitOut sp[3] = {a.sq, a.sk, a.sv};
  block_means(red, inv, row, sp, wmax, g.N, hd, b);
}

struct BwdArgs {
  Geo g;
  const __nv_bfloat16* q;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  SplitOut sq, sk, sv;  // the block means, as three bf16 components and a row lift (block_means)
  uint32_t* rowmask;  // [N][4], written by head 0's CTAs
  float* rcnt;        // [N] 1 / the block's tokens
  uint32_t* zero;     // zeroed by the grid: the order's histogram
  long long zero_words;
};

// Backward: the block means again (the forward's are gone). Grid (N, heads). The coarse branch's output gradient
// (the block's sum of gate * dout) is formed by glue.bwd_rows_kernel, which reads gate and dout anyway for D and
// dgate: one pass over them instead of two.
__global__ void __launch_bounds__(kThreads) bwd_prep_kernel(BwdArgs a) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  __shared__ float red[3][16][kDim];
  const Geo& g = a.g;
  const int b = blockIdx.x, hd = blockIdx.y;
  const int dc = threadIdx.x & 15, grp = threadIdx.x >> 4;
  // One tile coordinate for the block, not one per slot (see group_slots_at).
  // Zero-initialised, as in fwd_prep_kernel: `group_slots_at` reads ti/hi/wi before the branch.
  TileCoord c{};
  long long text_begin = 0;
  if (b >= g.P) c = tile_coord(g, b - g.P); else text_begin = (long long)b * kRows;
  const Group s = group_slots_at(g, b, grp, c, text_begin);
  const size_t row0 = ((size_t)hd * g.S + (size_t)b * kRows) * kDim + (size_t)s.slot[0] * kDim + dc * 8;
  float sum[3][8];
  // `~0u` for the token mask: bwd_prep wants every slot loaded and the padding rows are zero, which ptxas folds
  // into unpredicated loads (see group_sum_c). The block's token count still comes from `s`. The mask is in bounds
  // for both slot patterns -- a video tile's largest slot is 3*32 + 3*8 + 7 = 127 and a text block's is
  // grp * 8 + 7 <= 127, against kRows = 128 -- and that is the invariant which licenses the unpredicated form: a
  // future slot pattern must keep slot[7] < kRows or these loads read into the next block.
  // The licence has a second half, and it is a value claim rather than a bounds one: dropping the predicate is
  // neutral only because a padding row is ZERO in q, k and v (the spec's guarantee), so the extra rows add 0.0f.
  // A caller with non-zero padding would get the wrong block mean here and the right one from fwd_prep -- the two
  // passes would then disagree about the same block with nothing failing.
  if (b >= g.P) {
    group_sum_c<true>(reinterpret_cast<const uint4*>(a.q + row0), ~0u, sum[0]);
    group_sum_c<true>(reinterpret_cast<const uint4*>(a.k + row0), ~0u, sum[1]);
    group_sum_c<true>(reinterpret_cast<const uint4*>(a.v + row0), ~0u, sum[2]);
  } else {
    group_sum_c<false>(reinterpret_cast<const uint4*>(a.q + row0), ~0u, sum[0]);
    group_sum_c<false>(reinterpret_cast<const uint4*>(a.k + row0), ~0u, sum[1]);
    group_sum_c<false>(reinterpret_cast<const uint4*>(a.v + row0), ~0u, sum[2]);
  }
  tables_and_masks(g, b, hd, c, text_begin, a.rowmask, nullptr, a.zero, a.zero_words);
#pragma unroll
  for (int x = 0; x < 3; ++x) {
#pragma unroll
    for (int e = 0; e < 8; ++e) red[x][grp][dc * 8 + e] = sum[x][e];
  }
  __syncthreads();
  const float inv = 1.f / (float)block_count(g, b);
  if (hd == 0 && threadIdx.x == 0) a.rcnt[b] = inv;
  __shared__ float wmax[3][4];
  const size_t row = (size_t)hd * g.N + b;
  const SplitOut sp[3] = {a.sq, a.sk, a.sv};
  block_means(red, inv, row, sp, wmax, g.N, hd, b);
}

}  // namespace prep
}  // namespace vsa2t
