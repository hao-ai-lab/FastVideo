// coarse_tc.cuh -- the coarse branch's attention on tcgen05, with 16-bit operands that hold the fp32 block means as
// three bf16 components each: Oc = softmax(qc kc^T * scale) vc, the rows' log2-sum-exp2 for the backward, and the
// backward itself (dqc, dkc, dvc).
//
// Why: the coarse means feed an fp32 reference (reference.py, emulate.py), and one rounding of them to any 16-bit or
// tf32 format can fail a valid input outright ([I84]; lab/verify/coarse_precision.py). So every operand value is held
// as three bf16 components with exponents of their own, x = b0 + b1 + b2 (split3_x2): b0 = x rounded toward zero
// (never past bf16's largest finite), b1 the remainder rounded to nearest, b2 what is left, rounded to nearest; the
// subtractions are exact. bf16's exponent range is fp32's, so an entry with |x| >= 2^-110 is held exactly; below
// that, to 2^-133 absolutely (bf16's least subnormal), and below 2^-133 not at all. One refinement keeps small rows
// exact: a block-mean row (q, k, v, dco) whose largest |entry| is below 2^-20 is lifted by 2^l, l >= 0, into
// [2^-20, 2^-19) (lift_exp), so every entry within 90 binades of such a row's largest is exact too. The lifts come
// back as factors 2^-l <= 1, on the logits or folded into P's and dS's columns (the A operand of the second product),
// so they never overflow where fp32 would not. Each product is the sum of the eight component products b_i c_j with
// i + j <= 3, smallest first, so the dropped b2 c2 is below 2^-32 |x||y|; the sums are fp32, in tensor memory.
// The arithmetic around the components keeps subnormals (the *_nf helpers) whatever the build's flush-to-zero mode
// (--use_fast_math), since a subnormal entry against a large counterpart is not negligible. What is not held, stated
// exactly: an entry below 2^-110 (after its row's lift) that is not a multiple of 2^-133 loses its bits below that,
// in a block mean, P or dS; the loss reaches a logit, an output or a gradient only through a counterpart of 2^90 or
// more on the other side of a product.
//
// One CTA: a 128-row tile (the MMA's M) against one split of the columns; a cluster of `ksplit` CTAs per tile sums
// or merges the splits' partials through global memory (each split writes its own; after the cluster's barrier each
// rank combines the rows it owns). 256 threads, all of them loading and in the softmax: warps w and w + 4 hold
// tensor-memory lanes 32 (w % 4) .. + 31 (32 rows of the tile) and take alternate 32-column chunks; warp 4 also
// issues the MMAs and holds the tensor memory's allocation. The forward and the backward's recompute share one body
// (vsa2t.cu).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace vsa2t {
namespace coarse_tc {

constexpr int kDim = 128;
constexpr int kTile = 128;      // rows per CTA = the MMA's M = tensor-memory lanes
constexpr int kPassKeys = 128;  // the forward's keys per pass: S's tensor-memory columns
constexpr int kThreads = 256;
constexpr int kMaxSplit = 8;    // the cluster: a portable size
constexpr int kComps = 3;       // bf16 components per value
constexpr int kPlanes = 2 * kComps;  // a row's planes: component x dims 0-63 / 64-127, 128 bytes each
constexpr int kProducts = 8;    // component pairs (i, j) with i + j <= 3

constexpr uint32_t kQBytes = (uint32_t)kPlanes * kTile * 128u;       // a 128-row tile's six planes: 96 KB
constexpr uint32_t kKVBytes = (uint32_t)kPlanes * kPassKeys * 128u;  // K's or V's at 128 rows (or both at 64): 96 KB
constexpr uint32_t kMiscBytes = 8192;                                // the pass's column factors, barriers
constexpr uint32_t kFwdSmemBytes = 1024 + kQBytes + kKVBytes + kMiscBytes;
static_assert(kFwdSmemBytes <= 232448, "one CTA per SM");

// 2^e for e in [-126, 127].
__host__ __device__ __forceinline__ float exp2i(int e) {
  union {
    uint32_t u;
    float f;
  } x;
  x.u = (uint32_t)(e + 127) << 23;
  return x.f;
}
#ifdef __CUDACC__
// fp32 arithmetic that keeps subnormals whatever the build's flush-to-zero mode.
__device__ __forceinline__ float mul_nf(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float add_nf(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float sub_nf(float a, float b) {
  float r;
  asm("sub.rn.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fma_nf(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;\n" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float ex2_nf(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}
// The same on pairs (sm_100's packed fp32: one instruction for two lanes' worth, the same rounding per element).
__device__ __forceinline__ uint64_t f2_bits(float2 v) {
  return ((uint64_t)__float_as_uint(v.y) << 32) | (uint64_t)__float_as_uint(v.x);
}
__device__ __forceinline__ float2 f2_make(uint64_t b) {
  return make_float2(__uint_as_float((uint32_t)b), __uint_as_float((uint32_t)(b >> 32)));
}
__device__ __forceinline__ float2 mul2_nf(float2 a, float2 b) {
  uint64_t r;
  asm("mul.rn.f32x2 %0, %1, %2;\n" : "=l"(r) : "l"(f2_bits(a)), "l"(f2_bits(b)));
  return f2_make(r);
}
__device__ __forceinline__ float2 sub2_nf(float2 a, float2 b) {
  uint64_t r;
  asm("sub.rn.f32x2 %0, %1, %2;\n" : "=l"(r) : "l"(f2_bits(a)), "l"(f2_bits(b)));
  return f2_make(r);
}
__device__ __forceinline__ float2 fma2_nf(float2 a, float2 b, float2 c) {
  uint64_t r;
  asm("fma.rn.f32x2 %0, %1, %2, %3;\n" : "=l"(r) : "l"(f2_bits(a)), "l"(f2_bits(b)), "l"(f2_bits(c)));
  return f2_make(r);
}
// A block-mean row's lift from its largest |entry| m (m >= 0): l >= 0 with m * 2^l in [2^-20, 2^-19) when
// 0 < m < 2^-20, else 0. Integer tests, so that a subnormal m is not read as 0 under flush-to-zero.
__device__ __forceinline__ int lift_exp(float m) {
  const uint32_t b = __float_as_uint(m);
  if (b == 0u || b >= 0x35800000u) return 0;  // 0, or >= 2^-20 (or not finite)
  const int e = b >= 0x00800000u ? (int)(b >> 23) - 127 : (31 - __clz((int)b)) - 149;
  return -20 - e;  // <= 129
}
// x * 2^l for a lift l in [0, 129] (exact: the result is below 2^-19).
__device__ __forceinline__ float lift(float x, int l) {
  return mul_nf(mul_nf(x, exp2i(l >> 1)), exp2i(l - (l >> 1)));
}
// 2^-l for a lift l in [0, 149]: normal down to 2^-126, then subnormal.
__device__ __forceinline__ float unlift(int l) { return l <= 126 ? exp2i(-l) : __uint_as_float(1u << (149 - l)); }
// Two values' three bf16 components, as packed pairs: w[k] holds y0's component k in its low half and y1's in its
// high half (an MMA operand from tensor memory takes a column's low half first).
__device__ __forceinline__ void split3_x2(float y0, float y1, uint32_t (&w)[3]) {
  asm("cvt.rz.bf16x2.f32 %0, %1, %2;\n" : "=r"(w[0]) : "f"(y1), "f"(y0));
  const float r0 = sub_nf(y0, __uint_as_float(w[0] << 16)), r1 = sub_nf(y1, __uint_as_float(w[0] & 0xffff0000u));
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(w[1]) : "f"(r1), "f"(r0));
  const float s0 = sub_nf(r0, __uint_as_float(w[1] << 16)), s1 = sub_nf(r1, __uint_as_float(w[1] & 0xffff0000u));
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(w[2]) : "f"(s1), "f"(s0));
}
// The same on a pair, with packed subtractions.
__device__ __forceinline__ void split3_f2(float2 y, uint32_t (&w)[3]) {
  asm("cvt.rz.bf16x2.f32 %0, %1, %2;\n" : "=r"(w[0]) : "f"(y.y), "f"(y.x));
  const float2 r = sub2_nf(y, make_float2(__uint_as_float(w[0] << 16), __uint_as_float(w[0] & 0xffff0000u)));
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(w[1]) : "f"(r.y), "f"(r.x));
  const float2 s = sub2_nf(r, make_float2(__uint_as_float(w[1] << 16), __uint_as_float(w[1] & 0xffff0000u)));
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(w[2]) : "f"(s.y), "f"(s.x));
}
#endif

struct FwdArgs {
  const __nv_bfloat16* qp;  // [heads][6][rpad][64] the block means' component planes (prep.cuh SplitOut)
  const int* qe;            // [heads][N] the rows' lifts
  const __nv_bfloat16* kp;
  const int* ke;
  const __nv_bfloat16* vp;
  const int* ve;
  float* oc;     // [heads][N][128]
  float* lse2;   // [heads][npad] log2-sum-exp2 of the rows' scaled logits
  float* part;   // [heads][tiles][kMaxSplit][128][128] the splits' partial outputs (the merge)
  float* pstat;  // [heads][tiles][kMaxSplit][2][128] ... and their rows' m and l
  int N;
  int npad;
  int rpad;  // plane_rows(N)
  int tiles;
  int ksplit;
  float scale_log2;
};

inline size_t part_floats(int N, int heads) {
  return (size_t)heads * ((N + kTile - 1) / kTile) * kMaxSplit * kTile * kDim;
}
inline size_t pstat_floats(int N, int heads) {
  return (size_t)heads * ((N + kTile - 1) / kTile) * kMaxSplit * 2 * kTile;
}

// Rows per plane: whole 128-row tiles, and the 32-row rounding of a split that ends at N (which starts at a multiple
// of 8, split_keys) stays inside.
inline __host__ __device__ int plane_rows(int N) { return (N + kTile - 1) / kTile * kTile + 32; }
// Columns per split: a multiple of 8, so that every tile of columns starts on the planes' swizzle period.
inline __host__ __device__ int split_keys(int N, int ks) { return ((N + ks - 1) / ks + 7) / 8 * 8; }

__global__ void __launch_bounds__(kThreads, 1) fwd_coarse_tc_kernel(FwdArgs a);
__global__ void __launch_bounds__(kThreads, 1) bwd_coarse_tc_kernel(FwdArgs a);

// The coarse backward: one launch of three roles over the same split operands (plus dco's, glue.cu bwd_rows), with
// the recompute's lse2 and Dc = rowsum(dco * Oc):
//   P = exp2(qc kc^T * scale_log2 - lse2), dP = dco vc^T, dS = P (dP - Dc),
//   dqc = scale dS kc, dkc = scale dS^T qc, dvc = P^T dco, each row times its block's 1 / count.
// The dQ role: a 128-row query tile against a split of the keys: S = Q K^T, dP = dO V^T (dO's components in tensor
// memory), dQ += dS K. The dK role: a 128-row key tile against a split of the queries: S^T = K Q^T, dP^T = V dO^T (V's
// in tensor memory), dK += dS^T Q. The dV role: S^T = K Q^T, dV += P^T dO. Each takes its split's columns 64 at a
// time, with dS (or P) times the second product's column fold as three bf16 components in tensor memory, over S's and
// dP's own columns.
constexpr int kGradCols = 64;
constexpr uint32_t kRBytes = (uint32_t)kPlanes * kTile * 128u;      // the rows' tile: 96 KB
constexpr uint32_t kCBytes = (uint32_t)kPlanes * kGradCols * 128u;  // a column tile: 48 KB
constexpr uint32_t kGradSmemBytes = 1024 + kRBytes + 2 * kCBytes + kMiscBytes;
static_assert(kGradSmemBytes <= 232448, "one CTA per SM");

struct GradArgs {
  const __nv_bfloat16* qp;  // the means' planes and lifts (prep.cuh SplitOut)
  const int* qe;
  const __nv_bfloat16* kp;
  const int* ke;
  const __nv_bfloat16* vp;
  const int* ve;
  const __nv_bfloat16* op;  // dco's planes and lifts (glue.cu bwd_rows)
  const int* oe;
  const float* lse2;  // [heads][npad] the recompute's
  const float* dc;    // [heads][npad]
  const float* rcnt;  // [N] 1 / the block's tokens
  float* dqc;         // [heads][N][128], zeroed by bwd_rows: each split adds its share
  float* dkc;
  float* dvc;
  int N;
  int npad;
  int rpad;
  int tiles;
  int ksplit;
  float scale_log2;
  float scale;
};

__global__ void __launch_bounds__(kThreads, 1) bwd_coarse_grad_tc_kernel(GradArgs a);

// The backward's split of the columns (`groups` = roles x heads CTAs per tile and split, `fit` = co-resident CTAs):
// the least modeled time, in microseconds at the dev clock (J7374's phase stamps): waves of `fit` CTAs, each CTA a
// fixed 13 (its rows' loads, the merge) plus 3.5 per pass and 1.6 per 16-column step.
inline int grad_split(int N, int groups, int fit) {
  const int tiles = (N + kTile - 1) / kTile;
  if (fit < 1) fit = 1;
  int best = 1;
  double best_t = 1e30;
  for (int ks = 1; ks <= kMaxSplit; ++ks) {
    const int per = split_keys(N, ks);
    if ((ks - 1) * per >= N) continue;  // an empty split
    const long long ctas = (long long)tiles * groups * ks;
    const long long waves = (ctas + fit - 1) / fit;
    const int steps = ((per < N ? per : N) + 15) / 16, passes = (steps + 3) / 4;
    const double t = (double)waves * (13.0 + 3.5 * passes + 1.6 * steps);
    if (t < best_t - 1e-9) {
      best_t = t;
      best = ks;
    }
  }
  return best;
}

// The split of the columns: the fewest waves of co-resident clusters (`clusters[ks]`, measured by the caller with
// cudaOccupancyMaxActiveClusters), then the least time per CTA, modeled in microseconds at the dev clock as a fixed
// 5 (loads, merge) plus 0.06 per column (eight products each way and the softmax), plus 2 per further pass.
inline int fwd_split(int N, int heads, const int (&clusters)[kMaxSplit + 1]) {
  const int tiles = (N + kTile - 1) / kTile;
  int best = 1;
  double best_t = 1e30;
  for (int ks = 1; ks <= kMaxSplit; ++ks) {
    const int per = split_keys(N, ks);
    if ((ks - 1) * per >= N) continue;  // an empty split
    const long long cl = (long long)tiles * heads, fit = clusters[ks] > 0 ? clusters[ks] : 1;
    const long long waves = (cl + fit - 1) / fit;
    const int passes = (per + kPassKeys - 1) / kPassKeys;
    const double t = (double)waves * (5.0 + 0.06 * per + 2.0 * (passes - 1));
    if (t < best_t - 1e-9) {
      best_t = t;
      best = ks;
    }
  }
  return best;
}

}  // namespace coarse_tc
}  // namespace vsa2t
