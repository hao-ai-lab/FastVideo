// norm.cu — RMS norm + SiLU + pad for the Wan2.2 VAE decoder, one warp per position.
//
// This replaces the seed's norm_vector<C> / pad_vector pair in kernels.cu with the same
// call contract (norm_pad(x, gamma, hist, pt, ph, silu)) and the same layout (NDHWC,
// [T+pt][H+2ph][W][C], history frames copied through, pad rows zeroed).
//
// Why it is faster (measured in probe/norm_probe2.cu and norm_probe3.cu, one GB200):
//  - the seed is XU-pipe bound (w08's and w14's ncu: 78% SM busy, 16% DRAM, XU 75%):
//    per element it pays cvt bf16->f32 for the sum, five cvts for the reference's three
//    bf16 roundings, __expf and an IEEE divide. This does
//      * bf16 -> f32 by an integer shift (ALU pipe, exact, no XU op),
//      * one rounding for the whole normalize*sqrt(C)*gain product instead of three,
//        which is also closer to the fp32 reference the decode is graded against,
//      * SiLU as u + u*tanh(u) with u = 0.5*pre-activation (one tanh.approx.f32),
//      * bf16 stores two elements at a time (cvt.rn.bf16x2.f32),
//    i.e. ~1.5 XU ops per element instead of ~11.
//  - rows are blocked so the per-position integer divisions of the seed's flat-pixel loop
//    happen once per row, not once per position (at C=256 a position is only 8 elements).
//  - P positions per warp in flight, the number that measured best per channel count.
//
// Numerics: one bf16 rounding for the whole normalize*gain product instead of three, and
// SiLU in fp32 through tanh.approx.f32. That approximation's documented worst case is an
// absolute error of about 2^-11 (5e-4) in tanh, which is harmless here because it enters
// the result as u*2^-11 and the stored value is bf16, so the perturbation stays below the
// bf16 quantum of u + u*tanh(u) over the range these pre-activations take. Measured, and
// the place to look if that ever changes (bench.py, f121x1 vs the fp32 reference):
//   seed   max 0.1709 (0.34x budget) rms 0.002242 (0.38x budget)
//   this   max 0.1992 (0.40x budget) rms 0.002187 (0.37x budget)
// i.e. the max moves up a little, the rms moves down, both far inside the budget.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdlib>
#include <type_traits>

using B = __nv_bfloat16;

namespace {

__device__ __forceinline__ void split2(uint32_t w, float& lo, float& hi) {
  lo = __uint_as_float(w << 16);      // exact bf16 -> f32, ALU pipe
  hi = __uint_as_float(w & 0xffff0000u);
}
__device__ __forceinline__ float tanh_approx(float x) {
  float r; asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r;
}

// MODE 0: RMS norm * sqrt(C) * gain, then SiLU.  MODE 1: the same, no SiLU.  MODE 2: copy only.
template <int C, int P, int MODE>
__global__ void norm_pad_kernel(const B* __restrict__ x, const B* __restrict__ gain,
                                const B* __restrict__ hist, B* __restrict__ y,
                                int T, int H, int W, int PT, int PH) {
  constexpr int CV = C / 8;      // uint4 per position
  constexpr int NP = C / 256;    // uint4 per lane per position
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
  const int HP = H + 2 * PH;
  const int64_t rows = (int64_t)(T + PT) * HP;
  const float sqrtC = sqrtf((float)C);
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int h = (int)(row % HP), t = (int)(row / HP);
    const bool interior = (h >= PH && h < H + PH);
    const bool current = (t >= PT);       // a history frame is copied through, never re-normalized
    const int ph = h - PH;
    const uint4* src = nullptr;
    if (interior) {
      if (current) src = (const uint4*)x + ((int64_t)(t - PT) * H + ph) * W * CV;
      else if (hist) src = (const uint4*)hist + ((int64_t)t * H + ph) * W * CV;
    }
    for (int w0 = warp * P; w0 < W; w0 += warps * P) {
      uint4 v[P][NP];
      #pragma unroll
      for (int q = 0; q < P; q++) {
        const int w = w0 + q;
        #pragma unroll
        for (int j = 0; j < NP; j++) {
          v[q][j] = make_uint4(0, 0, 0, 0);
          if (src && w < W) v[q][j] = src[(int64_t)w * CV + lane + j * 32];
        }
      }
      if (MODE < 2 && interior && current) {
        float s[P][2];
        #pragma unroll
        for (int q = 0; q < P; q++) { s[q][0] = 0.f; s[q][1] = 0.f; }
        #pragma unroll
        for (int q = 0; q < P; q++) {
          #pragma unroll
          for (int j = 0; j < NP; j++) {
            float f[8];
            split2(v[q][j].x, f[0], f[1]); split2(v[q][j].y, f[2], f[3]);
            split2(v[q][j].z, f[4], f[5]); split2(v[q][j].w, f[6], f[7]);
            #pragma unroll
            for (int k = 0; k < 8; k += 2) {
              s[q][0] = fmaf(f[k], f[k], s[q][0]);
              s[q][1] = fmaf(f[k + 1], f[k + 1], s[q][1]);
            }
          }
        }
        #pragma unroll
        for (int q = 0; q < P; q++) {
          float t0 = s[q][0] + s[q][1];
          #pragma unroll
          for (int d = 16; d; d >>= 1) t0 += __shfl_xor_sync(0xffffffff, t0, d);
          // 0.5 * sqrt(C) / max(sqrt(sum), 1e-12): one multiply per element, u = 0.5*pre-activation
          const float hs = 0.5f * sqrtC * rsqrtf(fmaxf(t0, 1e-24f));
          #pragma unroll
          for (int j = 0; j < NP; j++) {
            const uint4 gn = ((const uint4*)gain)[lane + j * 32];
            float g[8], xf[8], yf[8];
            split2(gn.x, g[0], g[1]); split2(gn.y, g[2], g[3]);
            split2(gn.z, g[4], g[5]); split2(gn.w, g[6], g[7]);
            split2(v[q][j].x, xf[0], xf[1]); split2(v[q][j].y, xf[2], xf[3]);
            split2(v[q][j].z, xf[4], xf[5]); split2(v[q][j].w, xf[6], xf[7]);
            #pragma unroll
            for (int k = 0; k < 8; k++) {
              const float u = xf[k] * hs * g[k];
              yf[k] = (MODE == 0) ? fmaf(u, tanh_approx(u), u) : (2.f * u);
            }
            v[q][j].x = pack2(yf[0], yf[1]); v[q][j].y = pack2(yf[2], yf[3]);
            v[q][j].z = pack2(yf[4], yf[5]); v[q][j].w = pack2(yf[6], yf[7]);
          }
        }
      }
      #pragma unroll
      for (int q = 0; q < P; q++) {
        const int w = w0 + q;
        if (w < W) {
          uint4* dst = (uint4*)y + row * W * CV + (int64_t)w * CV + lane;
          #pragma unroll
          for (int j = 0; j < NP; j++) dst[j * 32] = v[q][j];
        }
      }
    }
  }
}

// Copy + pad for any C divisible by 8: the calls with no gain (post_quant_conv's input, the temporal convs).
__global__ void copy_row_kernel(const uint4* __restrict__ x, const uint4* __restrict__ hist,
                                uint4* __restrict__ y, int T, int H, int W, int PT, int PH, int CV) {
  const int HP = H + 2 * PH;
  const int64_t rows = (int64_t)(T + PT) * HP;
  const int64_t nrow = (int64_t)W * CV;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int h = (int)(row % HP), t = (int)(row / HP);
    const bool interior = (h >= PH && h < H + PH);
    const uint4* src = nullptr;
    if (interior) {
      const int tt = t - PT, ph = h - PH;
      if (tt >= 0) src = x + ((int64_t)tt * H + ph) * nrow;
      else if (hist) src = hist + ((int64_t)t * H + ph) * nrow;
    }
    uint4* dst = y + row * nrow;
    const uint4 zz = make_uint4(0, 0, 0, 0);
    for (int64_t i = threadIdx.x; i < nrow; i += blockDim.x) dst[i] = src ? src[i] : zz;
  }
}

}  // namespace

at::Tensor norm_pad(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& hist,
                    int pt, int ph, bool silu) {
  const int C = (int)x.size(1), T = (int)x.size(2), H = (int)x.size(3), W = (int)x.size(4);
  auto y = at::empty({1, C, T + pt, H + ph * 2, W},
                     x.options().memory_format(at::MemoryFormat::ChannelsLast3d));
  const B* xp = (const B*)x.data_ptr<at::BFloat16>();
  const B* gp = gamma.defined() ? (const B*)gamma.data_ptr<at::BFloat16>() : nullptr;
  const B* hp = hist.defined() ? (const B*)hist.data_ptr<at::BFloat16>() : nullptr;
  B* yp = (B*)y.data_ptr<at::BFloat16>();
  const int64_t rows = (int64_t)(T + pt) * (H + ph * 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = (int)std::min<int64_t>(rows, 1024);
  const int threads = 256;
  auto go = [&](auto tag) {
    constexpr int CC = decltype(tag)::value;
    // P positions in flight per warp: measured best at 4 for C<=512, 2 at C=1024
    constexpr int P = (CC == 1024) ? 2 : 4;
    if (!gp) norm_pad_kernel<CC, P, 2><<<blocks, threads, 0, stream>>>(xp, gp, hp, yp, T, H, W, pt, ph);
    else if (silu) norm_pad_kernel<CC, P, 0><<<blocks, threads, 0, stream>>>(xp, gp, hp, yp, T, H, W, pt, ph);
    else norm_pad_kernel<CC, P, 1><<<blocks, threads, 0, stream>>>(xp, gp, hp, yp, T, H, W, pt, ph);
  };
  const bool fast = (C == 1024 || C == 512 || C == 256);
  TORCH_CHECK(!gp || fast, "Unsupported RMS width ", C);
  if (fast) {
    if (C == 1024) go(std::integral_constant<int, 1024>{});
    else if (C == 512) go(std::integral_constant<int, 512>{});
    else go(std::integral_constant<int, 256>{});
  } else {
    TORCH_CHECK(C % 8 == 0, "norm_pad copy needs C divisible by 8, got ", C);
    copy_row_kernel<<<blocks, threads, 0, stream>>>((const uint4*)xp, (const uint4*)hp, (uint4*)yp, T, H, W,
                                                    pt, ph, C / 8);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
