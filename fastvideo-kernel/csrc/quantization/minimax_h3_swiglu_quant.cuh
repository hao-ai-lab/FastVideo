// SPDX-License-Identifier: Apache-2.0
// Private building block for a SwiGLU/NVFP4 GEMM epilogue. No runtime dispatch.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ float rcp_approx(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float swiglu_bf16(float value, float gate) {
  float e;
  const float log2e = -1.4426950408889634f;
  const float exponent = __fmul_rn(gate, log2e);
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(exponent));
  float sigmoid;
  const float denominator = __fadd_rn(1.0f, e);
  // Match Triton sigmoid, including subnormal reciprocals for large negative gates.
  asm("div.full.f32 %0, %1, %2;" : "=f"(sigmoid) : "f"(1.0f), "f"(denominator));
  return __bfloat162float(__float2bfloat16_rn(__fmul_rn(value, __fmul_rn(gate, sigmoid))));
}

__device__ __forceinline__ uint8_t scale_fp8(float x) {
  uint32_t packed;
  asm("{ .reg .b16 s; cvt.rn.satfinite.e4m3x2.f32 s, %1, %1; mov.b32 %0, {s,s}; }"
      : "=r"(packed) : "f"(x));
  return static_cast<uint8_t>(packed);
}

__device__ __forceinline__ float scale_float(uint8_t x) {
  // E4M3 positive finite values; use an exact bit construction including subnormals.
  const int exponent = (x >> 3) & 15;
  const int mantissa = x & 7;
  return exponent == 0 ? __fmul_rn(float(mantissa), 0.001953125f)
                       : __uint_as_float(((exponent + 120) << 23) | (mantissa << 20));
}

__device__ __forceinline__ uint8_t fp4_pair(float lo, float hi) {
  uint32_t packed;
  asm("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, %1, %2; mov.b32 %0, {q,q,q,q}; }"
      : "=r"(packed) : "f"(hi), "f"(lo));
  return static_cast<uint8_t>(packed);
}

__global__ void quant_probe(const __nv_bfloat16* x, const float* sf, uint8_t* q,
                            uint8_t* scales, float* amax_out, __nv_bfloat16* y_out,
                            int rows, int width, bool paired) {
  const int row = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  float value = 0, gate = 0;
  if (row < rows && col < width) {
    const int64_t base = int64_t(row) * width * 2;
    value = __bfloat162float(x[base + (paired ? 2 * col : col)]);
    gate = __bfloat162float(x[base + (paired ? 2 * col + 1 : width + col)]);
  }
  const float y = swiglu_bf16(value, gate);
  float amax = fabsf(y);
  for (int delta = 1; delta < 16; delta *= 2)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta, 16));
  const float global_sf = *sf;
  const uint8_t scale = scale_fp8(__fmul_rn(global_sf, __fmul_rn(amax, rcp_approx(6.0f))));
  const float inv = amax != 0 ? rcp_approx(__fmul_rn(scale_float(scale), rcp_approx(global_sf))) : 0;
  const float code = __fmul_rn(y, inv);
  const float hi = __shfl_xor_sync(0xffffffff, code, 1);
  if (row < rows && col < width) {
    if ((col & 1) == 0) q[int64_t(row) * (width / 2) + col / 2] = fp4_pair(code, hi);
    if (y_out) y_out[int64_t(row) * width + col] = __float2bfloat16_rn(y);
  }
  if (col < width && (col & 15) == 0) {
    const int group = col / 16;
    const int64_t index = (((((int64_t(row) / 128) * (width / 64) + group / 4) * 32
                           + row % 32) * 4 + (row / 32) % 4) * 4 + group % 4);
    scales[index] = scale;
    if (amax_out) amax_out[int64_t(row) * (width / 16) + group] = amax;
  }
}

extern "C" int launch_quant_probe(const void* x, const float* sf, void* q, void* scales,
                                  float* amax_out, void* y_out, int rows, int width,
                                  int paired, void* stream) {
  quant_probe<<<dim3((rows + 127) / 128 * 128, (width + 255) / 256), 256, 0,
                static_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16*>(x), sf,
      static_cast<uint8_t*>(q), static_cast<uint8_t*>(scales), amax_out,
      static_cast<__nv_bfloat16*>(y_out), rows, width, paired);
  return static_cast<int>(cudaGetLastError());
}
