/*
 * Copyright (c) 2025 by SageAttention team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/all.h>
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp8.h>

#include "cuda_utils.h"
#include "../blackwell/block_config.h"

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(pytorch_dtype, c_type, ...)                \
  if (pytorch_dtype == at::ScalarType::Half) {                                          \
    using c_type = half;                                                                \
    __VA_ARGS__                                                                         \
  } else if (pytorch_dtype == at::ScalarType::BFloat16) {                               \
    using c_type = nv_bfloat16;                                                         \
    __VA_ARGS__                                                                         \
  } else {                                                                              \
    std::ostringstream oss;                                                             \
    oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << pytorch_dtype;    \
    TORCH_CHECK(false, oss.str());                                                      \
  }

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)              \
  if (head_dim == 64) {                                         \
    constexpr int HEAD_DIM = 64;                                \
    __VA_ARGS__                                                 \
  } else if (head_dim == 128) {                                 \
    constexpr int HEAD_DIM = 128;                               \
    __VA_ARGS__                                                 \
  } else {                                                      \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported head dim: " << int(head_dim);       \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.is_cuda(), "Tensor " #x " must be on CUDA")
#define CHECK_DTYPE(x, true_dtype)     \
  TORCH_CHECK(x.dtype() == true_dtype, \
              "Tensor " #x " must have dtype (" #true_dtype ")")
#define CHECK_DIMS(x, true_dim)    \
  TORCH_CHECK(x.dim() == true_dim, \
              "Tensor " #x " must have dimension number (" #true_dim ")")
#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              "Tensor " #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous")
#define CHECK_LASTDIM_CONTIGUOUS(x) \
  TORCH_CHECK(x.stride(-1) == 1,    \
              "Tensor " #x " must be contiguous at the last dimension")

constexpr int CVT_FP4_ELTS_PER_THREAD = 16;

// Constants for two-level quantization (same as in softmax_fused.h for consistency)
constexpr float FP8_SCALE_X_FP4_SCALE = 1.f / (448.f * 6.f);

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 *array) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  return 0;
#endif
}

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T>
struct TypeConverter {
  using Type = half2;
};  // keep for generality

template <>
struct TypeConverter<half2> {
  using Type = half;
};

template <>
struct TypeConverter<half> {
  using Type = half2;
};

template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

// Define a 32 bytes packed data type.
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[8];
};

template <uint32_t head_dim, uint32_t BLOCK_SIZE, bool permute, typename T>
__global__ void scaled_fp4_quant_kernel(
    const T* input, uint8_t* output, uint8_t* output_sf,
    int batch_size, int num_heads, int num_tokens,
    int stride_bz_input, int stride_h_input, int stride_seq_input,
    int stride_bz_output, int stride_h_output, int stride_seq_output,
    int stride_bz_output_sf, int stride_h_output_sf, int stride_seq_output_sf) {
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 input are supported");
  using PackedVec = PackedVec<T>;

  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  const int token_block_id = blockIdx.x;

  static_assert(CVT_FP4_ELTS_PER_THREAD == 8 || CVT_FP4_ELTS_PER_THREAD == 16,
                "CVT_FP4_ELTS_PER_THREAD must be 8 or 16");
  static_assert(sizeof(PackedVec) == sizeof(T) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  constexpr uint32_t NUM_THREADS_PER_TOKEN = head_dim / CVT_FP4_ELTS_PER_THREAD;

  // load input
  const int token_id = token_block_id * BLOCK_SIZE + threadIdx.x / NUM_THREADS_PER_TOKEN;
  
  int load_token_id;
  if constexpr (!permute) {
    load_token_id = token_id;
  } else {
    int local_token_id = threadIdx.x / NUM_THREADS_PER_TOKEN;
    int local_token_id_residue = local_token_id % 32;
    // [0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31]
    load_token_id = token_block_id * BLOCK_SIZE + (local_token_id / 32) * 32 +
                    (local_token_id_residue / 8) * 2 + 
                    ((local_token_id_residue % 8) / 2) * 8 +
                    (local_token_id_residue % 8) % 2;
  }

  PackedVec in_vec;
  
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    reinterpret_cast<uint32_t&>(in_vec.elts[i]) = 0;
  }
  
  if (load_token_id < num_tokens) {
    in_vec = reinterpret_cast<PackedVec const*>(input + 
                                          batch_id * stride_bz_input + // batch dim
                                          head_id * stride_h_input +   // head dim
                                          load_token_id * stride_seq_input + // seq dim
                                          (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD)[0]; // feature dim
  }

  // calculate max of every consecutive 16 elements
  auto localMax = __habs2(in_vec.elts[0]);
  #pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) { // local max
    localMax = __hmax2(localMax, __habs2(in_vec.elts[i]));
  }

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) { // shuffle across two threads
    localMax = __hmax2(__shfl_xor_sync(0xffffffff, localMax, 1, 32), localMax);
  }

  float vecMax = float(__hmax(localMax.x, localMax.y));

  // scaling factor
  float SFValue = vecMax / 6.0f;
  uint8_t SFValueFP8;
  reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
  SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));

  float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;

  // convert input to float2 and apply scale
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same<T, half>::value) {
      fp2Vals[i] = __half22float2(in_vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(in_vec.elts[i]);
    }
    fp2Vals[i].x = fp2Vals[i].x * SFValueInv;
    fp2Vals[i].y = fp2Vals[i].y * SFValueInv;
  }

  // convert to e2m1
  uint32_t e2m1Vals[CVT_FP4_ELTS_PER_THREAD / 8];
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 8; i++) {
    e2m1Vals[i] = fp32_vec_to_e2m1(fp2Vals + i * 4);
  }

  // save, do not check range
  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    reinterpret_cast<uint32_t*>(output + 
                                batch_id * stride_bz_output +
                                head_id * stride_h_output +
                                token_id * stride_seq_output +
                                (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD / 2)[0] = e2m1Vals[0];
  } else {
    reinterpret_cast<uint64_t*>(output + 
                                batch_id * stride_bz_output +
                                head_id * stride_h_output +
                                token_id * stride_seq_output +
                                (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD / 2)[0] = reinterpret_cast<uint64_t*>(e2m1Vals)[0];
  }
  
  uint8_t* output_sf_save_base = output_sf + batch_id * stride_bz_output_sf + head_id * stride_h_output_sf + (token_id / 64) * 64 * stride_seq_output_sf;
  uint32_t token_id_local = token_id % 64;

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 16) {
    uint32_t col_id_local = threadIdx.x % NUM_THREADS_PER_TOKEN;
    uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                            (token_id_local / 16) * 4 + (token_id_local % 16) * 16;
    reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
  } else {
    if (threadIdx.x % 2 == 0) {
      uint32_t col_id_local = (threadIdx.x % NUM_THREADS_PER_TOKEN) / 2;
      uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                            (token_id_local / 16) * 4 + (token_id_local % 16) * 16;
      reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
    }
  }
}

template <uint32_t head_dim, uint32_t BLOCK_SIZE, typename T>
__global__ void scaled_fp4_quant_trans_kernel(
    const T* input, uint8_t* output, uint8_t* output_sf,
    int batch_size, int num_heads, int num_tokens,
    int stride_bz_input, int stride_h_input, int stride_seq_input,
    int stride_bz_output, int stride_h_output, int stride_d_output,
    int stride_bz_output_sf, int stride_h_output_sf, int stride_d_output_sf) {
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 input are supported");
  using PackedVec = PackedVec<T>;

  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  const int token_block_id = blockIdx.x;

  static_assert(CVT_FP4_ELTS_PER_THREAD == 8 || CVT_FP4_ELTS_PER_THREAD == 16,
                "CVT_FP4_ELTS_PER_THREAD must be 8 or 16");
  static_assert(sizeof(PackedVec) == sizeof(T) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  constexpr uint32_t NUM_THREADS_PER_TOKEN = head_dim / CVT_FP4_ELTS_PER_THREAD;
  constexpr uint32_t NUM_THREADS_PER_SEQ = BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD;

  // load input
  const int token_id = token_block_id * BLOCK_SIZE + threadIdx.x / NUM_THREADS_PER_TOKEN;

  PackedVec in_vec;
  
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    reinterpret_cast<uint32_t&>(in_vec.elts[i]) = 0;
  }
  
  if (token_id < num_tokens) {
    in_vec = reinterpret_cast<PackedVec const*>(input + 
                                          batch_id * stride_bz_input + // batch dim
                                          head_id * stride_h_input +   // head dim
                                          token_id * stride_seq_input + // seq dim
                                          (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD)[0]; // feature dim
  }

  // transpose
  __shared__ T shared_input[BLOCK_SIZE * head_dim];
  reinterpret_cast<PackedVec*>(shared_input)[threadIdx.x] = in_vec;
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    in_vec.elts[i].x = shared_input[(threadIdx.x / NUM_THREADS_PER_SEQ) + ((threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD + 2 * i) * head_dim];
    in_vec.elts[i].y = shared_input[(threadIdx.x / NUM_THREADS_PER_SEQ) + ((threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD + 2 * i + 1) * head_dim];
  }

  // calculate max of every consecutive 16 elements
  auto localMax = __habs2(in_vec.elts[0]);
  #pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) { // local max
    localMax = __hmax2(localMax, __habs2(in_vec.elts[i]));
  }

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) { // shuffle across two threads
    localMax = __hmax2(__shfl_xor_sync(0xffffffff, localMax, 1, 32), localMax);
  }

  float vecMax = float(__hmax(localMax.x, localMax.y));

  // scaling factor
  float SFValue = vecMax / 6.0f;
  uint8_t SFValueFP8;
  reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
  SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));

  float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;

  // convert input to float2 and apply scale
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same<T, half>::value) {
      fp2Vals[i] = __half22float2(in_vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(in_vec.elts[i]);
    }
    fp2Vals[i].x = fp2Vals[i].x * SFValueInv;
    fp2Vals[i].y = fp2Vals[i].y * SFValueInv;
  }

  // convert to e2m1
  uint32_t e2m1Vals[CVT_FP4_ELTS_PER_THREAD / 8];
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 8; i++) {
    e2m1Vals[i] = fp32_vec_to_e2m1(fp2Vals + i * 4);
  }

  // save
  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    reinterpret_cast<uint32_t*>(output + 
                                batch_id * stride_bz_output +
                                head_id * stride_h_output +
                                (threadIdx.x / NUM_THREADS_PER_SEQ) * stride_d_output +
                                (token_block_id * BLOCK_SIZE + (threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD) / 2)[0] = e2m1Vals[0];
  } else {
    reinterpret_cast<uint64_t*>(output + 
                                batch_id * stride_bz_output +
                                head_id * stride_h_output +
                                (threadIdx.x / NUM_THREADS_PER_SEQ) * stride_d_output +
                                (token_block_id * BLOCK_SIZE + (threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD) / 2)[0] = reinterpret_cast<uint64_t*>(e2m1Vals)[0];
  }

  uint8_t *output_sf_save_base = output_sf + 
                                batch_id * stride_bz_output_sf +
                                head_id * stride_h_output_sf +
                                (threadIdx.x / NUM_THREADS_PER_SEQ / 64) * 64 * stride_d_output_sf;
  uint32_t row_id_local = (threadIdx.x / NUM_THREADS_PER_SEQ) % 64;

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 16) {
    uint32_t col_id_local = token_block_id * BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD + threadIdx.x % NUM_THREADS_PER_SEQ;
    uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                            (row_id_local / 16) * 4 + (row_id_local % 16) * 16;
    reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
  } else {
    if (threadIdx.x % 2 == 0) {
      uint32_t col_id_local = token_block_id * BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD + (threadIdx.x % NUM_THREADS_PER_SEQ) / 2;
      uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                              (row_id_local / 16) * 4 + (row_id_local % 16) * 16;
      reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
    }
  }
}

// ============================================================================
// Two-Level Quantization Kernels
// Level 1: Per-row scale s_row = absmax_row(x) / (448 * 6)
// Level 2: Per-16-element scale s_16 = absmax_block(x/s_row) / 6
// This provides better dynamic range similar to P quantization in softmax_fused.h
// Note: absmax is used (not max) because Q, K, V can have negative values
// ============================================================================

template <uint32_t head_dim, uint32_t BLOCK_SIZE, bool permute, typename T>
__global__ void scaled_fp4_quant_two_level_kernel(
    const T* input, uint8_t* output, uint8_t* output_sf, uint8_t* output_sf_row,
    int batch_size, int num_heads, int num_tokens,
    int stride_bz_input, int stride_h_input, int stride_seq_input,
    int stride_bz_output, int stride_h_output, int stride_seq_output,
    int stride_bz_output_sf, int stride_h_output_sf, int stride_seq_output_sf,
    int stride_bz_output_sf_row, int stride_h_output_sf_row, int stride_seq_output_sf_row) {
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 input are supported");
  using PackedVec = PackedVec<T>;

  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  const int token_block_id = blockIdx.x;

  static_assert(CVT_FP4_ELTS_PER_THREAD == 8 || CVT_FP4_ELTS_PER_THREAD == 16,
                "CVT_FP4_ELTS_PER_THREAD must be 8 or 16");
  static_assert(sizeof(PackedVec) == sizeof(T) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  constexpr uint32_t NUM_THREADS_PER_TOKEN = head_dim / CVT_FP4_ELTS_PER_THREAD;

  // load input
  const int token_id = token_block_id * BLOCK_SIZE + threadIdx.x / NUM_THREADS_PER_TOKEN;
  
  int load_token_id;
  if constexpr (!permute) {
    load_token_id = token_id;
  } else {
    int local_token_id = threadIdx.x / NUM_THREADS_PER_TOKEN;
    int local_token_id_residue = local_token_id % 32;
    load_token_id = token_block_id * BLOCK_SIZE + (local_token_id / 32) * 32 +
                    (local_token_id_residue / 8) * 2 + 
                    ((local_token_id_residue % 8) / 2) * 8 +
                    (local_token_id_residue % 8) % 2;
  }

  PackedVec in_vec;
  
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    reinterpret_cast<uint32_t&>(in_vec.elts[i]) = 0;
  }
  
  if (load_token_id < num_tokens) {
    in_vec = reinterpret_cast<PackedVec const*>(input + 
                                          batch_id * stride_bz_input +
                                          head_id * stride_h_input +
                                          load_token_id * stride_seq_input +
                                          (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD)[0];
  }

  // Step 1: Calculate local max of this thread's 16 elements
  auto localMax = __habs2(in_vec.elts[0]);
  #pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(in_vec.elts[i]));
  }

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    localMax = __hmax2(__shfl_xor_sync(0xffffffff, localMax, 1, 32), localMax);
  }

  float blockMax = float(__hmax(localMax.x, localMax.y));

  // Step 2: Compute row absmax across all threads handling this token (for first-level scale)
  // Each token is handled by NUM_THREADS_PER_TOKEN threads
  // Note: blockMax is already absmax since we used __habs2 above
  float rowAbsMax = blockMax;
  // Reduction across threads within the same token
  int thread_in_token = threadIdx.x % NUM_THREADS_PER_TOKEN;
  for (int offset = NUM_THREADS_PER_TOKEN / 2; offset > 0; offset /= 2) {
    float other = __shfl_xor_sync(0xffffffff, rowAbsMax, offset, 32);
    rowAbsMax = fmaxf(rowAbsMax, other);
  }

  // Step 3: Compute first-level scale (per-row): s_row = absmax_row / (448 * 6)
  float SFRowValue = rowAbsMax * FP8_SCALE_X_FP4_SCALE;
  uint8_t SFRowValueFP8;
  reinterpret_cast<__nv_fp8_e4m3&>(SFRowValueFP8) = __nv_fp8_e4m3(SFRowValue);
  SFRowValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFRowValueFP8));
  float SFRowValueInv = (SFRowValue == 0.0f) ? 0.0f : 1.0f / SFRowValue;

  // Step 4: Pre-scale values by first-level scale and compute second-level max
  float2 fp2Vals_prescaled[CVT_FP4_ELTS_PER_THREAD / 2];
  float prescaledMax = 0.0f;
  
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same<T, half>::value) {
      fp2Vals_prescaled[i] = __half22float2(in_vec.elts[i]);
    } else {
      fp2Vals_prescaled[i] = __bfloat1622float2(in_vec.elts[i]);
    }
    fp2Vals_prescaled[i].x *= SFRowValueInv;
    fp2Vals_prescaled[i].y *= SFRowValueInv;
    prescaledMax = fmaxf(prescaledMax, fabsf(fp2Vals_prescaled[i].x));
    prescaledMax = fmaxf(prescaledMax, fabsf(fp2Vals_prescaled[i].y));
  }

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    prescaledMax = fmaxf(__shfl_xor_sync(0xffffffff, prescaledMax, 1, 32), prescaledMax);
  }

  // Step 5: Compute second-level scale (per-16-element block)
  float SFValue = prescaledMax / 6.0f;
  uint8_t SFValueFP8;
  reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
  SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));
  float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;

  // Step 6: Apply second-level scale to pre-scaled values
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    fp2Vals_prescaled[i].x *= SFValueInv;
    fp2Vals_prescaled[i].y *= SFValueInv;
  }

  // Step 7: Convert to e2m1
  uint32_t e2m1Vals[CVT_FP4_ELTS_PER_THREAD / 8];
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 8; i++) {
    e2m1Vals[i] = fp32_vec_to_e2m1(fp2Vals_prescaled + i * 4);
  }

  // Save quantized output
  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    reinterpret_cast<uint32_t*>(output + 
                                batch_id * stride_bz_output +
                                head_id * stride_h_output +
                                token_id * stride_seq_output +
                                (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD / 2)[0] = e2m1Vals[0];
  } else {
    reinterpret_cast<uint64_t*>(output + 
                                batch_id * stride_bz_output +
                                head_id * stride_h_output +
                                token_id * stride_seq_output +
                                (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD / 2)[0] = reinterpret_cast<uint64_t*>(e2m1Vals)[0];
  }
  
  // Save second-level scale (per-16-element)
  uint8_t* output_sf_save_base = output_sf + batch_id * stride_bz_output_sf + head_id * stride_h_output_sf + (token_id / 64) * 64 * stride_seq_output_sf;
  uint32_t token_id_local = token_id % 64;

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 16) {
    uint32_t col_id_local = threadIdx.x % NUM_THREADS_PER_TOKEN;
    uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                            (token_id_local / 16) * 4 + (token_id_local % 16) * 16;
    reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
  } else {
    if (threadIdx.x % 2 == 0) {
      uint32_t col_id_local = (threadIdx.x % NUM_THREADS_PER_TOKEN) / 2;
      uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                            (token_id_local / 16) * 4 + (token_id_local % 16) * 16;
      reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
    }
  }

  // Save first-level scale (per-row absmax) - only one thread per token writes this
  if (thread_in_token == 0 && token_id < num_tokens) {
    uint8_t* output_sf_row_ptr = output_sf_row + 
                                  batch_id * stride_bz_output_sf_row + 
                                  head_id * stride_h_output_sf_row + 
                                  token_id * stride_seq_output_sf_row;
    reinterpret_cast<uint8_t*>(output_sf_row_ptr)[0] = SFRowValueFP8;
  }
}

template <uint32_t head_dim, uint32_t BLOCK_SIZE, typename T>
__global__ void scaled_fp4_quant_trans_two_level_kernel(
    const T* input, uint8_t* output, uint8_t* output_sf, uint8_t* output_sf_row,
    int batch_size, int num_heads, int num_tokens,
    int stride_bz_input, int stride_h_input, int stride_seq_input,
    int stride_bz_output, int stride_h_output, int stride_d_output,
    int stride_bz_output_sf, int stride_h_output_sf, int stride_d_output_sf,
    int stride_bz_output_sf_row, int stride_h_output_sf_row, int stride_d_output_sf_row) {
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 input are supported");
  using PackedVec = PackedVec<T>;

  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  const int token_block_id = blockIdx.x;

  static_assert(CVT_FP4_ELTS_PER_THREAD == 8 || CVT_FP4_ELTS_PER_THREAD == 16,
                "CVT_FP4_ELTS_PER_THREAD must be 8 or 16");
  static_assert(sizeof(PackedVec) == sizeof(T) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  constexpr uint32_t NUM_THREADS_PER_TOKEN = head_dim / CVT_FP4_ELTS_PER_THREAD;
  constexpr uint32_t NUM_THREADS_PER_SEQ = BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD;

  const int token_id = token_block_id * BLOCK_SIZE + threadIdx.x / NUM_THREADS_PER_TOKEN;

  PackedVec in_vec;
  
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    reinterpret_cast<uint32_t&>(in_vec.elts[i]) = 0;
  }
  
  if (token_id < num_tokens) {
    in_vec = reinterpret_cast<PackedVec const*>(input + 
                                          batch_id * stride_bz_input +
                                          head_id * stride_h_input +
                                          token_id * stride_seq_input +
                                          (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD)[0];
  }

  // Transpose via shared memory
  __shared__ T shared_input[BLOCK_SIZE * head_dim];
  reinterpret_cast<PackedVec*>(shared_input)[threadIdx.x] = in_vec;
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    in_vec.elts[i].x = shared_input[(threadIdx.x / NUM_THREADS_PER_SEQ) + ((threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD + 2 * i) * head_dim];
    in_vec.elts[i].y = shared_input[(threadIdx.x / NUM_THREADS_PER_SEQ) + ((threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD + 2 * i + 1) * head_dim];
  }

  // After transpose, we have data along the sequence dimension
  // For V transpose, rows are now head_dim elements, cols are sequence
  
  // Step 1: Calculate local max
  auto localMax = __habs2(in_vec.elts[0]);
  #pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(in_vec.elts[i]));
  }

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    localMax = __hmax2(__shfl_xor_sync(0xffffffff, localMax, 1, 32), localMax);
  }

  float blockMax = float(__hmax(localMax.x, localMax.y));

  // Step 2: Compute row absmax (after transpose, row = head_dim element)
  // Each "row" after transpose is processed by threads with same (threadIdx.x / NUM_THREADS_PER_SEQ)
  // Note: blockMax is already absmax since we used __habs2 above
  float rowAbsMax = blockMax;
  int row_id = threadIdx.x / NUM_THREADS_PER_SEQ;
  // Reduction across threads handling same row
  for (int offset = NUM_THREADS_PER_SEQ / 2; offset > 0; offset /= 2) {
    int src_lane = (threadIdx.x ^ offset);
    if ((src_lane / NUM_THREADS_PER_SEQ) == row_id) {
      float other = __shfl_xor_sync(0xffffffff, rowAbsMax, offset, 32);
      rowAbsMax = fmaxf(rowAbsMax, other);
    }
  }

  // Step 3: First-level scale: s_row = absmax_row / (448 * 6)
  float SFRowValue = rowAbsMax * FP8_SCALE_X_FP4_SCALE;
  uint8_t SFRowValueFP8;
  reinterpret_cast<__nv_fp8_e4m3&>(SFRowValueFP8) = __nv_fp8_e4m3(SFRowValue);
  SFRowValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFRowValueFP8));
  float SFRowValueInv = (SFRowValue == 0.0f) ? 0.0f : 1.0f / SFRowValue;

  // Step 4: Pre-scale and compute second-level max
  float2 fp2Vals_prescaled[CVT_FP4_ELTS_PER_THREAD / 2];
  float prescaledMax = 0.0f;
  
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same<T, half>::value) {
      fp2Vals_prescaled[i] = __half22float2(in_vec.elts[i]);
    } else {
      fp2Vals_prescaled[i] = __bfloat1622float2(in_vec.elts[i]);
    }
    fp2Vals_prescaled[i].x *= SFRowValueInv;
    fp2Vals_prescaled[i].y *= SFRowValueInv;
    prescaledMax = fmaxf(prescaledMax, fabsf(fp2Vals_prescaled[i].x));
    prescaledMax = fmaxf(prescaledMax, fabsf(fp2Vals_prescaled[i].y));
  }

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    prescaledMax = fmaxf(__shfl_xor_sync(0xffffffff, prescaledMax, 1, 32), prescaledMax);
  }

  // Step 5: Second-level scale
  float SFValue = prescaledMax / 6.0f;
  uint8_t SFValueFP8;
  reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
  SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));
  float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;

  // Step 6: Apply second-level scale
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    fp2Vals_prescaled[i].x *= SFValueInv;
    fp2Vals_prescaled[i].y *= SFValueInv;
  }

  // Step 7: Convert to e2m1
  uint32_t e2m1Vals[CVT_FP4_ELTS_PER_THREAD / 8];
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 8; i++) {
    e2m1Vals[i] = fp32_vec_to_e2m1(fp2Vals_prescaled + i * 4);
  }

  // Save quantized output
  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    reinterpret_cast<uint32_t*>(output + 
                                batch_id * stride_bz_output +
                                head_id * stride_h_output +
                                (threadIdx.x / NUM_THREADS_PER_SEQ) * stride_d_output +
                                (token_block_id * BLOCK_SIZE + (threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD) / 2)[0] = e2m1Vals[0];
  } else {
    reinterpret_cast<uint64_t*>(output + 
                                batch_id * stride_bz_output +
                                head_id * stride_h_output +
                                (threadIdx.x / NUM_THREADS_PER_SEQ) * stride_d_output +
                                (token_block_id * BLOCK_SIZE + (threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD) / 2)[0] = reinterpret_cast<uint64_t*>(e2m1Vals)[0];
  }

  // Save second-level scale
  uint8_t *output_sf_save_base = output_sf + 
                                batch_id * stride_bz_output_sf +
                                head_id * stride_h_output_sf +
                                (threadIdx.x / NUM_THREADS_PER_SEQ / 64) * 64 * stride_d_output_sf;
  uint32_t row_id_local = (threadIdx.x / NUM_THREADS_PER_SEQ) % 64;

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 16) {
    uint32_t col_id_local = token_block_id * BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD + threadIdx.x % NUM_THREADS_PER_SEQ;
    uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                            (row_id_local / 16) * 4 + (row_id_local % 16) * 16;
    reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
  } else {
    if (threadIdx.x % 2 == 0) {
      uint32_t col_id_local = token_block_id * BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD + (threadIdx.x % NUM_THREADS_PER_SEQ) / 2;
      uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                              (row_id_local / 16) * 4 + (row_id_local % 16) * 16;
      reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
    }
  }

  // Save first-level scale (per head_dim element, one per row after transpose)
  // Only the first thread handling each row writes
  if ((threadIdx.x % NUM_THREADS_PER_SEQ) == 0) {
    uint8_t* output_sf_row_ptr = output_sf_row + 
                                  batch_id * stride_bz_output_sf_row + 
                                  head_id * stride_h_output_sf_row + 
                                  row_id * stride_d_output_sf_row;
    reinterpret_cast<uint8_t*>(output_sf_row_ptr)[0] = SFRowValueFP8;
  }
}

void scaled_fp4_quant(torch::Tensor const& input,
                            torch::Tensor const& output,
                            torch::Tensor const& output_sf,
                            int tensor_layout) {
  constexpr int BLOCK_SIZE = flash::BLOCK_M;
  
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(output_sf);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_LASTDIM_CONTIGUOUS(output_sf);

  CHECK_DTYPE(output, at::ScalarType::Byte);
  CHECK_DTYPE(output_sf, at::ScalarType::Float8_e4m3fn);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(output_sf, 4);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);
  const int stride_bz_output_sf = output_sf.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_seq_output, stride_seq_output_sf;
  int stride_h_input, stride_h_output, stride_h_output_sf;
  if (tensor_layout == 0) {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_seq_output = output.stride(1);
    stride_seq_output_sf = output_sf.stride(1);
    stride_h_input = input.stride(2);
    stride_h_output = output.stride(2);
    stride_h_output_sf = output_sf.stride(2);

    CHECK_SHAPE(output, batch_size, num_tokens, num_heads, head_dim / 2);
    CHECK_SHAPE(output_sf, batch_size, num_tokens, num_heads, head_dim / 16);
  } else {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_seq_output = output.stride(2);
    stride_seq_output_sf = output_sf.stride(2);
    stride_h_input = input.stride(1);
    stride_h_output = output.stride(1);
    stride_h_output_sf = output_sf.stride(1);

    CHECK_SHAPE(output, batch_size, num_heads, num_tokens, head_dim / 2);
    CHECK_SHAPE(output_sf, batch_size, num_heads, num_tokens, head_dim / 16);
  }

  auto input_dtype = input.scalar_type();
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1);
      dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);

      scaled_fp4_quant_kernel<HEAD_DIM, BLOCK_SIZE, false, c_type>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<c_type*>(input.data_ptr()),
              reinterpret_cast<uint8_t*>(output.data_ptr()),
              reinterpret_cast<uint8_t*>(output_sf.data_ptr()),
              batch_size, num_heads, num_tokens,
              stride_bz_input, stride_h_input, stride_seq_input,
              stride_bz_output, stride_h_output, stride_seq_output,
              stride_bz_output_sf, stride_h_output_sf, stride_seq_output_sf);
    });
  });
}

void scaled_fp4_quant_permute(torch::Tensor const& input,
                            torch::Tensor const& output,
                            torch::Tensor const& output_sf,
                            int tensor_layout) {
  constexpr int BLOCK_SIZE = flash::BLOCK_M;

  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(output_sf);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_LASTDIM_CONTIGUOUS(output_sf);

  CHECK_DTYPE(output, at::ScalarType::Byte);
  CHECK_DTYPE(output_sf, at::ScalarType::Float8_e4m3fn);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(output_sf, 4);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);
  const int stride_bz_output_sf = output_sf.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_seq_output, stride_seq_output_sf;
  int stride_h_input, stride_h_output, stride_h_output_sf;
  if (tensor_layout == 0) {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_seq_output = output.stride(1);
    stride_seq_output_sf = output_sf.stride(1);
    stride_h_input = input.stride(2);
    stride_h_output = output.stride(2);
    stride_h_output_sf = output_sf.stride(2);

    CHECK_SHAPE(output, batch_size, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, num_heads, head_dim / 2);
    CHECK_SHAPE(output_sf, batch_size, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, num_heads, head_dim / 16);
  } else {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_seq_output = output.stride(2);
    stride_seq_output_sf = output_sf.stride(2);
    stride_h_input = input.stride(1);
    stride_h_output = output.stride(1);
    stride_h_output_sf = output_sf.stride(1);

    CHECK_SHAPE(output, batch_size, num_heads, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, head_dim / 2);
    CHECK_SHAPE(output_sf, batch_size, num_heads, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, head_dim / 16);
  }

  auto input_dtype = input.scalar_type();
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr int BLOCK_SIZE = flash::BLOCK_M;
      dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1);
      dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);

      scaled_fp4_quant_kernel<HEAD_DIM, BLOCK_SIZE, true, c_type>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<c_type*>(input.data_ptr()),
              reinterpret_cast<uint8_t*>(output.data_ptr()),
              reinterpret_cast<uint8_t*>(output_sf.data_ptr()),
              batch_size, num_heads, num_tokens,
              stride_bz_input, stride_h_input, stride_seq_input,
              stride_bz_output, stride_h_output, stride_seq_output,
              stride_bz_output_sf, stride_h_output_sf, stride_seq_output_sf);
    });
  });
}

void scaled_fp4_quant_trans(torch::Tensor const& input,
                            torch::Tensor const& output,
                            torch::Tensor const& output_sf,
                            int tensor_layout) {
  constexpr int BLOCK_SIZE = flash::BLOCK_M;
  
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(output_sf);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_LASTDIM_CONTIGUOUS(output_sf);

  CHECK_DTYPE(output, at::ScalarType::Byte);
  CHECK_DTYPE(output_sf, at::ScalarType::Float8_e4m3fn);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(output_sf, 4);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);
  const int stride_bz_output_sf = output_sf.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input; 
  int stride_d_output, stride_d_output_sf;
  int stride_h_input, stride_h_output, stride_h_output_sf;
  if (tensor_layout == 0) {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_d_output = output.stride(1);
    stride_d_output_sf = output_sf.stride(1);
    stride_h_input = input.stride(2);
    stride_h_output = output.stride(2);
    stride_h_output_sf = output_sf.stride(2);

    CHECK_SHAPE(output, batch_size, head_dim, num_heads, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 2);
    CHECK_SHAPE(output_sf, batch_size, head_dim, num_heads, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 16);
  } else {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_d_output = output.stride(2);
    stride_d_output_sf = output_sf.stride(2);
    stride_h_input = input.stride(1);
    stride_h_output = output.stride(1);
    stride_h_output_sf = output_sf.stride(1);

    CHECK_SHAPE(output, batch_size, num_heads, head_dim, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 2);
    CHECK_SHAPE(output_sf, batch_size, num_heads, head_dim, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 16);
  }

  auto input_dtype = input.scalar_type();
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1);
      dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);

      scaled_fp4_quant_trans_kernel<HEAD_DIM, BLOCK_SIZE, c_type>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<c_type*>(input.data_ptr()),
              reinterpret_cast<uint8_t*>(output.data_ptr()),
              reinterpret_cast<uint8_t*>(output_sf.data_ptr()),
              batch_size, num_heads, num_tokens,
              stride_bz_input, stride_h_input, stride_seq_input,
              stride_bz_output, stride_h_output, stride_d_output,
              stride_bz_output_sf, stride_h_output_sf, stride_d_output_sf);
    });
  });
}

// ============================================================================
// Two-Level Quantization Host Functions
// ============================================================================

void scaled_fp4_quant_two_level(torch::Tensor const& input,
                                torch::Tensor const& output,
                                torch::Tensor const& output_sf,
                                torch::Tensor const& output_sf_row,
                                int tensor_layout) {
  constexpr int BLOCK_SIZE = flash::BLOCK_M;
  
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(output_sf);
  CHECK_CUDA(output_sf_row);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_LASTDIM_CONTIGUOUS(output_sf);
  CHECK_LASTDIM_CONTIGUOUS(output_sf_row);

  CHECK_DTYPE(output, at::ScalarType::Byte);
  CHECK_DTYPE(output_sf, at::ScalarType::Float8_e4m3fn);
  CHECK_DTYPE(output_sf_row, at::ScalarType::Float8_e4m3fn);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(output_sf, 4);
  CHECK_DIMS(output_sf_row, 4);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);
  const int stride_bz_output_sf = output_sf.stride(0);
  const int stride_bz_output_sf_row = output_sf_row.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_seq_output, stride_seq_output_sf, stride_seq_output_sf_row;
  int stride_h_input, stride_h_output, stride_h_output_sf, stride_h_output_sf_row;
  if (tensor_layout == 0) {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_seq_output = output.stride(1);
    stride_seq_output_sf = output_sf.stride(1);
    stride_seq_output_sf_row = output_sf_row.stride(1);
    stride_h_input = input.stride(2);
    stride_h_output = output.stride(2);
    stride_h_output_sf = output_sf.stride(2);
    stride_h_output_sf_row = output_sf_row.stride(2);

    CHECK_SHAPE(output, batch_size, num_tokens, num_heads, head_dim / 2);
    CHECK_SHAPE(output_sf, batch_size, num_tokens, num_heads, head_dim / 16);
    CHECK_SHAPE(output_sf_row, batch_size, num_tokens, num_heads, 1);
  } else {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_seq_output = output.stride(2);
    stride_seq_output_sf = output_sf.stride(2);
    stride_seq_output_sf_row = output_sf_row.stride(2);
    stride_h_input = input.stride(1);
    stride_h_output = output.stride(1);
    stride_h_output_sf = output_sf.stride(1);
    stride_h_output_sf_row = output_sf_row.stride(1);

    CHECK_SHAPE(output, batch_size, num_heads, num_tokens, head_dim / 2);
    CHECK_SHAPE(output_sf, batch_size, num_heads, num_tokens, head_dim / 16);
    CHECK_SHAPE(output_sf_row, batch_size, num_heads, num_tokens, 1);
  }

  auto input_dtype = input.scalar_type();
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1);
      dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);

      scaled_fp4_quant_two_level_kernel<HEAD_DIM, BLOCK_SIZE, false, c_type>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<c_type*>(input.data_ptr()),
              reinterpret_cast<uint8_t*>(output.data_ptr()),
              reinterpret_cast<uint8_t*>(output_sf.data_ptr()),
              reinterpret_cast<uint8_t*>(output_sf_row.data_ptr()),
              batch_size, num_heads, num_tokens,
              stride_bz_input, stride_h_input, stride_seq_input,
              stride_bz_output, stride_h_output, stride_seq_output,
              stride_bz_output_sf, stride_h_output_sf, stride_seq_output_sf,
              stride_bz_output_sf_row, stride_h_output_sf_row, stride_seq_output_sf_row);
    });
  });
}

void scaled_fp4_quant_permute_two_level(torch::Tensor const& input,
                                        torch::Tensor const& output,
                                        torch::Tensor const& output_sf,
                                        torch::Tensor const& output_sf_row,
                                        int tensor_layout) {
  constexpr int BLOCK_SIZE = flash::BLOCK_M;

  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(output_sf);
  CHECK_CUDA(output_sf_row);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_LASTDIM_CONTIGUOUS(output_sf);
  CHECK_LASTDIM_CONTIGUOUS(output_sf_row);

  CHECK_DTYPE(output, at::ScalarType::Byte);
  CHECK_DTYPE(output_sf, at::ScalarType::Float8_e4m3fn);
  CHECK_DTYPE(output_sf_row, at::ScalarType::Float8_e4m3fn);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(output_sf, 4);
  CHECK_DIMS(output_sf_row, 4);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);
  const int stride_bz_output_sf = output_sf.stride(0);
  const int stride_bz_output_sf_row = output_sf_row.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_seq_output, stride_seq_output_sf, stride_seq_output_sf_row;
  int stride_h_input, stride_h_output, stride_h_output_sf, stride_h_output_sf_row;
  if (tensor_layout == 0) {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_seq_output = output.stride(1);
    stride_seq_output_sf = output_sf.stride(1);
    stride_seq_output_sf_row = output_sf_row.stride(1);
    stride_h_input = input.stride(2);
    stride_h_output = output.stride(2);
    stride_h_output_sf = output_sf.stride(2);
    stride_h_output_sf_row = output_sf_row.stride(2);

    CHECK_SHAPE(output, batch_size, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, num_heads, head_dim / 2);
    CHECK_SHAPE(output_sf, batch_size, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, num_heads, head_dim / 16);
    CHECK_SHAPE(output_sf_row, batch_size, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, num_heads, 1);
  } else {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_seq_output = output.stride(2);
    stride_seq_output_sf = output_sf.stride(2);
    stride_seq_output_sf_row = output_sf_row.stride(2);
    stride_h_input = input.stride(1);
    stride_h_output = output.stride(1);
    stride_h_output_sf = output_sf.stride(1);
    stride_h_output_sf_row = output_sf_row.stride(1);

    CHECK_SHAPE(output, batch_size, num_heads, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, head_dim / 2);
    CHECK_SHAPE(output_sf, batch_size, num_heads, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, head_dim / 16);
    CHECK_SHAPE(output_sf_row, batch_size, num_heads, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, 1);
  }

  auto input_dtype = input.scalar_type();
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr int BLOCK_SIZE = flash::BLOCK_M;
      dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1);
      dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);

      scaled_fp4_quant_two_level_kernel<HEAD_DIM, BLOCK_SIZE, true, c_type>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<c_type*>(input.data_ptr()),
              reinterpret_cast<uint8_t*>(output.data_ptr()),
              reinterpret_cast<uint8_t*>(output_sf.data_ptr()),
              reinterpret_cast<uint8_t*>(output_sf_row.data_ptr()),
              batch_size, num_heads, num_tokens,
              stride_bz_input, stride_h_input, stride_seq_input,
              stride_bz_output, stride_h_output, stride_seq_output,
              stride_bz_output_sf, stride_h_output_sf, stride_seq_output_sf,
              stride_bz_output_sf_row, stride_h_output_sf_row, stride_seq_output_sf_row);
    });
  });
}

void scaled_fp4_quant_trans_two_level(torch::Tensor const& input,
                                      torch::Tensor const& output,
                                      torch::Tensor const& output_sf,
                                      torch::Tensor const& output_sf_row,
                                      int tensor_layout) {
  constexpr int BLOCK_SIZE = flash::BLOCK_M;
  
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(output_sf);
  CHECK_CUDA(output_sf_row);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_LASTDIM_CONTIGUOUS(output_sf);
  CHECK_LASTDIM_CONTIGUOUS(output_sf_row);

  CHECK_DTYPE(output, at::ScalarType::Byte);
  CHECK_DTYPE(output_sf, at::ScalarType::Float8_e4m3fn);
  CHECK_DTYPE(output_sf_row, at::ScalarType::Float8_e4m3fn);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(output_sf, 4);
  CHECK_DIMS(output_sf_row, 4);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);
  const int stride_bz_output_sf = output_sf.stride(0);
  const int stride_bz_output_sf_row = output_sf_row.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input; 
  int stride_d_output, stride_d_output_sf, stride_d_output_sf_row;
  int stride_h_input, stride_h_output, stride_h_output_sf, stride_h_output_sf_row;
  if (tensor_layout == 0) {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_d_output = output.stride(1);
    stride_d_output_sf = output_sf.stride(1);
    stride_d_output_sf_row = output_sf_row.stride(1);
    stride_h_input = input.stride(2);
    stride_h_output = output.stride(2);
    stride_h_output_sf = output_sf.stride(2);
    stride_h_output_sf_row = output_sf_row.stride(2);

    CHECK_SHAPE(output, batch_size, head_dim, num_heads, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 2);
    CHECK_SHAPE(output_sf, batch_size, head_dim, num_heads, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 16);
    CHECK_SHAPE(output_sf_row, batch_size, head_dim, num_heads, 1);
  } else {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_d_output = output.stride(2);
    stride_d_output_sf = output_sf.stride(2);
    stride_d_output_sf_row = output_sf_row.stride(2);
    stride_h_input = input.stride(1);
    stride_h_output = output.stride(1);
    stride_h_output_sf = output_sf.stride(1);
    stride_h_output_sf_row = output_sf_row.stride(1);

    CHECK_SHAPE(output, batch_size, num_heads, head_dim, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 2);
    CHECK_SHAPE(output_sf, batch_size, num_heads, head_dim, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 16);
    CHECK_SHAPE(output_sf_row, batch_size, num_heads, head_dim, 1);
  }

  auto input_dtype = input.scalar_type();
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1);
      dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);

      scaled_fp4_quant_trans_two_level_kernel<HEAD_DIM, BLOCK_SIZE, c_type>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<c_type*>(input.data_ptr()),
              reinterpret_cast<uint8_t*>(output.data_ptr()),
              reinterpret_cast<uint8_t*>(output_sf.data_ptr()),
              reinterpret_cast<uint8_t*>(output_sf_row.data_ptr()),
              batch_size, num_heads, num_tokens,
              stride_bz_input, stride_h_input, stride_seq_input,
              stride_bz_output, stride_h_output, stride_d_output,
              stride_bz_output_sf, stride_h_output_sf, stride_d_output_sf,
              stride_bz_output_sf_row, stride_h_output_sf_row, stride_d_output_sf_row);
    });
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scaled_fp4_quant", &scaled_fp4_quant);
  m.def("scaled_fp4_quant_permute", &scaled_fp4_quant_permute);
  m.def("scaled_fp4_quant_trans", &scaled_fp4_quant_trans);
  m.def("scaled_fp4_quant_two_level", &scaled_fp4_quant_two_level);
  m.def("scaled_fp4_quant_permute_two_level", &scaled_fp4_quant_permute_two_level);
  m.def("scaled_fp4_quant_trans_two_level", &scaled_fp4_quant_trans_two_level);
}