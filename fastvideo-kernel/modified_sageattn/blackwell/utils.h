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

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator> 
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, bool warp_reduce=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
    if constexpr (warp_reduce) { quad_allreduce_(sum, sum, sum_op); }
}

__forceinline__ __device__ __half2 half_exp(__half2 x) {
    uint32_t tmp_out, tmp_in;
    tmp_in = reinterpret_cast<uint32_t&>(x);
    asm ("ex2.approx.f16x2 %0, %1;\n"
      : "=r"(tmp_out)
      : "r"(tmp_in));
    __half2 out = reinterpret_cast<__half2&>(tmp_out);
    return out;
}

// Apply the exp to all the elements.
template <bool zero_init=false, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void max_scale_exp2_sum(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, Tensor<Engine1, Layout1> &sum, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor"); static_assert(Layout1::rank == 1, "Only support 1D Tensor"); CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        MaxOp<float> max_op;
        max(mi) = zero_init ? tensor(mi, 0) : max_op(max(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            max(mi) = max_op(max(mi), tensor(mi, ni));
        }
        max(mi) = Allreduce<4>::run(max(mi), max_op);
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * scale;
        sum(mi) = 0;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            sum(mi) += tensor(mi, ni);
        }
    }
}

// Apply the exp to all the elements.
template <bool Scale_max=true, bool Check_inf=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = Check_inf
            ? (max(mi) == -INFINITY ? 0.f : (max(mi) * (Scale_max ? scale : float(M_LOG2E))))
            : (max(mi) * (Scale_max ? scale : float(M_LOG2E)));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

CUTLASS_DEVICE void
packed_float_to_ue4m3(
  float const &f0, float const &f1, float const &f2, float const &f3,
  uint32_t &out
) {
  asm volatile( \
    "{\n" \
    ".reg .b16 lo;\n" \
    ".reg .b16 hi;\n" \
    "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n" \
    "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n" \
    "mov.b32 %0, {lo, hi};\n" \
    "}" \
    : "=r"(out) : "f"(f0), "f"(f1), "f"(f2), "f"(f3));
}

CUTLASS_DEVICE void 
packed_float_to_e2m1(
  float const &f0, float const &f1, float const &f2, float const& f3,
  float const &f4, float const &f5, float const &f6, float const& f7,
  uint32_t &out
) {

    asm volatile( \
    "{\n" \
    ".reg .b8 byte0;\n" \
    ".reg .b8 byte1;\n" \
    ".reg .b8 byte2;\n" \
    ".reg .b8 byte3;\n" \
    "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n" \
    "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n" \
    "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n" \
    "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n" \
    "mov.b32 %0, {byte0, byte1, byte2, byte3};\n" \
    "}" \
    : "=r"(out) : "f"(f0), "f"(f1), "f"(f2), "f"(f3),
                  "f"(f4), "f"(f5), "f"(f6), "f"(f7));

}

CUTLASS_DEVICE void 
add(float2      & c,
    float2 const& a, 
    float2 const& b) 
{
asm volatile("add.f32x2 %0, %1, %2;\n"
  : "=l"(reinterpret_cast<uint64_t      &>(c))
  :  "l"(reinterpret_cast<uint64_t const&>(a)),
      "l"(reinterpret_cast<uint64_t const&>(b)));
}

CUTLASS_DEVICE void 
add_inplace(float2 &a, 
            float2 const& b) 
{
  asm volatile("add.f32x2 %0, %0, %1;\n"
    : "+l"(reinterpret_cast<uint64_t &>(a))     // a: input/output
    :  "l"(reinterpret_cast<uint64_t const&>(b)) // b: input
  );
}


CUTLASS_DEVICE void 
sub(float2      & c,
    float2 const& a, 
    float2 const& b) 
{
asm volatile("sub.f32x2 %0, %1, %2;\n"
  : "=l"(reinterpret_cast<uint64_t      &>(c))
  :  "l"(reinterpret_cast<uint64_t const&>(a)),
      "l"(reinterpret_cast<uint64_t const&>(b)));
}

CUTLASS_DEVICE void 
sub_inplace(float2 &a, 
            float2 const& b) 
{
  asm volatile("sub.f32x2 %0, %0, %1;\n"
    : "+l"(reinterpret_cast<uint64_t &>(a))     // a: input/output
    :  "l"(reinterpret_cast<uint64_t const&>(b)) // b: input
  );
}


CUTLASS_DEVICE void 
mul(float2      & c,
    float2 const& a, 
    float2 const& b) 
{
  asm volatile("mul.f32x2 %0, %1, %2;\n"
    : "=l"(reinterpret_cast<uint64_t      &>(c))
    :  "l"(reinterpret_cast<uint64_t const&>(a)),
       "l"(reinterpret_cast<uint64_t const&>(b)));
}

CUTLASS_DEVICE void 
fma(float2      & d,
    float2 const& a, 
    float2 const& b, 
    float2 const& c) 
{
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
    : "=l"(reinterpret_cast<uint64_t      &>(d))
    :  "l"(reinterpret_cast<uint64_t const&>(a)),
       "l"(reinterpret_cast<uint64_t const&>(b)),
       "l"(reinterpret_cast<uint64_t const&>(c)));
}

CUTLASS_DEVICE void 
fma_inplace(float2 &a, 
            float2 const& b, 
            float2 const& c) 
{
  asm volatile("fma.rn.f32x2 %0, %0, %1, %2;\n"
    : "+l"(reinterpret_cast<uint64_t      &>(a))
    :  "l"(reinterpret_cast<uint64_t const&>(b)),
       "l"(reinterpret_cast<uint64_t const&>(c)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class Layout
>
CUTLASS_DEVICE constexpr
auto convert_to_reduction_layout(Layout mma_layout) { 
  static_assert(rank(mma_layout) == 3, "Mma Layout should be (MmaAtom, MmaM, MmaN)");
  static_assert(rank(get<0>(shape(mma_layout))) == 2, "MmaAtom should be (AtomN, AtomM)");

  return make_layout(
    make_layout(get<0,1>(mma_layout), get<1>(mma_layout)),
    make_layout(get<0,0>(mma_layout), get<2>(mma_layout))
  );
}

template <
  class Tensor
>
CUTLASS_DEVICE constexpr
auto convert_to_reduction_tensor(Tensor mma_tensor) {
  return make_tensor(mma_tensor.data(), convert_to_reduction_layout(mma_tensor.layout()));
}


template <
  class Layout
>
CUTLASS_DEVICE constexpr
auto convert_to_conversion_layout(Layout mma_layout) {
  static_assert(rank(mma_layout) == 3, "Mma Layout should be (MmaAtom, MmaM, MmaN)");
  static_assert(rank(get<0>(shape(mma_layout))) == 2, "MmaAtom should be (AtomN, AtomM)");

  constexpr int MmaAtomN = size<0, 0>(mma_layout);
  constexpr int MmaAtomM = size<0, 1>(mma_layout);
  constexpr int MmaM = size<1>(mma_layout);
  constexpr int MmaN = size<2>(mma_layout);

  static_assert(MmaAtomN == 8, "MmaAtomN should be 8.");
  static_assert(MmaAtomM == 2, "MmaAtomM should be 2.");
  static_assert(MmaN % 2 == 0, "MmaN should be multiple of 2.");

  auto mma_n_division = zipped_divide(
    layout<2>(mma_layout), make_tile(_2{})
  );
  return make_layout(
    make_layout(layout<0,0>(mma_layout), make_layout(layout<0,1>(mma_layout), layout<0>(mma_n_division))),
    layout<1>(mma_layout), layout<1>(mma_n_division)
  );
}

template <
  class Tensor
>
CUTLASS_DEVICE constexpr
auto convert_to_conversion_tensor(Tensor mma_tensor) {
  return make_tensor(mma_tensor.data(), convert_to_conversion_layout(mma_tensor.layout()));
}
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
CUTLASS_DEVICE void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                         Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                         Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

}  // namespace flash
