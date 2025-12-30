/* 
 * Copyright (c) 2025 by TurboDiffusion team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * 
 * Citation (please cite if you use this code):
 * 
 * @article{zhang2025turbodiffusion,
 *   title={TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times},
 *   author={Zhang, Jintao and Zheng, Kaiwen and Jiang, Kai and Wang, Haoxu and Stoica, Ion and Gonzalez, Joseph E and Chen, Jianfei and Zhu, Jun},
 *   journal={arXiv preprint arXiv:2512.16093},
 *   year={2025}
 * }
 */

#pragma once

#include <cuda.h>
#include "cute/tensor.hpp"

template <
  bool IsEven, 
  class TiledCopy,
  class SrcTensor,
  class DstTensor,
  class PrdTensor
>
CUTLASS_DEVICE void
copy_AB(
    TiledCopy const& _copy, 
    SrcTensor const &S, 
    DstTensor &D, 
    PrdTensor const &ID, 
    const int64_t &i_read, 
    const int64_t &i_write, 
    const int64_t &limit
) {
  using namespace cute;
  if constexpr (IsEven)
    cute::copy(_copy, S(_, _, _, i_read), D(_, _, _, i_write));
  else {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(ID); ++i)
      if (get<0>(ID(0, i, 0)) < limit)
        cute::copy(_copy, S(_, i, _, i_read), D(_, i, _, i_write));
  }
}

template <int N>
CUTLASS_DEVICE void copy_async(
  void const* gmem_src,
  void* smem_dst
) {
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));;
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
      :: "r"(smem_int_ptr),
          "l"(gmem_src),
          "n"(N));
}

template<class LoadType, class T, int NumThreads>
CUTLASS_DEVICE void copy_aligned(const void* src, void* dst, size_t N, int64_t thread_idx) {
  static constexpr int NumElementPerLoad = sizeof(LoadType) / sizeof(T);
  for (int64_t i = thread_idx * NumElementPerLoad; i < N; i += NumElementPerLoad * NumThreads) {
    if (i + NumElementPerLoad <= N) {
      copy_async<sizeof(LoadType)>(
        (void*)((T*)src + i),
        (void*)((T*)dst + i)
      );
    } else {
      for (int64_t j = 0; j < N - i; ++j)
        copy_async<sizeof(T)>(
          (void*)((T*)src + i + j),
          (void*)((T*)dst + i + j)
        );
    }
  }
}

template<class T, int NumThreads, bool Wait = true, bool Commit = true>
CUTLASS_DEVICE void g2s_vector_copy(const void* src, void* dst, size_t N, int64_t thread_idx) {
  
  uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);

  if (src_addr % 16 == 0) {
      copy_aligned<int4, T, NumThreads>(src, dst, N, thread_idx);
  } else if (src_addr % 8 == 0) {
      copy_aligned<int2, T, NumThreads>(src, dst, N, thread_idx);
  } else if (src_addr % 4 == 0) {
      copy_aligned<int, T, NumThreads>(src, dst, N, thread_idx);
  } else {
    assert(0);
  }
  if constexpr (Commit) {
    asm volatile("cp.async.commit_group;\n" ::);
  }
  if constexpr (Wait) {
    asm volatile("cp.async.wait_all;\n" ::);
  }
}

template<class T, int N, bool FastInt2Float>
CUTLASS_DEVICE
static void dequant(
  T* mma_accum_ptr,
  float* float_accum_ptr,
  float scale
) {
  static int const ic = 0x4B400000;
  if constexpr (FastInt2Float && std::is_same_v<T, int32_t>) {
    CUTLASS_PRAGMA_UNROLL
    for (size_t i = 0; i < N; ++i) {
      *(float_accum_ptr + i) += (__int_as_float(*(mma_accum_ptr + i)) - __int_as_float(ic)) * scale;
      *(mma_accum_ptr + i) = ic;
    }
  } else if constexpr (std::is_same_v<T, int32_t>) {
    CUTLASS_PRAGMA_UNROLL
    for (size_t i = 0; i < N; ++i) {
      *(float_accum_ptr + i) += __int2float_rn(*(mma_accum_ptr + i)) * scale;
      *(mma_accum_ptr + i) = 0;
    }
  } else {
    CUTLASS_PRAGMA_UNROLL
    for (size_t i = 0; i < N; ++i) {
      *(float_accum_ptr + i) += (*(mma_accum_ptr + i)) * scale;
      *(mma_accum_ptr + i) = 0;
    }
  }
}