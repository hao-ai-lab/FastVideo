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
#include <cuda_runtime.h>
#include "cutlass/numeric_conversion.h"

#include "common/load.hpp"
#include "common/store.hpp"
#include "common/launch.hpp"

template <
  class InputDtype_, 
  int NumThrPerCta_,
  bool IsEvenM,
  bool IsEvenN
>
class Quantization {
public:
  using InputDtype = InputDtype_;
  using OutputDtype = int8_t;
  using FPConverter = cutlass::NumericConverter<int8_t, float, cutlass::FloatRoundStyle::round_to_nearest>;

  static constexpr int BlockSize = 128;
  static constexpr int NumThrPerCta = NumThrPerCta_;
  static constexpr int NumElementPerThread = BlockSize * BlockSize / NumThrPerCta;
  static constexpr int NumThrPerRow = BlockSize / NumElementPerThread;

  static_assert(BlockSize * BlockSize % NumThrPerCta == 0);
  static_assert(NumThrPerCta % BlockSize == 0);

  static constexpr size_t ShmSize = 32;

  static constexpr float int8_max = 128.f;

  struct Params {
    void const *Iptr;
    void *Optr;
    void *OSptr;
    int64_t const m;
    int64_t const n;
  };

  using Arguments = Params;

  static Params to_underlying_arguments(Arguments const& args) {
    return args;
  }

  static dim3 get_grid_size(int64_t m, int64_t n) {
    return dim3(
      cdiv(n, BlockSize),
      cdiv(m, BlockSize) 
    );
  }

  static dim3 get_cta_size(int64_t m, int64_t n) {
    return dim3(
      NumThrPerCta, 1, 1
    );
  }
  
  CUTLASS_DEVICE
  void quantization(
    float *float_reg, 
    void *Optr, void *OSptr, 
    int64_t const m, int64_t const n, 
    int blk_m, int blk_n, int tidx,
    char *shared_data
  ) {

    OutputDtype output_reg[NumElementPerThread];


    Saver<OutputDtype, BlockSize, BlockSize, NumThrPerCta, IsEvenM, IsEvenN> saver;

    float amax = _reduce_amax(float_reg, (float*)shared_data);
    
    
    _quantization(float_reg, output_reg, int8_max / amax);
    
    float scale_inv = amax / int8_max;

    saver.store(Optr, OSptr, output_reg, scale_inv, m, n, blk_m, blk_n, tidx);

    __syncthreads();
    }
  

  CUTLASS_DEVICE 
  void operator()(Params const& params, char *shared_data) {
    int blk_m = blockIdx.y;
    int blk_n = blockIdx.x;
    int tidx = threadIdx.x;

    float float_reg[NumElementPerThread];

    // load float32 data
    Loader<InputDtype, BlockSize, BlockSize, NumThrPerCta, IsEvenM, IsEvenN> loader;
    loader.load(params.Iptr, float_reg, params.m, params.n, blk_m, blk_n, tidx);
    quantization(
      float_reg, params.Optr, params.OSptr, params.m, params.n, blk_m, blk_n, tidx, shared_data
    );
  }

private: 

  CUTLASS_DEVICE float 
  _reduce_amax(float *reg, float *smem_ptr) {
    float amax = 1e-8;
    // thread reduction
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NumElementPerThread; ++i)
      amax = max(amax, fabs(reg[i]));

    __syncwarp();

    // warp reduction
    CUTLASS_PRAGMA_UNROLL
    for (int i = 16; i >= 1; i /= 2) {
      amax = max(
        __shfl_xor_sync(0xffffffff, amax, i, 32),
        amax
      );
    }

    // cta reduction
    if (threadIdx.x == 0) {
      *smem_ptr = 0;
    }
    __syncthreads();

    atomicMax((uint32_t*)smem_ptr, reinterpret_cast<const uint32_t&>(amax));

    __syncthreads();

    amax = *smem_ptr;

    return amax;
  }

  CUTLASS_DEVICE void
  _quantization(float *float_reg, OutputDtype *out_reg, float scale) {
    FPConverter converter;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NumElementPerThread; ++i) {
        out_reg[i] = converter(float_reg[i] * scale);
    }

  }
};

template <
  class InputDtype,
  int BlockSize,
  int NumThrPerCta
>
bool quantization(
  void const *Iptr, void *Optr, void *OSptr,
  int64_t m, int64_t n, 
  cudaStream_t stream = nullptr
) {
  BOOL_SWITCH(m % BlockSize == 0, IsEvenM, [&] {
    BOOL_SWITCH(n % BlockSize == 0, IsEvenN, [&] {
      using Kernel = Quantization<
        InputDtype, NumThrPerCta, IsEvenM, IsEvenN>;
      using Arguments = typename Kernel::Arguments;
      Arguments args = {
        Iptr, Optr, OSptr,
        m, n
      };
      auto params = Kernel::to_underlying_arguments(args);
      auto grid_shape = Kernel::get_grid_size(m, n);
      auto cta_shape = Kernel::get_cta_size(m, n);
      static constexpr size_t ShmSize = Kernel::ShmSize;
      launch_kernel<Kernel>(params, grid_shape, cta_shape, ShmSize, stream);
    });
  });

  return true;
}
