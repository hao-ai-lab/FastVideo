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

#include "common/common.hpp"
#include "common/launch.hpp"
#include "gemm/kernel.hpp"


template <class OutputDtype>
bool int8_gemm_(
  int8_t const *Aptr, float const *ASptr,
  int8_t const *Bptr, float const *BSptr,
  OutputDtype* Dptr, int64_t m, int64_t n, int64_t k,
  int swizzle_dir = 1, int swizzle_size_log = 0,
  cudaStream_t stream = nullptr
) {
  BOOL_SWITCH(m % 128 == 0, IsEvenM, [&] {
    BOOL_SWITCH(n % 128 == 0, IsEvenN, [&] {
      using Kernel = GemmKernel<OutputDtype, IsEvenM, IsEvenN>;
      if (!Kernel::can_implement(m, n, k))
        return false;
      using Args = typename Kernel::Arguments;
      Args args {
        (void*)Aptr, (void*)ASptr,
        (void*)Bptr, (void*)BSptr, (void*)Dptr,
        m, n, k, swizzle_dir,
        swizzle_size_log
      };

      auto params = Kernel::to_underlying_arguments(args);

      static constexpr size_t ShmSize = Kernel::ShmSize;
      dim3 grid_shape = Kernel::get_grid_size(m, n);
      dim3 block_shape = dim3(Kernel::ThreadNum);
      auto func = device_kernel<Kernel>;
      if (ShmSize >= 48 * 1024) {
        cudaFuncSetAttribute(
          func,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          ShmSize
        );
      }
      func<<<grid_shape, block_shape, ShmSize, stream>>>(
        params
      );
      return true;
    });
  });
  return true;
}
