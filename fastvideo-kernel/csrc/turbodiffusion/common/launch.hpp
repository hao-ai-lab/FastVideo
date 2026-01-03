#pragma once

#include <cuda.h>
#include <cuda_runtime.h>


template <class Kernel>
__global__ void device_kernel(
  __grid_constant__ typename Kernel::Params const params
) {
  extern __shared__ char smem[];
  Kernel op;
  op(params, smem);
}

template <class Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void device_kernel_with_launch_bounds(
  __grid_constant__ typename Kernel::Params const params
) {
  extern __shared__ char smem[];
  Kernel op;
  op(params, smem);
}

template <class Kernel>
void launch_kernel(
  typename Kernel::Params const &params,
  dim3 grid_shape,
  dim3 cta_shape,
  size_t ShmSize,
  cudaStream_t stream = nullptr
) {
  auto func = device_kernel<Kernel>;
  if (ShmSize >= 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(
      func,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      ShmSize
    ));
  }
  func<<<grid_shape, cta_shape, ShmSize, stream>>>(params);
  CUDA_CHECK(cudaGetLastError());
}
