#pragma once

#include "common/load.hpp"
#include "common/store.hpp"
#include "common/launch.hpp"


template <
  class InputDtype_,
  class OutputDtype_,
  class WeightDtype_,
  bool Affine_,
  bool Bias_,
  int MaxHiddenSize_,
  int NumThrPerCta_,
  bool IsEven
>
class LayerNorm {
public:
  using InputDtype = InputDtype_;
  using OutputDtype = OutputDtype_;
  using WeightDtype = WeightDtype_;
  static constexpr int NumThrPerCta = NumThrPerCta_;
  static constexpr int MaxHiddenSize = MaxHiddenSize_;
  static constexpr bool Affine = Affine_;
  static constexpr bool Bias = Bias_;

  static constexpr size_t ShmSize = 32;
  static constexpr int NumElementPerThread = MaxHiddenSize / NumThrPerCta;

  static_assert(MaxHiddenSize % NumThrPerCta == 0);

  struct Params {
    void const *Iptr;
    void const *Wptr;
    void const *Bptr;
    void *Optr;
    float eps;
    int64_t m;
    int64_t n;
  };

  using Arguments = Params;

  static Params to_underlying_arguments(Arguments const& args) {
    return args;
  }

  static dim3 get_grid_size(int64_t m, int64_t n) {
    return dim3(m);
  }

  static dim3 get_cta_size(int64_t m, int64_t n) {
    return dim3(NumThrPerCta, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char *shared_data) {
    int const blk_m = blockIdx.x;
    int const blk_n = 1;
    int tidx = threadIdx.x;
    float x[NumElementPerThread];

    // load
    Loader<InputDtype, 1, MaxHiddenSize, NumThrPerCta, true, IsEven> loader;
    loader.load(params.Iptr, x, params.m, params.n, blk_m, 0, tidx);

    // mean reduction
    float u = _reduce_sum(x, shared_data) / params.n;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NumElementPerThread; ++i)
      x[i] -= u;

    __syncthreads();
    // var reduction
    float v = sqrtf(_reduce_square(x, shared_data) / params.n + params.eps);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NumElementPerThread; ++i)
      x[i] /= v;

    if constexpr (Affine) {
      // load weight
      Loader<WeightDtype, 1, MaxHiddenSize, NumThrPerCta, true, IsEven> weight_loader;
      float w[NumElementPerThread];
      weight_loader.load(params.Wptr, w, 1, params.n, 0, 0, tidx);
      if constexpr (Bias) {
        float b[NumElementPerThread];
        weight_loader.load(params.Bptr, b, 1, params.n, 0, 0, tidx);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i)
          x[i] = x[i] * w[i] + b[i];
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i)
          x[i] = x[i] * w[i];
      }
    }

    // save y
    {
      Saver<OutputDtype, 1, MaxHiddenSize, NumThrPerCta, true, IsEven, false, false> saver;
      if constexpr (std::is_same_v<OutputDtype, float>) {
        saver.store(params.Optr, nullptr, x, 0, params.m, params.n, blk_m, 0, tidx);
      } else {
        OutputDtype tmp[NumElementPerThread];
        for (int i = 0; i < NumElementPerThread; ++i)
          tmp[i] = OutputDtype(x[i]);
        saver.store(params.Optr, nullptr, tmp, 0, params.m, params.n, blk_m, 0, tidx);
      }
    }
    
  }

private:
  CUTLASS_DEVICE
  float _reduce_square(float *reg, char *shared_data) {
    // thread
    float sum_square = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NumElementPerThread; ++i)
      sum_square += reg[i] * reg[i];

    CUTLASS_PRAGMA_UNROLL
    for (int i = 16; i >= 1; i >>= 1) {
      sum_square += __shfl_down_sync(0xFFFFFFFF, sum_square, i);
    }
    if (threadIdx.x == 0) {
      *(float*)shared_data = 0;
    }
    __syncthreads();

    if (threadIdx.x % 32 == 0) {
      atomicAdd((float*)shared_data, sum_square);
    }

    __syncthreads();
    sum_square = *(float*)shared_data;
    return sum_square;
  }

  CUTLASS_DEVICE
  float _reduce_sum(float *reg, char *shared_data) {
    // thread
    float sum = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NumElementPerThread; ++i)
      sum += reg[i];

    CUTLASS_PRAGMA_UNROLL
    for (int i = 16; i >= 1; i >>= 1) {
      sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
    }
    if (threadIdx.x == 0) {
      *(float*)shared_data = 0;
    }
    __syncthreads();

    if (threadIdx.x % 32 == 0) {
      atomicAdd((float*)shared_data, sum);
    }

    __syncthreads();
    sum = *(float*)shared_data;
    return sum;
  }
};

template <
  class InputDtype,
  class OutputDtype,
  class WeightDtype,
  bool Affine,
  bool Bias,
  int MaxHiddenSize,
  int NumThrPerCta
>
bool layernorm(
  void const *Iptr, void const *Wptr, void const *Bptr,
  void *Optr, float eps, int64_t m, int64_t n, 
  cudaStream_t stream = nullptr
) {
  BOOL_SWITCH(n % MaxHiddenSize == 0, IsEven, [&] {
    using Kernel = LayerNorm<
      InputDtype, OutputDtype, WeightDtype,
      Affine, Bias,
      MaxHiddenSize, NumThrPerCta,
      IsEven>;
    using Arguments = typename Kernel::Arguments;
    Arguments args = {
      Iptr, Wptr, Bptr, Optr, 
      eps, m, n
    };
    auto params = Kernel::to_underlying_arguments(args);
    auto grid_shape = Kernel::get_grid_size(m, n);
    auto cta_shape = Kernel::get_cta_size(m, n);
    static constexpr size_t ShmSize = Kernel::ShmSize;
    launch_kernel<Kernel>(params, grid_shape, cta_shape, ShmSize, stream);
  });
  return true;
}