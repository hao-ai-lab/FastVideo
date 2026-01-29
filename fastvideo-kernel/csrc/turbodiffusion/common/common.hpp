#pragma once

#include <torch/torch.h>
#include <torch/extension.h>

CUTLASS_HOST_DEVICE int64_t cdiv(int64_t const& a, int64_t const &b) {
  return (a + b - 1) / b;
}

template <class T>
CUTLASS_HOST_DEVICE T max(T a, T b) { return a > b ? a : b; }

template <class T>
CUTLASS_HOST_DEVICE T min(T a, T b) { return a > b ? b : a; }

#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define BOOL_SWITCH(COND, CONST_NAME, ...)    \
[&] {                                         \
  if (COND) {                                 \
    static constexpr bool CONST_NAME = true;  \
    return (__VA_ARGS__)();                   \
  } else {                                    \
    static constexpr bool CONST_NAME = false; \
    return (__VA_ARGS__)();                   \
  }                                           \
}() 

#define CUDA_CHECK(call)    \
{                           \
  cudaError_t err = call;   \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(err);              \
  }                         \
}

#define CONFIG_SWITCH(N, ...)                                   \
[&] {                                                           \
  if (N <= 1024) {                                              \
    constexpr int NUM_THR_PER_CTA = 128;                        \
    constexpr int MAX_HIDDEN_SIZE = 1024;                       \
    return (__VA_ARGS__)();                                     \
  } else if (N <= 2048) {                                       \
    constexpr int NUM_THR_PER_CTA = 128;                        \
    constexpr int MAX_HIDDEN_SIZE = 2048;                       \
    return (__VA_ARGS__)();                                     \
  } else if (N <= 4096) {                                       \
    constexpr int NUM_THR_PER_CTA = 128;                        \
    constexpr int MAX_HIDDEN_SIZE = 4096;                       \
    return (__VA_ARGS__)();                                     \
  } else if (N <= 8192) {                                       \
    constexpr int NUM_THR_PER_CTA = 256;                        \
    constexpr int MAX_HIDDEN_SIZE = 8192;                       \
    return (__VA_ARGS__)();                                     \
  } else {                                                      \
    constexpr int NUM_THR_PER_CTA = 256;                        \
    constexpr int MAX_HIDDEN_SIZE = 16384;                      \
    return (__VA_ARGS__)();                                     \
  }                                                             \
}()


template <int BlockSize>
  void create_tensor(
    torch::Device const &device,
    std::optional<at::Tensor> &output,
    std::optional<at::Tensor> &scale,
    int m, int n
  ) {
    int num_block_m = cdiv(m, BlockSize);
    int num_block_n = cdiv(n, BlockSize);
    if (!output.has_value()) {
      output.emplace(torch::empty(
        {m, n},
        torch::TensorOptions().device(device).dtype(torch::kInt8)
      ));
      scale.emplace(torch::empty(
        {num_block_m, num_block_n},
        torch::TensorOptions().device(device).dtype(torch::kFloat32)
      ));
    }
  }

