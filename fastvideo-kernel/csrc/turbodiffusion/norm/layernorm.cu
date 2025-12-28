#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/python.h>
#include <cutlass/cutlass.h>
#include "common/common.hpp"
#include "norm/layernorm.hpp"

auto layer_norm(
  at::Tensor const Input, 
  float eps,
  std::optional<at::Tensor const> W,
  std::optional<at::Tensor const> const B,
  std::optional<at::Tensor> Output
) {
  using ElementIn = float;
  using ElementOut = float;
  using ElementWeight = float;

  int64_t const m = Input.size(0);
  int64_t const n = Input.size(1);
  torch::Device const input_device = Input.device();

  if (!Output.has_value()) {
    Output.emplace(
      torch::empty(
        {m, n},
        torch::TensorOptions().device(input_device).dtype(torch::kFloat32)
      )
    );
  }


  void *Iptr = Input.data_ptr();
  void *Wptr = W.has_value() ? W.value().data_ptr() : nullptr;
  void *Bptr = B.has_value() ? B.value().data_ptr() : nullptr;
  void *Optr = Output.value().data_ptr();

  BOOL_SWITCH(B.has_value(), BIAS, [&]{
    BOOL_SWITCH(W.has_value(), AFFINE, [&]{
    CONFIG_SWITCH(n, [&]{
    layernorm<
    ElementIn, ElementOut, ElementWeight,
    AFFINE, BIAS,
    MAX_HIDDEN_SIZE, NUM_THR_PER_CTA> (
      Iptr, Wptr, Bptr,
      Optr, eps, m, n,
      at::cuda::getCurrentCUDAStream().stream()
        );
      });
    });
  });
  
    

  return Output;
}

void register_layer_norm(pybind11::module_ &m) {
    m.def("layer_norm_cuda", &layer_norm);
}

