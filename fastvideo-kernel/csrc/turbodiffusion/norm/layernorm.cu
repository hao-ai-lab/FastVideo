#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/python.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "common/common.hpp"
#include "norm/layernorm.hpp"

auto layer_norm(
  at::Tensor const Input, 
  float eps,
  std::optional<at::Tensor const> W,
  std::optional<at::Tensor const> const B,
  std::optional<at::Tensor> Output
) {
  int64_t const m = Input.size(0);
  int64_t const n = Input.size(1);
  torch::Device const input_device = Input.device();

  if (!Output.has_value()) {
    Output.emplace(
      torch::empty(
        {m, n},
        torch::TensorOptions().device(input_device).dtype(Input.scalar_type())
      )
    );
  }

  TORCH_CHECK(Output.value().scalar_type() == Input.scalar_type(),
              "Output dtype must match Input dtype. Got Output=",
              Output.value().scalar_type(), ", Input=", Input.scalar_type());
  if (W.has_value()) {
    TORCH_CHECK(W.value().scalar_type() == Input.scalar_type(),
                "W dtype must match Input dtype. Got W=",
                W.value().scalar_type(), ", Input=", Input.scalar_type());
  }
  if (B.has_value()) {
    TORCH_CHECK(B.value().scalar_type() == Input.scalar_type(),
                "B dtype must match Input dtype. Got B=",
                B.value().scalar_type(), ", Input=", Input.scalar_type());
  }

  void *Iptr = Input.data_ptr();
  void *Wptr = W.has_value() ? W.value().data_ptr() : nullptr;
  void *Bptr = B.has_value() ? B.value().data_ptr() : nullptr;
  void *Optr = Output.value().data_ptr();

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (Input.scalar_type() == at::kHalf) {
    using ElementIn = cutlass::half_t;
    using ElementOut = cutlass::half_t;
    using ElementWeight = cutlass::half_t;
    BOOL_SWITCH(B.has_value(), BIAS, [&]{
      BOOL_SWITCH(W.has_value(), AFFINE, [&]{
        CONFIG_SWITCH(n, [&]{
          layernorm<ElementIn, ElementOut, ElementWeight, AFFINE, BIAS, MAX_HIDDEN_SIZE, NUM_THR_PER_CTA>(
              Iptr, Wptr, Bptr, Optr, eps, m, n, stream);
        });
      });
    });
  } else if (Input.scalar_type() == at::kBFloat16) {
    using ElementIn = cutlass::bfloat16_t;
    using ElementOut = cutlass::bfloat16_t;
    using ElementWeight = cutlass::bfloat16_t;
    BOOL_SWITCH(B.has_value(), BIAS, [&]{
      BOOL_SWITCH(W.has_value(), AFFINE, [&]{
        CONFIG_SWITCH(n, [&]{
          layernorm<ElementIn, ElementOut, ElementWeight, AFFINE, BIAS, MAX_HIDDEN_SIZE, NUM_THR_PER_CTA>(
              Iptr, Wptr, Bptr, Optr, eps, m, n, stream);
        });
      });
    });
  } else if (Input.scalar_type() == at::kFloat) {
    using ElementIn = float;
    using ElementOut = float;
    using ElementWeight = float;
    BOOL_SWITCH(B.has_value(), BIAS, [&]{
      BOOL_SWITCH(W.has_value(), AFFINE, [&]{
        CONFIG_SWITCH(n, [&]{
          layernorm<ElementIn, ElementOut, ElementWeight, AFFINE, BIAS, MAX_HIDDEN_SIZE, NUM_THR_PER_CTA>(
              Iptr, Wptr, Bptr, Optr, eps, m, n, stream);
        });
      });
    });
  } else {
    TORCH_CHECK(false, "Unsupported dtype for layer_norm_cuda: ", Input.scalar_type());
  }
  
    

  return Output;
}

void register_layer_norm(pybind11::module_ &m) {
    m.def("layer_norm_cuda", &layer_norm);
}

