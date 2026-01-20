#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/python.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <pybind11/pybind11.h>

#include "common/common.hpp"
#include "norm/rmsnorm.hpp"

auto rms_norm(
  at::Tensor const& Input, 
  float eps,
  const std::optional<at::Tensor>& Weight,
  std::optional<at::Tensor>& Output
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
  if (Weight.has_value()) {
    TORCH_CHECK(Weight.value().scalar_type() == Input.scalar_type(),
                "Weight dtype must match Input dtype. Got Weight=",
                Weight.value().scalar_type(), ", Input=", Input.scalar_type());
  }

  void *Iptr = Input.data_ptr();
  void *Wptr = Weight.has_value() ? Weight.value().data_ptr() : nullptr;
  void *Optr = Output.value().data_ptr();

  if (Input.scalar_type() == at::kHalf) {
    using ElementIn = cutlass::half_t;
    using ElementOut = cutlass::half_t;
    using ElementWeight = cutlass::half_t;
    CONFIG_SWITCH(n, [&]{
      rmsnorm<ElementIn, ElementOut, ElementWeight, MAX_HIDDEN_SIZE, NUM_THR_PER_CTA>(
          Iptr, Wptr, Optr, eps, m, n, at::cuda::getCurrentCUDAStream().stream());
    });
  } else if (Input.scalar_type() == at::kBFloat16) {
    using ElementIn = cutlass::bfloat16_t;
    using ElementOut = cutlass::bfloat16_t;
    using ElementWeight = cutlass::bfloat16_t;
    CONFIG_SWITCH(n, [&]{
      rmsnorm<ElementIn, ElementOut, ElementWeight, MAX_HIDDEN_SIZE, NUM_THR_PER_CTA>(
          Iptr, Wptr, Optr, eps, m, n, at::cuda::getCurrentCUDAStream().stream());
    });
  } else if (Input.scalar_type() == at::kFloat) {
    using ElementIn = float;
    using ElementOut = float;
    using ElementWeight = float;
    CONFIG_SWITCH(n, [&]{
      rmsnorm<ElementIn, ElementOut, ElementWeight, MAX_HIDDEN_SIZE, NUM_THR_PER_CTA>(
          Iptr, Wptr, Optr, eps, m, n, at::cuda::getCurrentCUDAStream().stream());
    });
  } else {
    TORCH_CHECK(false, "Unsupported dtype for rms_norm_cuda: ", Input.scalar_type());
  }
  

  return Output;
}

void register_rms_norm(pybind11::module_ &m) {
    m.def("rms_norm_cuda", &rms_norm);
}
