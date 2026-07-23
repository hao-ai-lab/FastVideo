#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

torch::Tensor rms_norm_cuda(
    torch::Tensor const& input,
    double eps,
    std::optional<torch::Tensor> const& weight,
    std::optional<torch::Tensor> const& output) {
  std::optional<at::Tensor> mutable_output = output;
  auto result = rms_norm(input, static_cast<float>(eps), weight, mutable_output);
  TORCH_CHECK(result.has_value(), "rms_norm_cuda did not produce an output tensor");
  return result.value();
}

torch::Tensor layer_norm_cuda(
    torch::Tensor const& input,
    double eps,
    std::optional<torch::Tensor> const& weight,
    std::optional<torch::Tensor> const& bias,
    std::optional<torch::Tensor> const& output) {
  auto result = layer_norm(input, static_cast<float>(eps), weight, bias, output);
  TORCH_CHECK(result.has_value(), "layer_norm_cuda did not produce an output tensor");
  return result.value();
}

std::tuple<torch::Tensor, torch::Tensor> quant_cuda(torch::Tensor const& input) {
  std::optional<torch::Tensor> output;
  std::optional<torch::Tensor> output_scale;
  auto result = quant(input, output, output_scale);
  TORCH_CHECK(std::get<0>(result).has_value(), "quant_cuda did not produce an int8 tensor");
  TORCH_CHECK(std::get<1>(result).has_value(), "quant_cuda did not produce a scale tensor");
  return std::make_tuple(std::get<0>(result).value(), std::get<1>(result).value());
}

torch::Tensor gemm_cuda(
    torch::Tensor const& a,
    torch::Tensor const& a_scale,
    torch::Tensor const& b,
    torch::Tensor const& b_scale,
    torch::Tensor c) {
  int8_gemm(a, a_scale, b, b_scale, c);
  return c;
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "sta_fwd(Tensor q, Tensor k, Tensor v, Tensor! out, int kernel_t_size, int kernel_h_size, "
      "int kernel_w_size, int text_length, bool process_text, bool has_text, "
      "int kernel_aspect_ratio_flag) -> Tensor");
#if defined(CUDA_KERNEL)
  ops.impl("sta_fwd", torch::kCUDA, &sta_forward);
#endif

  ops.def(
      "block_sparse_fwd(Tensor q, Tensor k, Tensor v, Tensor q2k_block_sparse_index, "
      "Tensor q2k_block_sparse_num, Tensor kv_block_size) -> Tensor[]");
#if defined(CUDA_KERNEL)
  ops.impl("block_sparse_fwd", torch::kCUDA, &block_sparse_attention_forward);
#endif

  ops.def(
      "block_sparse_bwd(Tensor q, Tensor k, Tensor v, Tensor out, Tensor l_vec, Tensor out_grad, "
      "Tensor k2q_block_sparse_index, Tensor k2q_block_sparse_num, Tensor kv_block_size) -> Tensor[]");
#if defined(CUDA_KERNEL)
  ops.impl("block_sparse_bwd", torch::kCUDA, &block_sparse_attention_backward);
#endif

  ops.def("rms_norm_cuda(Tensor input, float eps, Tensor? weight=None, Tensor? output=None) -> Tensor");
#if defined(CUDA_KERNEL)
  ops.impl("rms_norm_cuda", torch::kCUDA, &rms_norm_cuda);
#endif

  ops.def(
      "layer_norm_cuda(Tensor input, float eps, Tensor? weight=None, Tensor? bias=None, "
      "Tensor? output=None) -> Tensor");
#if defined(CUDA_KERNEL)
  ops.impl("layer_norm_cuda", torch::kCUDA, &layer_norm_cuda);
#endif

  ops.def("quant_cuda(Tensor input) -> (Tensor, Tensor)");
#if defined(CUDA_KERNEL)
  ops.impl("quant_cuda", torch::kCUDA, &quant_cuda);
#endif

  ops.def("gemm_cuda(Tensor a, Tensor a_scale, Tensor b, Tensor b_scale, Tensor! c) -> Tensor");
#if defined(CUDA_KERNEL)
  ops.impl("gemm_cuda", torch::kCUDA, &gemm_cuda);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
