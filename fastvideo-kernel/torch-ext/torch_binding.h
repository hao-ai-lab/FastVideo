#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include <torch/torch.h>

torch::Tensor sta_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    int kernel_t_size,
    int kernel_h_size,
    int kernel_w_size,
    int text_length,
    bool process_text,
    bool has_text,
    int kernel_aspect_ratio_flag);

std::vector<torch::Tensor> block_sparse_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor q2k_block_sparse_index,
    torch::Tensor q2k_block_sparse_num,
    torch::Tensor kv_block_size);

std::vector<torch::Tensor> block_sparse_attention_backward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    torch::Tensor l_vec,
    torch::Tensor og,
    torch::Tensor k2q_block_sparse_index,
    torch::Tensor k2q_block_sparse_num,
    torch::Tensor kv_block_size);

std::optional<at::Tensor> rms_norm(
    at::Tensor const& input,
    float eps,
    std::optional<at::Tensor> const& weight,
    std::optional<at::Tensor>& output);

std::optional<at::Tensor> layer_norm(
    at::Tensor const input,
    float eps,
    std::optional<at::Tensor> weight,
    std::optional<at::Tensor> const bias,
    std::optional<at::Tensor> output);

std::tuple<std::optional<torch::Tensor>, std::optional<torch::Tensor>> quant(
    torch::Tensor const& input,
    std::optional<torch::Tensor>& output,
    std::optional<torch::Tensor>& output_scale);

void int8_gemm(
    at::Tensor const& a,
    at::Tensor const& a_scale,
    at::Tensor const& b,
    at::Tensor const& b_scale,
    torch::Tensor& c);

torch::Tensor rms_norm_cuda(
    torch::Tensor const& input,
    double eps,
    std::optional<torch::Tensor> const& weight,
    std::optional<torch::Tensor> const& output);

torch::Tensor layer_norm_cuda(
    torch::Tensor const& input,
    double eps,
    std::optional<torch::Tensor> const& weight,
    std::optional<torch::Tensor> const& bias,
    std::optional<torch::Tensor> const& output);

std::tuple<torch::Tensor, torch::Tensor> quant_cuda(torch::Tensor const& input);

torch::Tensor gemm_cuda(
    torch::Tensor const& a,
    torch::Tensor const& a_scale,
    torch::Tensor const& b,
    torch::Tensor const& b_scale,
    torch::Tensor c);
