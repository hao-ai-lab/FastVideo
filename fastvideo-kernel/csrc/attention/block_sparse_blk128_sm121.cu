// SPDX-License-Identifier: Apache-2.0
// block_sparse_blk128_sm121.cu -- torch binding for the GB10 (DGX Spark, sm_121a) VSA block-sparse forward,
// 128-token blocks (block_sparse_kernel_blk128_sm121.cuh).
//
// The inference counterpart of block_sparse_ptx_blackwell_blk128_fwd with need_lse=False: it returns {out} only.
// out is allocated here and written whole by the kernel (padding rows and tiles that select nothing are written
// zero), so it never needs to be zero-initialized.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>
#include <vector>

#include "block_sparse_kernel_blk128_sm121.cuh"

namespace {

constexpr int64_t kBlock = 128;
constexpr int64_t kHeadDim = 128;

void check_tensor(const torch::Tensor& t, const char* name, at::ScalarType dtype, const at::Device& device) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.device() == device, name, " is on ", t.device(), ", but q is on ", device);
  TORCH_CHECK(t.scalar_type() == dtype, name, " must be ", dtype, ", not ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

// q, k, v: [B, H, S, 128] bf16, contiguous (BHSD), S = num_blocks * 128. q2k_idx [B, H, num_blocks, max_kv],
// q2k_num [B, H, num_blocks] and variable_block_sizes [num_blocks]: int32, contiguous. Returns {out}.
std::vector<torch::Tensor> block_sparse_sm121_blk128_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                                         torch::Tensor q2k_idx, torch::Tensor q2k_num,
                                                         torch::Tensor variable_block_sizes) {
  const at::Device device = q.device();
  check_tensor(q, "q", at::kBFloat16, device);
  check_tensor(k, "k", at::kBFloat16, device);
  check_tensor(v, "v", at::kBFloat16, device);
  TORCH_CHECK(q.dim() == 4, "q must be [batch, heads, seq, 128], got ", q.sizes());
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "q, k and v must have one shape, got ", q.sizes(),
              ", ", k.sizes(), " and ", v.sizes());
  const int64_t batch = q.size(0), heads = q.size(1), seq = q.size(2);
  TORCH_CHECK(q.size(3) == kHeadDim, "the head dimension must be ", kHeadDim, ", got ", q.size(3));

  check_tensor(q2k_idx, "q2k_idx", at::kInt, device);
  check_tensor(q2k_num, "q2k_num", at::kInt, device);
  check_tensor(variable_block_sizes, "variable_block_sizes", at::kInt, device);
  TORCH_CHECK(variable_block_sizes.dim() == 1, "variable_block_sizes must be [num_blocks], got ",
              variable_block_sizes.sizes());
  const int64_t blocks = variable_block_sizes.size(0);
  TORCH_CHECK(seq == blocks * kBlock, "seqlen ", seq, " must equal num_blocks (", blocks, ") * ", kBlock,
              "; FastVideo pads the sequence up to whole blocks");
  TORCH_CHECK(q2k_idx.dim() == 4 && q2k_idx.size(0) == batch && q2k_idx.size(1) == heads && q2k_idx.size(2) == blocks,
              "q2k_idx must be [", batch, ", ", heads, ", ", blocks, ", max_kv], got ", q2k_idx.sizes());
  TORCH_CHECK(q2k_num.dim() == 3 && q2k_num.size(0) == batch && q2k_num.size(1) == heads && q2k_num.size(2) == blocks,
              "q2k_num must be [", batch, ", ", heads, ", ", blocks, "], got ", q2k_num.sizes());
  // The grid is (num_blocks, batch * heads).
  TORCH_CHECK(batch * heads <= 65535, "batch * heads must be at most 65535, got ", batch * heads);
  TORCH_CHECK(blocks <= std::numeric_limits<int>::max() && q2k_idx.size(3) <= std::numeric_limits<int>::max(),
              "too many blocks: ", blocks, " (max_kv ", q2k_idx.size(3), ")");

  auto out = torch::empty_like(q);
  if (q.numel() == 0) return {out};

  const at::cuda::CUDAGuard guard(device);
  vsa_sm121::Vsa128Args args;
  args.q = q.data_ptr();
  args.k = k.data_ptr();
  args.v = v.data_ptr();
  args.out = out.data_ptr();
  args.q2k_idx = q2k_idx.data_ptr<int32_t>();
  args.q2k_num = q2k_num.data_ptr<int32_t>();
  args.block_sizes = variable_block_sizes.data_ptr<int32_t>();
  args.batch_heads = static_cast<int>(batch * heads);
  args.tiles = static_cast<int>(blocks);
  args.max_kv = static_cast<int>(q2k_idx.size(3));
  const cudaError_t err = vsa_sm121::launch(args, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(err == cudaSuccess, "block_sparse_sm121_blk128_fwd launch failed: ", cudaGetErrorString(err));
  return {out};
}
