#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "block_sparse_launch_ptx_sm107a.cuh"

namespace {

void check_qkv(const torch::Tensor& t, const char* name, int64_t B, int64_t H, int64_t S, int64_t D) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bfloat16, got ", t.scalar_type());
  TORCH_CHECK(t.dim() == 4, name, " must be 4-D, got ", t.dim(), " dims");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.size(0) == B && t.size(1) == H && t.size(2) == S && t.size(3) == D, name, " has shape ", t.sizes(),
              ", expected [", B, ",", H, ",", S, ",", D, "]");
}

void check_index(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kInt, name, " must be int32, got ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}

std::vector<torch::Tensor> block_sparse_ptx_sm107a_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                                       torch::Tensor q2k_idx, torch::Tensor q2k_num,
                                                       torch::Tensor variable_block_sizes, double sm_scale,
                                                       bool need_lse, bool q_prescaled) {
  using namespace vsa_ptx_sm107a;
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 10 && prop->minor == 7, "block_sparse_ptx_sm107a: needs an sm_107 (Rubin) device, got sm_",
              prop->major, prop->minor);

  const int64_t B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
  check_qkv(q, "q", B, H, S, D);
  check_qkv(k, "k", B, H, S, D);
  check_qkv(v, "v", B, H, S, D);
  check_index(q2k_idx, "q2k_idx");
  check_index(q2k_num, "q2k_num");
  check_index(variable_block_sizes, "variable_block_sizes");

  const int64_t num_blocks = variable_block_sizes.numel();
  TORCH_CHECK(S == num_blocks * BLOCK, "seqlen ", S, " must equal num_blocks (", num_blocks, ") * ", BLOCK,
              "; FastVideo pads the sequence up to whole blocks");
  const int64_t max_kv = q2k_idx.dim() > 0 ? q2k_idx.size(-1) : 0;
  TORCH_CHECK(max_kv >= 1 && q2k_idx.numel() == B * H * num_blocks * max_kv,
              "q2k_idx must hold B*H*num_blocks rows of max_kv >= 1 entries, got shape ", q2k_idx.sizes());
  TORCH_CHECK(q2k_num.numel() == B * H * num_blocks, "q2k_num must have B*H*num_blocks entries");

  auto out = torch::empty_like(q);
  torch::Tensor lse;
  if (need_lse) lse = torch::empty({B, H, S}, q.options().dtype(torch::kFloat32));

  BlockSparseVsaArgs a{};
  a.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  a.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  a.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  a.o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.lse = need_lse ? lse.data_ptr<float>() : nullptr;
  a.q2k_idx = q2k_idx.data_ptr<int>();
  a.q2k_num = q2k_num.data_ptr<int>();
  a.variable_block_sizes = variable_block_sizes.data_ptr<int>();
  a.batch = (int)B;
  a.num_heads = (int)H;
  a.seqlen = (int)S;
  a.head_dim = (int)D;
  a.num_blocks = (int)num_blocks;
  a.max_kv = (int)max_kv;
  a.sm_scale = (float)sm_scale;
  a.q_prescaled = q_prescaled;

  TORCH_CHECK(block_sparse_supported(a) == cudaSuccess,
              "block_sparse_ptx_sm107a: unsupported configuration -- requires head_dim == ", HEAD_DIM,
              ", an even num_blocks >= 2, seqlen == num_blocks * ", BLOCK, " and max_kv >= 1. Got head_dim=", D,
              " num_blocks=", num_blocks, " seqlen=", S, " max_kv=", a.max_kv);

  const cudaError_t err = launch_block_sparse_ptx_sm107a(a, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(err == cudaSuccess, "block_sparse_ptx_sm107a launch failed: ", cudaGetErrorString(err),
              " (needs CUDA 13.4+ and a driver with Rubin's oversized shared memory mode)");

  if (need_lse) return {out, lse};
  return {out};
}
