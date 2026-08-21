// block_sparse_sm100a.cu -- torch binding for the sm_100a VSA block-sparse FMHA forward.
//
// Forward only: returns (out, lse) so FastVideo's existing Triton backward keeps working
// unchanged. lse is exactly the M tensor triton_block_sparse_attn_forward writes --
// max(qk * qk_scale) + log2(l), [B, H, S] fp32 -- which is what lets
// block_sparse_attn_backward_triton run against our forward untouched.
//
// The build is fixed at compile time by two flags, so one extension carries one configuration:
//   VSA_BLK128  false -> 64-token sparse blocks, true -> 128-token
//   VSA_BHSD    false -> [B, S, H, D], true -> [B, H, S, D]
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "block_sparse_launch_sm100a.cuh"

namespace {

void check_qkv(const torch::Tensor& t, const char* name, int64_t B, int64_t H, int64_t S,
               int64_t D) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bfloat16, got ", t.scalar_type());
  TORCH_CHECK(t.dim() == 4, name, " must be 4-D, got ", t.dim(), " dims");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  if (VSA_BHSD) {
    TORCH_CHECK(t.size(0) == B && t.size(1) == H && t.size(2) == S && t.size(3) == D, name,
                " has shape ", t.sizes(), ", expected [", B, ",", H, ",", S, ",", D, "]");
  } else {
    TORCH_CHECK(t.size(0) == B && t.size(1) == S && t.size(2) == H && t.size(3) == D, name,
                " has shape ", t.sizes(), ", expected [", B, ",", S, ",", H, ",", D, "]");
  }
}

void check_index(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kInt, name, " must be int32, got ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

// The exported symbol carries the block size: block_sparse_sm100a_fwd is the 64-token build,
// block_sparse_sm100a_blk128_fwd the 128-token one (block_sparse_blk128_sm100a.cu re-includes
// this file with VSA_BLK128 set). The python backend picks by the metadata's block size.
#if VSA_BLK128
#define BLOCK_SPARSE_SM100A_FWD block_sparse_sm100a_blk128_fwd
#else
#define BLOCK_SPARSE_SM100A_FWD block_sparse_sm100a_fwd
#endif

// Returns {out} or {out, lse}. Layout of out matches the inputs.
std::vector<torch::Tensor> BLOCK_SPARSE_SM100A_FWD(torch::Tensor q, torch::Tensor k,
                                                       torch::Tensor v,
                                                       c10::optional<torch::Tensor> v_t,
                                                       torch::Tensor q2k_idx,
                                                       torch::Tensor q2k_num,
                                                       torch::Tensor variable_block_sizes,
                                                       double sm_scale, bool need_lse) {
  const at::cuda::OptionalCUDAGuard guard(device_of(q));

  const int64_t B = q.size(0);
  const int64_t H = VSA_BHSD ? q.size(1) : q.size(2);
  const int64_t S = VSA_BHSD ? q.size(2) : q.size(1);
  const int64_t D = q.size(3);

  check_qkv(q, "q", B, H, S, D);
  check_qkv(k, "k", B, H, S, D);
  check_qkv(v, "v", B, H, S, D);
  check_index(q2k_idx, "q2k_idx");
  check_index(q2k_num, "q2k_num");
  check_index(variable_block_sizes, "variable_block_sizes");

  const int64_t num_blocks = variable_block_sizes.numel();
  const int64_t max_kv = q2k_idx.size(-1);
  TORCH_CHECK(S == num_blocks * BLOCK, "seqlen ", S, " must equal num_blocks (", num_blocks,
              ") * ", BLOCK, "; FastVideo pads the sequence up to whole blocks");

  auto out = torch::empty_like(q);
  torch::Tensor lse;
  if (need_lse) lse = torch::empty({B, H, S}, q.options().dtype(torch::kFloat32));

  BlockSparseVsaArgs a{};
  a.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  a.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  a.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  a.v_t = v_t.has_value() ? reinterpret_cast<const __nv_bfloat16*>(v_t->data_ptr()) : nullptr;
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

  // Report an unsupported regime loudly rather than returning plausible-looking wrong values.
  TORCH_CHECK(block_sparse_supported(a) == cudaSuccess,
              "block_sparse_sm100a: unsupported configuration -- requires head_dim==",
              HEAD_DIM, ", an even num_blocks, seqlen == num_blocks*", BLOCK,
              ", and a variable_block_sizes tensor. Got head_dim=", D, " num_blocks=",
              num_blocks, " seqlen=", S);

  const cudaError_t err = launch_block_sparse_sm100a(a, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(err == cudaSuccess,
              "block_sparse_sm100a launch failed: ", cudaGetErrorString(err));

  if (need_lse) return {out, lse};
  return {out};
}
