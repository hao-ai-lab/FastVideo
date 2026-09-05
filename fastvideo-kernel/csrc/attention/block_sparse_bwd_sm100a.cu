// block_sparse_bwd_sm100a.cu -- torch binding for the sm_100a VSA block-sparse FMHA backward.
//
// Pairs with block_sparse_sm100a_fwd. Inputs are the forward's operands plus its output o and
// its lse; lse is the Triton "M format" tensor the forward returns -- [B, H, S] fp32,
// M = max(qk * sm_scale * log2e) + log2(l) -- and is consumed as-is. Sparsity arrives as
// FastVideo's k2q metadata (fastvideo_kernel.triton_kernels.index.invert_indices): for every
// (batch, head, kv block) the LOCAL q64 block ids that selected it, padded to max_q_blocks,
// plus a count; entries past the count are never read. Returns {dq, dk, dv} in bf16 with the
// inputs' layout and the Triton backward's scaling: dk and dq carry sm_scale, dv does not.
//
// The layout is fixed at compile time: VSA_BHSD true -> [B, H, S, 128] (FastVideo's build),
// false -> [B, S, H, 128] (repo native; the kernel addresses it as [B*S tokens, H, 128]).
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <vector>

#include "block_sparse_bwd_launch_sm100a.cuh"

namespace {

// Identity work-item order [0, 1, ..., items) for the shapes below ORDER_MIN_KV_BLOCKS, where the
// order kernel costs more than the length-binned order saves. Built once per device with
// torch::arange, grown on demand, read-only afterwards; the launch reads it on the current stream.
const int* identity_remap(int64_t items, const torch::Device& device) {
  static std::vector<torch::Tensor> cache(64);
  torch::Tensor& t = cache.at(device.index() < 0 ? 0 : (size_t)device.index());
  if (!t.defined() || t.numel() < items) {
    t = torch::arange(items, torch::dtype(torch::kInt32).device(device));
  }
  return t.data_ptr<int>();
}

void check_activation(const torch::Tensor& t, const char* name, int64_t B, int64_t H, int64_t S,
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

__nv_bfloat16* bf16_ptr(const torch::Tensor& t) {
  return reinterpret_cast<__nv_bfloat16*>(t.data_ptr());
}

}  // namespace

// Returns {dq, dk, dv}: bf16, each with the shape and layout of q, k, v respectively.
std::vector<torch::Tensor> block_sparse_sm100a_bwd(torch::Tensor grad_o, torch::Tensor q,
                                                   torch::Tensor k, torch::Tensor v,
                                                   torch::Tensor o, torch::Tensor lse,
                                                   torch::Tensor k2q_idx, torch::Tensor k2q_num,
                                                   torch::Tensor variable_block_sizes,
                                                   double sm_scale) {
  const c10::cuda::OptionalCUDAGuard guard(device_of(q));

  TORCH_CHECK(q.dim() == 4, "q must be 4-D, got ", q.dim(), " dims");
  const int64_t B = q.size(0);
  const int64_t H = VSA_BHSD ? q.size(1) : q.size(2);
  const int64_t S = VSA_BHSD ? q.size(2) : q.size(1);
  const int64_t D = q.size(3);

  check_activation(q, "q", B, H, S, D);
  check_activation(k, "k", B, H, S, D);
  check_activation(v, "v", B, H, S, D);
  check_activation(o, "o", B, H, S, D);
  check_activation(grad_o, "grad_o", B, H, S, D);

  TORCH_CHECK(lse.is_cuda(), "lse must be a CUDA tensor");
  TORCH_CHECK(lse.scalar_type() == at::kFloat, "lse must be float32, got ", lse.scalar_type());
  TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");
  TORCH_CHECK(lse.numel() == B * H * S, "lse must hold [B, H, S] = ", B * H * S,
              " values (Triton M format), got ", lse.numel());

  check_index(k2q_idx, "k2q_idx");
  check_index(k2q_num, "k2q_num");
  check_index(variable_block_sizes, "variable_block_sizes");

  const int64_t num_kv_blocks_per_seq = variable_block_sizes.numel();
  TORCH_CHECK(S == num_kv_blocks_per_seq * BLOCK, "seqlen ", S,
              " must equal num_kv_blocks_per_seq (", num_kv_blocks_per_seq, ") * ", BLOCK,
              "; FastVideo pads the sequence up to whole blocks");

  const int64_t num_items = B * H * num_kv_blocks_per_seq;
  TORCH_CHECK(k2q_idx.dim() == 4 || k2q_idx.dim() == 2,
              "k2q_idx must be [B, H, num_kv_blocks_per_seq, max_q_blocks] or "
              "[B*H*num_kv_blocks_per_seq, max_q_blocks], got shape ",
              k2q_idx.sizes());
  const int64_t max_q_blocks = k2q_idx.size(-1);
  const bool k2q_dims_ok = k2q_idx.dim() == 2 || (k2q_idx.size(0) == B && k2q_idx.size(1) == H &&
                                                  k2q_idx.size(2) == num_kv_blocks_per_seq);
  TORCH_CHECK(k2q_dims_ok && k2q_idx.numel() == num_items * max_q_blocks, "k2q_idx has shape ",
              k2q_idx.sizes(), ", expected [", B, ",", H, ",", num_kv_blocks_per_seq,
              ",max_q_blocks] or [", num_items, ",max_q_blocks]");
  TORCH_CHECK(k2q_num.numel() == num_items,
              "k2q_num must hold one count per (batch, head, kv "
              "block) = ",
              num_items, " values, got ", k2q_num.numel());

  auto dq = torch::empty_like(q);
  auto dk = torch::empty_like(k);
  auto dv = torch::empty_like(v);

  // Workspace: torch::empty is enough. The preprocess kernel zeroes dqaccum and fully writes
  // qt, dot and delta before the main kernel reads them; it also zeroes the dk/dv rows of kv
  // blocks that no q block selects, so the empty_like outputs above come back fully defined.
  const auto bytes = q.options().dtype(at::kByte);
  const int b = (int)B, h = (int)H, s = (int)S;
  auto dqaccum = torch::empty({(int64_t)block_sparse_bwd_dqaccum_bytes(b, h, s)}, bytes);
  auto qt      = torch::empty({(int64_t)block_sparse_bwd_transposed_bytes(b, h, s)}, bytes);
  auto dot     = torch::empty({(int64_t)block_sparse_bwd_transposed_bytes(b, h, s)}, bytes);
  auto delta   = torch::empty({(int64_t)block_sparse_bwd_delta_bytes(b, h, s)}, bytes);
  // Work-item order: from ORDER_MIN_KV_BLOCKS on, the launch computes the length-binned order
  // into this workspace; below, the cached identity array is passed as the explicit remap.
  torch::Tensor order;
  const bool device_order = num_kv_blocks_per_seq >= ORDER_MIN_KV_BLOCKS;
  if (device_order) {
    order = torch::empty({(int64_t)block_sparse_bwd_order_bytes(b, h, (int)num_kv_blocks_per_seq)},
                         bytes);
  }

  BlockSparseVsaBwdArgs a{};
  a.q                     = bf16_ptr(q);
  a.k                     = bf16_ptr(k);
  a.v                     = bf16_ptr(v);
  a.o                     = bf16_ptr(o);
  a.dout                  = bf16_ptr(grad_o);
  a.dq                    = bf16_ptr(dq);
  a.dk                    = bf16_ptr(dk);
  a.dv                    = bf16_ptr(dv);
  a.lse                   = lse.data_ptr<float>();
  a.k2q_idx               = k2q_idx.data_ptr<int>();
  a.k2q_num               = k2q_num.data_ptr<int>();
  a.variable_block_sizes  = variable_block_sizes.data_ptr<int>();
  a.workitem_remap        = device_order ? nullptr : identity_remap(num_items, q.device());
  a.order_workspace       = device_order ? reinterpret_cast<int*>(order.data_ptr()) : nullptr;
  a.dqaccum               = reinterpret_cast<dq_accum_t*>(dqaccum.data_ptr());
  a.qt                    = bf16_ptr(qt);
  a.dot                   = bf16_ptr(dot);
  a.delta                 = reinterpret_cast<float*>(delta.data_ptr());
  a.batch                 = b;
  a.num_heads             = h;
  a.seqlen                = s;
  a.head_dim              = (int)D;
  a.num_kv_blocks_per_seq = (int)num_kv_blocks_per_seq;
  a.max_q_blocks          = (int)max_q_blocks;
  a.sm_scale              = (float)sm_scale;

  // Report an unsupported regime loudly rather than returning plausible-looking wrong values.
  TORCH_CHECK(block_sparse_bwd_supported(a) == cudaSuccess,
              "block_sparse_sm100a_bwd: unsupported configuration -- requires head_dim==", HEAD_DIM,
              ", seqlen == num_kv_blocks_per_seq*", BLOCK,
              " with seqlen % 128 == 0, "
              "max_q_blocks >= 1 and a finite sm_scale. Got head_dim=",
              D, " num_kv_blocks_per_seq=", num_kv_blocks_per_seq, " seqlen=", S,
              " max_q_blocks=", max_q_blocks, " sm_scale=", sm_scale);

  const cudaError_t err = launch_block_sparse_bwd_sm100a(a, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(err == cudaSuccess,
              "block_sparse_sm100a_bwd launch failed: ", cudaGetErrorString(err));

  return {dq, dk, dv};
}
