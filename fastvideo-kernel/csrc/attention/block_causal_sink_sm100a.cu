// block_causal_sink_sm100a.cu -- torch binding for the sm_100a block-causal + sink +
// sliding-window FMHA forward.
//
// Forward only: returns (out, lse) so the existing Triton backward keeps working
// unchanged -- lse is exactly the tensor _fwd_kernel writes today.
//
// Nothing is copied or reordered here. Q/K/V/O only need head_dim contiguous; the outer
// strides are read off the tensors, so a permuted (non-contiguous) view costs nothing.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "block_causal_sink_launch_sm100a.cuh"

namespace {

void check_qkv(const torch::Tensor& t, const char* name, int64_t B, int64_t H, int64_t L, int64_t D) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bfloat16, got ", t.scalar_type());
  TORCH_CHECK(t.dim() == 4, name, " must be [B, H, L, D], got ", t.dim(), " dims");
  TORCH_CHECK(t.size(0) == B && t.size(1) == H && t.size(2) == L && t.size(3) == D,
              name, " has shape ", t.sizes(), ", expected [", B, ",", H, ",", L, ",", D, "]");
  // head_dim contiguous is the ONLY layout requirement -- it is what lets the TMA descriptor
  // index the caller's tensor directly instead of forcing a .contiguous() copy.
  TORCH_CHECK(t.stride(3) == 1, name, " must have a contiguous head_dim (stride(-1) == 1); "
                                      "permuted views are fine, .contiguous() is not required");
  // TMA requires 16-byte aligned strides; bf16 -> multiples of 8 elements.
  for (int d = 0; d < 3; ++d)
    TORCH_CHECK(t.stride(d) % 8 == 0, name, " stride(", d, ")=", t.stride(d),
                " must be a multiple of 8 elements (16 B) for TMA");
}

}  // namespace

// Returns {out, lse}. lse is [B*H_q, L] float32 -- FastVideo's backward input.
std::vector<torch::Tensor> block_causal_sink_sm100a_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    c10::optional<torch::Tensor> q_sink,
    int64_t tokens_per_block, int64_t sink_tokens, int64_t rolling_window_tokens,
    double sm_scale, bool need_lse) {
  const at::cuda::OptionalCUDAGuard guard(device_of(q));

  const int64_t B = q.size(0), Hq = q.size(1), L = q.size(2), D = q.size(3);
  const int64_t Hkv = k.size(1);
  check_qkv(q, "q", B, Hq, L, D);
  check_qkv(k, "k", B, Hkv, L, D);
  check_qkv(v, "v", B, Hkv, L, D);
  TORCH_CHECK(Hq % Hkv == 0, "num_q_heads (", Hq, ") must be divisible by num_kv_heads (", Hkv, ")");

  const bool has_delta = q_sink.has_value();
  if (has_delta) check_qkv(*q_sink, "q_sink", B, Hq, L, D);

  auto out = torch::empty_like(q);
  torch::Tensor lse;
  if (need_lse)
    lse = torch::empty({B * Hq, L}, q.options().dtype(torch::kFloat32));

  BlockCausalSinkArgs a;
  a.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  a.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  a.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  a.q_sink = has_delta ? reinterpret_cast<const __nv_bfloat16*>(q_sink->data_ptr()) : nullptr;
  a.o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.lse = need_lse ? lse.data_ptr<float>() : nullptr;
  a.batch = (int)B;
  a.seqlen = (int)L;
  a.num_q_heads = (int)Hq;
  a.num_kv_heads = (int)Hkv;
  a.head_dim = (int)D;
  a.tokens_per_block = (int)tokens_per_block;
  a.sink_tokens = (int)sink_tokens;
  a.rolling_window_tokens = (int)rolling_window_tokens;
  a.sm_scale = (float)sm_scale;
  a.has_delta = has_delta;

  // Report an unsupported regime loudly. Outside it the decode/masking are out of spec and the
  // kernel would return plausible-looking but wrong values.
  TORCH_CHECK(block_causal_sink_supported(a) == cudaSuccess,
              "block_causal_sink_sm100a: unsupported configuration -- requires head_dim==", HEAD_DIM,
              ", seqlen divisible by tokens_per_block (no partial last block), and a sink reaching "
              "at most one K_TILE past a block end. Got head_dim=", D, " seqlen=", L,
              " tokens_per_block=", tokens_per_block, " sink_tokens=", sink_tokens);

  const cudaError_t err = launch_block_causal_sink_sm100a(a, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(err == cudaSuccess, "block_causal_sink_sm100a launch failed: ", cudaGetErrorString(err));

  if (need_lse) return {out, lse};
  return {out};
}
