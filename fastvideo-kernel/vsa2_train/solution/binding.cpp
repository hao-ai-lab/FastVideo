// binding.cpp -- the module's three entry points, destination passing (task.md's interface):
//
//   forward(q, k, v, gate, t, h, w, text, topk, out, selection, lse)
//   backward(q, k, v, gate, out, lse, selection, dout, t, h, w, text, dq, dk, dv, dgate)
//   run(q, k, v, gate, dout, t, h, w, text, topk, out, selection, lse, dq, dk, dv, dgate)
//
// Each checks its arguments, allocates its own scratch from the caching allocator and launches its kernels on the
// current stream (vsa2t.h). run only calls forward, then backward; backward reads nothing but its arguments.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <climits>

#include "vsa2t.h"

namespace {

void check_tensor(const at::Tensor& x, const char* name, at::ScalarType dtype, at::IntArrayRef shape) {
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == dtype, name, " must be ", dtype, ", got ", x.scalar_type());
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.sizes() == shape, name, " has shape ", x.sizes(), ", expected ", shape);
}

// The geometry from q's shape and the layout arguments, checked.
vsa2t::Geo geometry(const at::Tensor& q, int64_t t, int64_t h, int64_t w, int64_t text, int64_t topk) {
  TORCH_CHECK(q.dim() == 4 && q.size(3) == vsa2t::kDim, "q must be [batch, heads, S, 128], got ", q.sizes());
  TORCH_CHECK(t > 0 && h > 0 && w > 0 && t <= INT_MAX && h <= INT_MAX && w <= INT_MAX,
              "t, h and w must be positive ints, got ", t, ", ", h, ", ", w);
  TORCH_CHECK(text >= 0 && text <= INT_MAX, "text must be a non-negative int, got ", text);
  const int64_t heads = q.size(0) * q.size(1);
  TORCH_CHECK(heads > 0 && heads <= 65535, "batch * heads must be in [1, 65535], got ", heads);
  const int64_t nv = ((t + 3) / 4) * ((h + 3) / 4) * ((w + 7) / 8);
  const int64_t n = (text + vsa2t::kRows - 1) / vsa2t::kRows + nv;
  // The lists hold N^2 entries per head, indexed in 64 bits; N itself and N_v^2 stay within 32 bits.
  TORCH_CHECK(n <= 65535, "N = ", n, " blocks is over the solution's limit of 65535");
  const int topk_i = (int)std::max<int64_t>(0, std::min<int64_t>(topk, nv));
  const vsa2t::Geo g = vsa2t::make_geo((int)heads, (int)text, (int)t, (int)h, (int)w, topk_i);
  TORCH_CHECK(q.size(2) == g.S, "q holds ", q.size(2), " rows, expected N * 128 = ", g.S);
  return g;
}

void forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& gate, int64_t t,
             int64_t h, int64_t w, int64_t text, int64_t topk, at::Tensor out, at::Tensor selection, at::Tensor lse) {
  const at::cuda::CUDAGuard guard(q.device());
  const vsa2t::Geo g = geometry(q, t, h, w, text, topk);
  const auto B = q.size(0), H = q.size(1);
  for (auto* x : {&q, &k, &v, &gate}) check_tensor(*x, "q, k, v and gate", at::kBFloat16, {B, H, g.S, vsa2t::kDim});
  check_tensor(out, "out", at::kBFloat16, {B, H, g.S, vsa2t::kDim});
  check_tensor(selection, "selection", at::kFloat, {B, H, g.Nv, g.Nv});
  check_tensor(lse, "lse", at::kFloat, {B, H, g.S});
  at::Tensor scratch = at::empty({(int64_t)vsa2t::forward_scratch_bytes(g)}, q.options().dtype(at::kByte));
  vsa2t::forward(g, q.data_ptr(), k.data_ptr(), v.data_ptr(), gate.data_ptr(), out.data_ptr(),
                 selection.data_ptr<float>(), lse.data_ptr<float>(), scratch.data_ptr(),
                 at::cuda::getCurrentCUDAStream());
}

void backward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& gate,
              const at::Tensor& out, const at::Tensor& lse, const at::Tensor& selection, const at::Tensor& dout,
              int64_t t, int64_t h, int64_t w, int64_t text, at::Tensor dq, at::Tensor dk, at::Tensor dv,
              at::Tensor dgate) {
  const at::cuda::CUDAGuard guard(q.device());
  // The backward needs no topk: the selection holds it.
  const vsa2t::Geo g = geometry(q, t, h, w, text, 0);
  const auto B = q.size(0), H = q.size(1);
  for (auto* x : {&q, &k, &v, &gate, &out, &dout})
    check_tensor(*x, "q, k, v, gate, out and dout", at::kBFloat16, {B, H, g.S, vsa2t::kDim});
  check_tensor(lse, "lse", at::kFloat, {B, H, g.S});
  check_tensor(selection, "selection", at::kFloat, {B, H, g.Nv, g.Nv});
  for (auto* x : {&dq, &dk, &dv, &dgate}) check_tensor(*x, "dq, dk, dv and dgate", at::kBFloat16, {B, H, g.S, vsa2t::kDim});
  at::Tensor scratch = at::empty({(int64_t)vsa2t::backward_scratch_bytes(g)}, q.options().dtype(at::kByte));
  vsa2t::backward(g, q.data_ptr(), k.data_ptr(), v.data_ptr(), gate.data_ptr(), out.data_ptr(), lse.data_ptr<float>(),
                  selection.data_ptr<float>(), dout.data_ptr(), dq.data_ptr(), dk.data_ptr(), dv.data_ptr(),
                  dgate.data_ptr(), scratch.data_ptr(), at::cuda::getCurrentCUDAStream());
}

void run(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& gate, const at::Tensor& dout,
         int64_t t, int64_t h, int64_t w, int64_t text, int64_t topk, at::Tensor out, at::Tensor selection,
         at::Tensor lse, at::Tensor dq, at::Tensor dk, at::Tensor dv, at::Tensor dgate) {
  forward(q, k, v, gate, t, h, w, text, topk, out, selection, lse);
  backward(q, k, v, gate, out, lse, selection, dout, t, h, w, text, dq, dk, dv, dgate);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "VSA2 attention forward: writes out, selection and lse");
  m.def("backward", &backward, "VSA2 attention backward: writes dq, dk, dv and dgate");
  m.def("run", &run, "one training step: forward, then backward");
}
