/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Adapted from ThunderKittens' NVLink all-to-all kernel:
 * https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/parallel/all_to_all/all_to_all.cu
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Torch host bindings for the fused-transpose Ulysses NVLink-P2P all-to-all.
// Kernel logic lives in include/comm/ulysses_all_to_all.cuh (vendored verbatim).
//
// Translated from the TVM-FFI bindings in flashinfer-ai/flashinfer
// @ 8a94642d83cba0939035868fb6c309b4474a13d6, csrc/ulysses_all_to_all.cu.
// FlashInfer uses TVM FFI so one .so serves torch, JAX and TVM; fastvideo-kernel
// is a pybind11/libtorch extension throughout, so the wrapper is rewritten while
// the validation, dispatch and launch logic are kept as they were upstream:
//
//   TVM_FFI_ICHECK(c) << "m"          ->  TORCH_CHECK(c, "m")
//   TensorView                        ->  torch::Tensor
//   tvm::ffi::Array<fptr_t>           ->  std::vector<int64_t>
//   ffi::CUDADeviceGuard              ->  at::cuda::CUDAGuard
//   get_stream(inp.device())          ->  at::cuda::getCurrentCUDAStream()
//   encode_dlpack_dtype(...)          ->  at::ScalarType
//   TVM_FFI_DLL_EXPORT_TYPED_FUNC     ->  m.def(...) in common_extension.cpp

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "comm/ulysses_all_to_all.cuh"

// Fake pointer type, matching the fptr_t used by the vLLM custom all-reduce
// bindings: an opaque host-side handle passed across the Python boundary as an
// int. It is NOT a device pointer.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

namespace fi = flashinfer::comm::ulysses;

namespace {

inline void check_operand(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

int64_t init_ulysses_a2a(const std::vector<int64_t>& out_ipc_ptrs,
                         const std::vector<int64_t>& signal_ipc_ptrs, int64_t rank,
                         int64_t world_size, bool full_nvlink) {
  TORCH_CHECK(world_size <= 8, "ulysses a2a world size > 8 is not supported");
  TORCH_CHECK(world_size == 2 || world_size == 4 || world_size == 6 || world_size == 8,
              "ulysses a2a only supports world size in (2, 4, 6, 8), got ", world_size);
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank passed in");
  TORCH_CHECK(static_cast<int64_t>(out_ipc_ptrs.size()) == world_size,
              "out_ipc_ptrs size must equal world_size");
  TORCH_CHECK(static_cast<int64_t>(signal_ipc_ptrs.size()) == world_size,
              "signal_ipc_ptrs size must equal world_size");

  fi::Signal* signals[8];
  void* out_bufs[8];
  for (int i = 0; i < world_size; i++) {
    signals[i] = reinterpret_cast<fi::Signal*>(signal_ipc_ptrs[i]);
    out_bufs[i] = reinterpret_cast<void*>(out_ipc_ptrs[i]);
  }

  // The multi_gpu_barrier counters must start at zero, and cudaMalloc does not
  // zero memory. Each rank owns signals[rank] (the others are IPC-mapped peer
  // buffers), so every rank zeroes its own signal here. The zeroing is enqueued
  // on the current stream and is asynchronous with respect to the host, so
  // callers MUST (1) synchronize this device after init returns and (2) issue a
  // process-group barrier before the first all-to-all.
  auto st = cudaMemsetAsync(signals[rank], 0, sizeof(fi::Signal),
                            at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(st == cudaSuccess, "failed to zero the ulysses a2a signal buffer: ",
              cudaGetErrorString(st));
  return (fptr_t) new fi::UlyssesA2A(signals, out_bufs, static_cast<int>(rank),
                                     static_cast<int>(world_size), full_nvlink);
}

void dispose_ulysses_a2a(int64_t _fa) { delete reinterpret_cast<fi::UlyssesA2A*>(_fa); }

// Size of the IPC-shared signal buffer each rank must allocate.
int64_t ulysses_signal_size() { return static_cast<int64_t>(sizeof(fi::Signal)); }

// Fused-transpose Ulysses all-to-all.
//   mode == 0: inp [B, S_local, H, D]        -> out [B, S_global, H_local, D]
//   mode == 1: inp [B, S_global, H_local, D] -> out [B, S_local, H, D]
// where H is the *global* head count and H_local = H / world_size.
void ulysses_a2a(int64_t _fa, torch::Tensor inp, torch::Tensor out, int64_t B, int64_t S_local,
                 int64_t H, int64_t D, int64_t mode) {
  auto fa = reinterpret_cast<fi::UlyssesA2A*>(_fa);
  TORCH_CHECK(fa != nullptr, "fa must be a handle returned by init_ulysses_a2a");

  const at::cuda::CUDAGuard device_guard(inp.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  check_operand(inp, "inp");
  check_operand(out, "out");
  TORCH_CHECK(inp.device() == out.device(), "inp and out must be on the same CUDA device");
  TORCH_CHECK(inp.scalar_type() == out.scalar_type(), "inp and out must share a dtype");
  TORCH_CHECK(inp.numel() == out.numel(), "inp and out must have equal element counts");
  TORCH_CHECK(mode == 0 || mode == 1, "ulysses_a2a mode must be 0 or 1");
  const int W = fa->world_size_;
  TORCH_CHECK(H % W == 0, "global head count must be divisible by world size");
  const int H_local = static_cast<int>(H / W);

  // Full 4-D shape validation for both operands.
  //   mode 0: inp [B, S_local, H, D]         -> out [B, W*S_local, H_local, D]
  //   mode 1: inp [B, W*S_local, H_local, D] -> out [B, S_local, H, D]
  TORCH_CHECK(inp.dim() == 4, "inp must be 4-D");
  TORCH_CHECK(out.dim() == 4, "out must be 4-D");
  const torch::Tensor& local_op = (mode == 0) ? inp : out;   // [B, S_local, H, D]
  const torch::Tensor& global_op = (mode == 0) ? out : inp;  // [B, S_global, H_local, D]
  TORCH_CHECK(local_op.size(0) == B && local_op.size(1) == S_local && local_op.size(2) == H &&
                  local_op.size(3) == D,
              "the [B, S_local, H, D] operand of mode ", mode, " has shape (", local_op.size(0),
              ", ", local_op.size(1), ", ", local_op.size(2), ", ", local_op.size(3),
              "), expected (", B, ", ", S_local, ", ", H, ", ", D, ")");
  TORCH_CHECK(global_op.size(0) == B && global_op.size(1) == W * S_local &&
                  global_op.size(2) == H_local && global_op.size(3) == D,
              "the [B, S_global, H_local, D] operand of mode ", mode, " has shape (",
              global_op.size(0), ", ", global_op.size(1), ", ", global_op.size(2), ", ",
              global_op.size(3), "), expected (", B, ", ", W * S_local, ", ", H_local, ", ", D,
              ")");

  const int64_t num_rows = B * static_cast<int64_t>(W) * S_local;
  const int blocks =
      static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(fi::kMaxBlocks, num_rows)));
  const int threads = fi::kUlyssesThreads;
  const size_t out_bytes = out.numel() * out.element_size();

#define LAUNCH_ULYSSES_A2A(T, NG, MODE)                                                            \
  fi::ulysses_a2a_kernel<T, NG, MODE><<<blocks, threads, 0, stream>>>(                             \
      reinterpret_cast<const T*>(inp.data_ptr()), fa->out_ptrs_, fa->sg_, fa->self_sg_, fa->rank_, \
      static_cast<int>(B), static_cast<int>(S_local), H_local, static_cast<int>(D))

#define DISPATCH_NGPUS(T, MODE)                                                      \
  switch (W) {                                                                       \
    case 2:                                                                          \
      LAUNCH_ULYSSES_A2A(T, 2, MODE);                                                \
      break;                                                                         \
    case 4:                                                                          \
      LAUNCH_ULYSSES_A2A(T, 4, MODE);                                                \
      break;                                                                         \
    case 6:                                                                          \
      LAUNCH_ULYSSES_A2A(T, 6, MODE);                                                \
      break;                                                                         \
    case 8:                                                                          \
      LAUNCH_ULYSSES_A2A(T, 8, MODE);                                                \
      break;                                                                         \
    default:                                                                         \
      TORCH_CHECK(false, "ulysses_a2a only supports world size in (2,4,6,8)");       \
  }

#define DISPATCH_DTYPE(MODE)                                                                 \
  switch (out.scalar_type()) {                                                               \
    case at::ScalarType::Float: {                                                            \
      DISPATCH_NGPUS(float, MODE);                                                           \
      break;                                                                                 \
    }                                                                                        \
    case at::ScalarType::Half: {                                                             \
      DISPATCH_NGPUS(half, MODE);                                                            \
      break;                                                                                 \
    }                                                                                        \
    case at::ScalarType::BFloat16: {                                                         \
      DISPATCH_NGPUS(nv_bfloat16, MODE);                                                     \
      break;                                                                                 \
    }                                                                                        \
    default:                                                                                 \
      TORCH_CHECK(false, "ulysses_a2a only supports float32, float16 and bfloat16, got ",    \
                  out.scalar_type());                                                        \
  }

  if (mode == 0) {
    DISPATCH_DTYPE(0);
  } else {
    DISPATCH_DTYPE(1);
  }

#undef DISPATCH_DTYPE
#undef DISPATCH_NGPUS
#undef LAUNCH_ULYSSES_A2A

  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "ulysses_a2a kernel launch failed");
  // Copy this rank's completed result out of the staging buffer.
  auto status = cudaMemcpyAsync(out.data_ptr(), fa->local_out_buf_, out_bytes,
                                cudaMemcpyDeviceToDevice, stream);
  TORCH_CHECK(status == cudaSuccess, "ulysses_a2a copy-out failed: ", cudaGetErrorString(status));
}

void register_ulysses_a2a(pybind11::module_& m) {
  m.def("init_ulysses_a2a", &init_ulysses_a2a, "initialize the ulysses a2a IPC context");
  m.def("dispose_ulysses_a2a", &dispose_ulysses_a2a, "release a ulysses a2a handle");
  m.def("ulysses_signal_size", &ulysses_signal_size, "bytes per IPC signal buffer");
  m.def("ulysses_a2a", &ulysses_a2a, "fused-transpose Ulysses all-to-all over NVLink P2P");
}
