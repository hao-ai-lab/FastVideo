/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Adapted from flashinfer-ai/flashinfer @ 8a94642d83cba0939035868fb6c309b4474a13d6
 * (PR #3820), csrc/ulysses_all_to_all.cu.
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

// Torch host bindings for the fused Ulysses all-to-all. Kernel in
// include/comm/ulysses_all_to_all.cuh.
//
// The per-group context is an ncclDevComm plus a registered symmetric window,
// both created here from the caller's ncclComm_t.

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <nccl.h>
#include <nccl_device.h>

#include "comm/ulysses_all_to_all.cuh"

namespace fi = fastvideo::comm::ulysses;

namespace {

// A symmetric window every rank can store into, plus the ncclDevComm the
// kernel opens barrier sessions on.
struct UlyssesContext {
  ncclComm_t comm = nullptr;
  ncclWindow_t win = nullptr;
  void* buf = nullptr;
  size_t nbytes = 0;
  ncclDevComm devComm{};
  bool dev_comm_created = false;
  int device = -1;
  int rank = 0;
  int world = 0;
};

#define NCCL_TRY(expr, what)                                                   \
  do {                                                                         \
    ncclResult_t _r = (expr);                                                  \
    TORCH_CHECK(_r == ncclSuccess, what " failed: rc=", static_cast<int>(_r)); \
  } while (0)

}  // namespace

// Allocate the local half of a context. This is intentionally separate from
// registration so Python can vote after local allocation: if one rank is OOM,
// no peer enters a collective window registration alone.
int64_t allocate_ulysses_a2a(int64_t nbytes, int64_t rank, int64_t world_size,
                             int64_t device_index) {
  TORCH_CHECK(world_size == 2 || world_size == 4 || world_size == 6 || world_size == 8,
              "ulysses a2a only supports world size in (2, 4, 6, 8), got ", world_size);
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");
  TORCH_CHECK(nbytes > 0, "nbytes must be positive");
  TORCH_CHECK(device_index >= 0, "device index must be non-negative");

  const at::cuda::CUDAGuard device_guard(
      c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device_index)));
  auto ctx = std::make_unique<UlyssesContext>();
  ctx->nbytes = static_cast<size_t>(nbytes);
  ctx->device = static_cast<int>(device_index);
  ctx->rank = static_cast<int>(rank);
  ctx->world = static_cast<int>(world_size);

  // The window must come from NCCL's allocator (4096B aligned per
  // NCCL_WIN_REQUIRED_ALIGNMENT), which is why this is not a torch tensor.
  NCCL_TRY(ncclMemAlloc(&ctx->buf, ctx->nbytes), "ncclMemAlloc");

  return reinterpret_cast<int64_t>(ctx.release());
}

// Register the user window. Collective: every rank must call together.
void register_ulysses_a2a_window(int64_t handle, int64_t comm_ptr) {
  auto* ctx = reinterpret_cast<UlyssesContext*>(handle);
  TORCH_CHECK(ctx != nullptr, "handle must come from allocate_ulysses_a2a");
  TORCH_CHECK(ctx->buf != nullptr && ctx->win == nullptr && !ctx->dev_comm_created,
              "ulysses a2a context is not in the allocated state");

  const at::cuda::CUDAGuard device_guard(
      c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(ctx->device)));
  ctx->comm = reinterpret_cast<ncclComm_t>(comm_ptr);
  NCCL_TRY(ncclCommWindowRegister(ctx->comm, ctx->buf, ctx->nbytes, &ctx->win,
                                  NCCL_WIN_COLL_SYMMETRIC),
           "ncclCommWindowRegister");
}

// Create the device communicator only after Python has voted that every rank
// registered its window. This operation is collective as well.
void create_ulysses_a2a_dev_comm(int64_t handle) {
  auto* ctx = reinterpret_cast<UlyssesContext*>(handle);
  TORCH_CHECK(ctx != nullptr, "handle must come from allocate_ulysses_a2a");
  TORCH_CHECK(ctx->comm != nullptr && ctx->win != nullptr && !ctx->dev_comm_created,
              "ulysses a2a context is not in the window-registered state");

  const at::cuda::CUDAGuard device_guard(
      c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(ctx->device)));

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = fi::kMaxBlocks;
  NCCL_TRY(ncclDevCommCreate(ctx->comm, &reqs, &ctx->devComm), "ncclDevCommCreate");
  ctx->dev_comm_created = true;
}

static ncclResult_t first_error(ncclResult_t current, ncclResult_t next) {
  return current == ncclSuccess ? next : current;
}

void dispose_ulysses_a2a(int64_t handle) {
  auto* ctx = reinterpret_cast<UlyssesContext*>(handle);
  if (ctx == nullptr) return;

  const at::cuda::CUDAGuard device_guard(
      c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(ctx->device)));
  ncclResult_t result = ncclSuccess;
  if (ctx->comm != nullptr && ctx->dev_comm_created) {
    result = first_error(result, ncclDevCommDestroy(ctx->comm, &ctx->devComm));
    ctx->dev_comm_created = false;
  }
  if (ctx->comm != nullptr && ctx->win != nullptr) {
    result = first_error(result, ncclCommWindowDeregister(ctx->comm, ctx->win));
    ctx->win = nullptr;
  }
  if (ctx->buf != nullptr) {
    result = first_error(result, ncclMemFree(ctx->buf));
    ctx->buf = nullptr;
  }
  delete ctx;
  TORCH_CHECK(result == ncclSuccess, "ulysses a2a cleanup failed: rc=", static_cast<int>(result));
}

// Whether the whole group is load-store accessible. NCCL determined this at
// ncclCommInitRank.
bool ulysses_lsa_covers_group(int64_t comm_ptr, int64_t world_size) {
  auto comm = reinterpret_cast<ncclComm_t>(comm_ptr);
  ncclCommProperties properties = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_TRY(ncclCommQueryProperties(comm, &properties), "ncclCommQueryProperties");
  ncclTeam_t lsa = ncclTeamLsa(comm);
  return properties.deviceApiSupport && lsa.nRanks == static_cast<int>(world_size);
}

// Fused-transpose Ulysses all-to-all.
//   mode == 0: inp [B, S_local, H, D]        -> out [B, S_global, H_local, D]
//   mode == 1: inp [B, S_global, H_local, D] -> out [B, S_local, H, D]
// where H is the *global* head count and H_local = H / world_size.
void ulysses_a2a(int64_t handle, torch::Tensor inp, torch::Tensor out, int64_t B, int64_t S_local,
                 int64_t H, int64_t D, int64_t mode) {
  auto* ctx = reinterpret_cast<UlyssesContext*>(handle);
  TORCH_CHECK(ctx != nullptr, "handle must come from allocate_ulysses_a2a");

  const at::cuda::CUDAGuard device_guard(inp.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(inp.is_cuda() && out.is_cuda(), "inp and out must be CUDA tensors");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(), "inp and out must be contiguous");
  TORCH_CHECK(inp.device() == out.device(), "inp and out must be on the same device");
  TORCH_CHECK(inp.scalar_type() == out.scalar_type(), "inp and out must share a dtype");
  TORCH_CHECK(inp.numel() == out.numel(), "inp and out must have equal element counts");
  TORCH_CHECK(inp.get_device() == ctx->device, "input is on CUDA device ", inp.get_device(),
              " but the Ulysses context belongs to device ", ctx->device);
  TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");
  TORCH_CHECK(inp.dim() == 4 && out.dim() == 4, "inp and out must be 4-D");

  const int W = ctx->world;
  TORCH_CHECK(H % W == 0, "global head count must be divisible by world size");
  const int H_local = static_cast<int>(H / W);

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

  const size_t out_bytes = out.numel() * out.element_size();
  TORCH_CHECK(out_bytes <= ctx->nbytes, "operand of ", out_bytes,
              " bytes exceeds the window capacity ", ctx->nbytes);

  const int64_t num_rows = B * static_cast<int64_t>(W) * S_local;
  const int blocks =
      static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(fi::kMaxBlocks, num_rows)));
  const int threads = fi::kUlyssesThreads;

#define LAUNCH_ULYSSES_A2A(T, NG, MODE)                                             \
  fi::ulysses_a2a_kernel<T, NG, MODE><<<blocks, threads, 0, stream>>>(              \
      reinterpret_cast<const T*>(inp.data_ptr()), ctx->devComm, ctx->win, /*off=*/0, \
      ctx->rank, static_cast<int>(B), static_cast<int>(S_local), H_local, static_cast<int>(D))

#define DISPATCH_NGPUS(T, MODE)                                                \
  switch (W) {                                                                 \
    case 2:                                                                    \
      LAUNCH_ULYSSES_A2A(T, 2, MODE);                                          \
      break;                                                                   \
    case 4:                                                                    \
      LAUNCH_ULYSSES_A2A(T, 4, MODE);                                          \
      break;                                                                   \
    case 6:                                                                    \
      LAUNCH_ULYSSES_A2A(T, 6, MODE);                                          \
      break;                                                                   \
    case 8:                                                                    \
      LAUNCH_ULYSSES_A2A(T, 8, MODE);                                          \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false, "ulysses_a2a only supports world size in (2,4,6,8)"); \
  }

#define DISPATCH_DTYPE(MODE)                                                              \
  switch (out.scalar_type()) {                                                            \
    case at::ScalarType::Float: {                                                         \
      DISPATCH_NGPUS(float, MODE);                                                        \
      break;                                                                              \
    }                                                                                     \
    case at::ScalarType::Half: {                                                          \
      DISPATCH_NGPUS(half, MODE);                                                         \
      break;                                                                              \
    }                                                                                     \
    case at::ScalarType::BFloat16: {                                                      \
      DISPATCH_NGPUS(nv_bfloat16, MODE);                                                  \
      break;                                                                              \
    }                                                                                     \
    default:                                                                              \
      TORCH_CHECK(false, "ulysses_a2a only supports float32, float16 and bfloat16, got ", \
                  out.scalar_type());                                                     \
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
  // Copy this rank's completed result out of the window.
  auto status = cudaMemcpyAsync(out.data_ptr(), ctx->buf, out_bytes, cudaMemcpyDeviceToDevice,
                                stream);
  TORCH_CHECK(status == cudaSuccess, "ulysses_a2a copy-out failed: ", cudaGetErrorString(status));
}

void register_ulysses_a2a(pybind11::module_& m) {
  m.def("allocate_ulysses_a2a", &allocate_ulysses_a2a, "allocate a local ulysses a2a window");
  m.def("register_ulysses_a2a_window", &register_ulysses_a2a_window,
        "register the ulysses a2a window collectively");
  m.def("create_ulysses_a2a_dev_comm", &create_ulysses_a2a_dev_comm,
        "create the ulysses a2a device communicator collectively");
  m.def("dispose_ulysses_a2a", &dispose_ulysses_a2a, "release a ulysses a2a context");
  m.def("ulysses_lsa_covers_group", &ulysses_lsa_covers_group,
        "whether the whole group is load-store accessible");
  m.def("ulysses_a2a", &ulysses_a2a, "fused-transpose Ulysses all-to-all over NVLink");
}
