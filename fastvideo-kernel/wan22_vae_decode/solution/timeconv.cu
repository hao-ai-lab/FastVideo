// w11: a CUTLASS implicit convolution whose epilogue stores into a caller-given destination with arbitrary
// (N,D,H,W) strides, and can accumulate on what the destination already holds (beta=1, source == destination).
// One user: the temporal upsampler, which writes the frame-interleaved video straight out (stride 2 in D).
// The resampler's phase tensors come from conv.cu's phase_conv, not from this file.
// Nothing else changes: same tile (256x256x64), same 2-CTA cluster, same TMA im2col, same bf16 epilogue.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/convnd_problem_shape.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/conv/dispatch_policy.hpp"
#include "cutlass/conv/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/conv/device/conv_universal_adapter.hpp"
#include "cutlass/conv/kernel/conv_universal.hpp"
at::Tensor nhwc_empty(int64_t c, int64_t t, int64_t h, int64_t w, const at::Tensor& like);
using namespace cute;
using B = __nv_bfloat16;
using ElementBF16 = cutlass::bfloat16_t;
using LayoutNDHWC = cutlass::layout::TensorNDHWC;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
constexpr auto Fprop = cutlass::conv::Operator::kFprop;

// x: activation [1,C,D,H,W] channels-last; w: [K,C,KT,KH,KW] channels-last; y: the destination, with its own strides.
template <int M, int N, int CL>
static void launch_into(const at::Tensor& x, const at::Tensor& w, at::Tensor& y, int ph, int pw, float beta,
                        int phu = -1, int pwu = -1) {
  if (phu < 0) phu = ph;
  if (pwu < 0) pwu = pw;
  using Tile = Shape<Int<M>, Int<N>, Shape<_64>>;
  using Cluster = Shape<Int<CL>, _1, _1>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      ElementBF16, LayoutNDHWC, 8, ElementBF16, LayoutNDHWC, 8,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop = typename cutlass::conv::collective::CollectiveBuilder<
      ArchTag, OperatorClass, Fprop, ElementBF16, LayoutNDHWC, 8, ElementBF16, LayoutNDHWC, 8, float, Tile, Cluster,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto>::CollectiveOp;
  using Problem = cutlass::conv::ConvProblemShape<Fprop, 3>;
  using Kernel = cutlass::conv::kernel::ConvUniversal<Problem, Mainloop, Epilogue>;
  using Conv = cutlass::conv::device::ConvUniversalAdapter<Kernel>;
  int C = x.size(1), D = x.size(2), H = x.size(3), W = x.size(4);
  int K = w.size(0), KT = w.size(2), KH = w.size(3), KW = w.size(4);
  int OD = y.size(2), OH = y.size(3), OW = y.size(4);
  TORCH_CHECK(OD == D - KT + 1 && OH == H + ph + phu - KH + 1 && OW == W + pw + pwu - KW + 1,
              "launch_into: destination shape does not match the convolution");
  TORCH_CHECK(y.size(1) == K && y.stride(1) == 1, "launch_into: destination must be channel-innermost");
  Problem problem(cutlass::conv::Mode::kCrossCorrelation,
                  {1, D, H, W, C}, {x.stride(0), x.stride(2), x.stride(3), x.stride(4), x.stride(1)},
                  {K, KT, KH, KW, C}, {w.stride(0), w.stride(2), w.stride(3), w.stride(4), w.stride(1)},
                  {0, ph, pw}, {0, phu, pwu}, {1, 1, 1}, {1, 1, 1}, 1);
  // The output's own strides: index 0 N, 1 D, 2 H, 3 W, 4 C (the layout the C strides already use).
  problem.stride_C[0] = y.stride(0);
  problem.stride_C[1] = y.stride(2);
  problem.stride_C[2] = y.stride(3);
  problem.stride_C[3] = y.stride(4);
  problem.stride_C[4] = 1;
  typename Kernel::StrideC sc{};
  typename Kernel::StrideD sd{};
  cute::for_each(cute::make_seq<cute::rank<0>(sc)>{},
                 [&](auto i) { cute::get<0, i>(sc) = problem.stride_C[Problem::RankT - 2 - i]; });
  cute::for_each(cute::make_seq<cute::rank<0>(sd)>{},
                 [&](auto i) { cute::get<0, i>(sd) = problem.stride_C[Problem::RankT - 2 - i]; });
  const ElementBF16* dst = (const ElementBF16*)y.data_ptr();
  typename Conv::Arguments args{problem, {(const ElementBF16*)x.data_ptr(), (const ElementBF16*)w.data_ptr()},
                                {{1.f, beta}, dst, sc, (ElementBF16*)y.data_ptr(), sd}};
  Conv conv;
  TORCH_CHECK(conv.can_implement(args) == cutlass::Status::kSuccess, "CUTLASS cannot implement shape");
  size_t size = Conv::get_workspace_size(args);
  auto workspace = at::empty({(int64_t)size}, x.options().dtype(at::kByte));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto status = conv.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS initialization failed ", int(status));
  status = conv.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS convolution failed ", int(status));
}

// The temporal upsampler: y[t] = w[:,:,2] x[t-2] + w[:,:,1] x[t-1] + w[:,:,0] x[t] + bias, with frame 0 passed
// through unchanged when `first`, the two temporal halves of the 2C channels going to consecutive frames.
// p is the activation already padded with its two causal history frames ([1,C,nf+2,H,W], channels-last).
// INVARIANT: every element of y is written by head_bias_kernel, and the two parity arms below cover the
// non-head frames disjointly, each exactly once -- which is what makes beta=1 onto an at::empty safe.
static __global__ void head_bias_kernel(const B* head, const B* bias, B* y, int C, int T2, int64_t HW, int off) {
  int64_t n = (int64_t)T2 * HW * (C / 8);
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    int cg = (int)(i % (C / 8)); int64_t r = i / (C / 8);
    int64_t pos = r % HW; int t = (int)(r / HW);
    uint4 v;
    if (t < off) v = ((const uint4*)head)[pos * (C / 8) + cg];
    else v = ((const uint4*)(bias + ((t - off) & 1) * C))[cg];
    ((uint4*)y)[i] = v;
  }
}

at::Tensor time_upsample(const at::Tensor& p, const at::Tensor& head, const at::Tensor& w, const at::Tensor& bias,
                         int64_t T, int64_t nf, bool first) {
  const at::Tensor& ref = p.defined() ? p : head;
  int C = bias.size(0) / 2, H = ref.size(3), W = ref.size(4);
  int T2 = (int)(2 * T - (first ? 1 : 0));
  at::Tensor y = nhwc_empty(C, T2, H, W, ref);
  int off = first ? 1 : 0;
  head_bias_kernel<<<1024, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      first ? (const B*)head.data_ptr() : nullptr, (const B*)bias.data_ptr(), (B*)y.data_ptr(), C, T2,
      (int64_t)H * W, off);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (nf > 0) {
    int64_t sN = y.stride(0), sD = 2 * y.stride(2), sH = y.stride(3), sW = y.stride(4);
    for (int par = 0; par < 2; par++) {
      auto yv = y.as_strided({1, (int64_t)C, nf, (int64_t)H, (int64_t)W}, {sN, 1, sD, sH, sW},
                             (int64_t)(off + par) * y.stride(2));
      launch_into<256, 256, 2>(p, w.narrow(0, (int64_t)par * C, C), yv, 0, 0, 1.f);
    }
  }
  return y;
}
