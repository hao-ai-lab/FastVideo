// Mid-block self-attention with our own kernels, replacing ATen's
// scaled_dot_product_attention (fmha_cutlassF, 8.5 ms of f121x1's 954 ms).
//
// The reference computes, per latent frame, one head of softmax(q k^T/32) v over
// the H*W=3432 positions with C=1024 channels and no mask. ATen's path cannot use
// flash attention here (head dim 1024 > 256) and falls back to the xformers
// cutlass kernel, which reads and writes far more per frame than the products
// need: the decode's whole attention is 1.5 TFLOP (0.63 ms at 16-bit peak for
// f121x1) against 8.5 ms measured, ~13x off.
//
// Three stages, each at its own bound:
//   1. pack: q,k into contiguous [HW,C] per frame and v transposed into the
//      [C,PAD] weight layout the convolution GEMM wants (~21 MB/frame).
//   2. S = q k^T per frame (fp32 out) and O = p v per frame (bf16 out), both as
//      1x1x1 implicit GEMMs through the vendored CUTLASS SM100 convolution
//      collective (2-CTA tcgen05 MMA, TMA, fp32 accumulation) -- the same
//      machinery conv.cu uses, so no new GEMM path is introduced.
//   3. row softmax: S row -> exp(S/sqrt(C) - max)/sum in bf16, one CTA per row,
//      the row held in shared memory (one read of S, one write of P).
// P's rows are padded to a multiple of 64 columns and v's transpose is zero
// there, so the GEMM's tail contributes nothing.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "attn.h"

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

using namespace cute;
using ElementBF16 = cutlass::bfloat16_t;
using LayoutNDHWC = cutlass::layout::TensorNDHWC;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
constexpr auto Fprop = cutlass::conv::Operator::kFprop;

// One 1x1x1 implicit GEMM, y = x (per position) times w, fp32 accumulation and
// ElementD out. x: channels-last3d [1,C,D,1,1] (may be a strided view); w:
// [K,C,1,1,1] with explicit strides; y: packed channels-last3d [1,K,D,1,1].
template <class ElementD, int M, int N, int CL>
static void gemm1x1(const at::Tensor& x, const at::Tensor& w, const at::Tensor& y) {
  using Tile = Shape<Int<M>, Int<N>, Shape<_64>>;
  using Cluster = Shape<Int<CL>, _1, _1>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
      float, float, ElementD, LayoutNDHWC, 8, ElementD, LayoutNDHWC, 8,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop = typename cutlass::conv::collective::CollectiveBuilder<
      ArchTag, OperatorClass, Fprop, ElementBF16, LayoutNDHWC, 8, ElementBF16, LayoutNDHWC, 8, float,
      Tile, Cluster,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto>::CollectiveOp;
  using Problem = cutlass::conv::ConvProblemShape<Fprop, 3>;
  using Kernel = cutlass::conv::kernel::ConvUniversal<Problem, Mainloop, Epilogue>;
  using Conv = cutlass::conv::device::ConvUniversalAdapter<Kernel>;
  int C = x.size(1), D = x.size(2);
  int K = w.size(0);
  TORCH_CHECK(y.size(1) == K && y.size(2) == D, "attn gemm: output shape mismatch");
  Problem problem(cutlass::conv::Mode::kCrossCorrelation,
                  {1, D, 1, 1, C}, {x.stride(0), x.stride(2), x.stride(3), x.stride(4), x.stride(1)},
                  {K, 1, 1, 1, C}, {w.stride(0), w.stride(2), w.stride(3), w.stride(4), w.stride(1)},
                  {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 1);
  typename Kernel::StrideC sc{};
  typename Kernel::StrideD sd{};
  cute::for_each(cute::make_seq<cute::rank<0>(sc)>{}, [&](auto i) { cute::get<0, i>(sc) = problem.stride_C[Problem::RankT - 2 - i]; });
  cute::for_each(cute::make_seq<cute::rank<0>(sd)>{}, [&](auto i) { cute::get<0, i>(sd) = problem.stride_C[Problem::RankT - 2 - i]; });
  typename Conv::Arguments args{problem, {(const ElementBF16*)x.data_ptr(), (const ElementBF16*)w.data_ptr()},
                                {{1.f, 0.f}, (const ElementD*)y.data_ptr(), sc, (ElementD*)y.data_ptr(), sd}};
  Conv conv;
  TORCH_CHECK(conv.can_implement(args) == cutlass::Status::kSuccess,
              "attn gemm: CUTLASS cannot implement D=", D, " N=", K, " C=", C);
  size_t size = Conv::get_workspace_size(args);
  auto workspace = at::empty({(int64_t)size}, x.options().dtype(at::kByte));
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(conv.initialize(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess,
              "attn gemm: initialization failed");
  TORCH_CHECK(conv.run(stream) == cutlass::Status::kSuccess, "attn gemm: run failed");
}

// Channels-last-3d views with explicit strides: a size-1 dim built by unsqueeze
// carries stride 1, which the im2col TMA descriptor rejects.
static at::Tensor nhwc_in(const at::Tensor& m, int64_t C, int64_t D) {  // [D,C] -> [1,C,D,1,1]
  return m.as_strided({1, C, D, 1, 1}, {C * D, 1, C, C, C});
}
static at::Tensor nhwc_w(const at::Tensor& m, int64_t K, int64_t C) {  // [K,C] -> [K,C,1,1,1]
  return m.as_strided({K, C, 1, 1, 1}, {C, 1, C, C, C});
}
static at::Tensor nhwc_out(const at::Tensor& m, int64_t K, int64_t D) {  // [D,K] -> [1,K,D,1,1]
  return m.as_strided({1, K, D, 1, 1}, {K * D, 1, K, K, K});
}

// v [T,1,HW,3C] at offset 2C -> vt [T,C,PAD] (transposed, [T,C,PAD] zeroed first).
static __global__ void transpose_v_kernel(const __nv_bfloat16* __restrict__ qkv, int T, int HW, int C, int PAD,
                                   __nv_bfloat16* __restrict__ vt) {
  __shared__ __nv_bfloat16 tile[32][33];
  int i0 = blockIdx.x * 32, c0 = blockIdx.y * 32, t = blockIdx.z;
  {
    int i = i0 + threadIdx.y, c = c0 + threadIdx.x;
    if (i < HW && c < C) tile[threadIdx.y][threadIdx.x] = qkv[((size_t)t * HW + i) * (3 * C) + 2 * C + c];
  }
  __syncthreads();
  {
    int i = i0 + threadIdx.x, c = c0 + threadIdx.y;
    if (i < HW && c < C) vt[((size_t)t * C + c) * PAD + i] = tile[threadIdx.x][threadIdx.y];
  }
}

// One CTA per row: p = exp(s*scale - max)/sum in bf16, then zero the padded tail.
static __global__ void row_softmax_kernel(const float* __restrict__ s, int cols, int PAD, float scale,
                                   __nv_bfloat16* __restrict__ p) {
  extern __shared__ float row[];
  const int n4 = cols / 4;
  const float4* src = (const float4*)(s + (size_t)blockIdx.x * cols);
  __nv_bfloat16* dst = p + (size_t)blockIdx.x * PAD;
  float4* r4 = (float4*)row;
  for (int j = threadIdx.x; j < n4; j += blockDim.x) {
    float4 v = src[j];
    v.x *= scale;
    v.y *= scale;
    v.z *= scale;
    v.w *= scale;
    r4[j] = v;
  }
  __syncthreads();
  float m = -INFINITY;
  for (int j = threadIdx.x; j < n4; j += blockDim.x) {
    float4 v = r4[j];
    m = fmaxf(m, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
  }
  for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
  __shared__ float red[32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    m = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    if (threadIdx.x == 0) red[0] = m;
  }
  __syncthreads();
  m = red[0];
  __syncthreads();
  float sum = 0.f;
  for (int j = threadIdx.x; j < n4; j += blockDim.x) {
    float4 v = r4[j];
    v.x = __expf(v.x - m);
    v.y = __expf(v.y - m);
    v.z = __expf(v.z - m);
    v.w = __expf(v.w - m);
    r4[j] = v;
    sum += (v.x + v.y) + (v.z + v.w);
  }
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if (threadIdx.x == 0) red[0] = sum;
  }
  __syncthreads();
  const float inv = 1.f / red[0];
  const int npair = cols / 2;
  for (int j = threadIdx.x; j < npair; j += blockDim.x)
    *(__nv_bfloat162*)(dst + 2 * j) = __floats2bfloat162_rn(row[2 * j] * inv, row[2 * j + 1] * inv);
  for (int j = cols + threadIdx.x; j < PAD; j += blockDim.x) dst[j] = __float2bfloat16(0.f);
}

at::Tensor attn_forward(const at::Tensor& qkv) {
  TORCH_CHECK(qkv.dim() == 4 && qkv.size(1) == 1 && qkv.size(3) % 3 == 0, "attn: expected [T,1,HW,3C]");
  int T = qkv.size(0), HW = qkv.size(2), C = qkv.size(3) / 3;
  TORCH_CHECK(C % 64 == 0, "attn: channels must be a multiple of 64");
  int PAD = (HW + 63) / 64 * 64;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto bf = qkv.options();
  auto vt = at::zeros({T, C, PAD}, bf);
  auto s = at::empty({T, HW, HW}, bf.dtype(at::kFloat));
  auto p = at::empty({T, HW, PAD}, bf);
  auto o = at::empty({T, 1, HW, C}, bf);
  const __nv_bfloat16* qkv_p = (const __nv_bfloat16*)qkv.data_ptr();

  transpose_v_kernel<<<dim3((HW + 31) / 32, (C + 31) / 32, T), dim3(32, 32), 0, stream>>>(
      qkv_p, T, HW, C, PAD, (__nv_bfloat16*)vt.data_ptr());

  // S = q k^T : M = positions, N = keys, K = channels. q and k are read in place:
  // their channel stride is 1 and their position stride 3C, which TMA accepts.
  for (int t = 0; t < T; ++t) {
    auto qv = qkv.select(0, t).select(0, 0);  // [HW, 3C]
    auto x = qv.narrow(1, 0, C).as_strided({1, C, HW, 1, 1}, {(int64_t)3 * C * HW, 1, 3 * C, 3 * C, 3 * C});
    auto w = qv.narrow(1, C, C).as_strided({HW, C, 1, 1, 1}, {3 * C, 1, 3 * C, 3 * C, 3 * C});
    auto y = nhwc_out(s.select(0, t), HW, HW);
    gemm1x1<float, 256, 256, 2>(x, w, y);
  }

  float* s_p = (float*)s.data_ptr();
  float scale = 1.f / sqrtf((float)C);
  row_softmax_kernel<<<T * HW, 256, HW * sizeof(float), stream>>>(
      s_p, HW, PAD, scale, (__nv_bfloat16*)p.data_ptr());

  // O = p v : M = positions, N = channels, K = keys (padded).
  for (int t = 0; t < T; ++t) {
    auto x = nhwc_in(p.select(0, t), PAD, HW);
    auto w = nhwc_w(vt.select(0, t), C, PAD);
    auto y = nhwc_out(o.select(0, t).select(0, 0), C, HW);
    gemm1x1<ElementBF16, 256, 256, 2>(x, w, y);
  }
  return o;
}
