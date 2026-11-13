// CUTLASS 4.2 Blackwell implicit convolution: 2-CTA 256x256x64 tiles,
// TMA im2col loads, tcgen05 MMA, fp32 accumulation, bf16 epilogue.
// Every included CUTLASS/CuTe dependency is vendored beside this source.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
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
using ElementBF16=cutlass::bfloat16_t;
using LayoutNDHWC=cutlass::layout::TensorNDHWC;
using ArchTag=cutlass::arch::Sm100;
using OperatorClass=cutlass::arch::OpClassTensorOp;
constexpr auto Fprop=cutlass::conv::Operator::kFprop;

template<int M,int N,int CL>
at::Tensor launch(const at::Tensor& x,const at::Tensor& w,int ph,int pw,int phu=-1,int pwu=-1) {
  if(phu<0)phu=ph;if(pwu<0)pwu=pw;
  using Tile=Shape<Int<M>,Int<N>,Shape<_64>>;
  using Cluster=Shape<Int<CL>,_1,_1>;
  using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,OperatorClass,Tile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,
    float,float,ElementBF16,LayoutNDHWC,8,ElementBF16,LayoutNDHWC,8,cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop=typename cutlass::conv::collective::CollectiveBuilder<
    ArchTag,OperatorClass,Fprop,ElementBF16,LayoutNDHWC,8,ElementBF16,LayoutNDHWC,8,float,Tile,Cluster,
    cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::conv::collective::KernelScheduleAuto>::CollectiveOp;
  using Problem=cutlass::conv::ConvProblemShape<Fprop,3>;
  using Kernel=cutlass::conv::kernel::ConvUniversal<Problem,Mainloop,Epilogue>;
  using Conv=cutlass::conv::device::ConvUniversalAdapter<Kernel>;
  int C=x.size(1),D=x.size(2),H=x.size(3),W=x.size(4);
  int K=w.size(0),KT=w.size(2),KH=w.size(3),KW=w.size(4);
  auto y=at::empty({1,K,D-KT+1,H+ph+phu-KH+1,W+pw+pwu-KW+1},x.options().memory_format(at::MemoryFormat::ChannelsLast3d));
  Problem problem(cutlass::conv::Mode::kCrossCorrelation,
                  {1,D,H,W,C},{x.stride(0),x.stride(2),x.stride(3),x.stride(4),x.stride(1)},
                  {K,KT,KH,KW,C},{w.stride(0),w.stride(2),w.stride(3),w.stride(4),w.stride(1)},
                  {0,ph,pw},{0,phu,pwu},{1,1,1},{1,1,1},1);
  typename Kernel::StrideC sc{};
  typename Kernel::StrideD sd{};
  cute::for_each(cute::make_seq<cute::rank<0>(sc)>{},[&](auto i){cute::get<0,i>(sc)=problem.stride_C[Problem::RankT-2-i];});
  cute::for_each(cute::make_seq<cute::rank<0>(sd)>{},[&](auto i){cute::get<0,i>(sd)=problem.stride_C[Problem::RankT-2-i];});
  typename Conv::Arguments args{problem,{(const ElementBF16*)x.data_ptr(),(const ElementBF16*)w.data_ptr()},
    {{1.f,0.f},(const ElementBF16*)y.data_ptr(),sc,(ElementBF16*)y.data_ptr(),sd}};
  Conv conv;
  TORCH_CHECK(conv.can_implement(args)==cutlass::Status::kSuccess,"CUTLASS cannot implement shape");
  size_t size=Conv::get_workspace_size(args);
  auto workspace=at::empty({(int64_t)size},x.options().dtype(at::kByte));
  auto stream=at::cuda::getCurrentCUDAStream();
  auto status=conv.initialize(args,workspace.data_ptr(),stream);
  TORCH_CHECK(status==cutlass::Status::kSuccess,"CUTLASS initialization failed ",int(status));
  status=conv.run(stream);
  TORCH_CHECK(status==cutlass::Status::kSuccess,"CUTLASS convolution failed ",int(status));
  return y;
}
at::Tensor custom_conv(const at::Tensor& x,const at::Tensor& w,int64_t ph,int64_t pw,int64_t variant) {
  return launch<256,256,2>(x,w,ph,pw);
}

// The output head's 3x1x1 convolution: N=128 because the head has 112 output channels, and a
// 256-wide tile would spend 144 of its 256 columns on nothing. Measured at one clock (1951-2022
// MHz, both arms in J1044 and J1343): N=128 is 0.83 PFLOP/s = 34% of peak, N=256 is 0.69 = 28%.
// The time is what decides it and it is not close: 6.30 ms against 15.2 ms.
at::Tensor conv_n128(const at::Tensor& x,const at::Tensor& w) {
  return launch<256,128,2>(x,w,0,0);
}

at::Tensor phase_conv(const at::Tensor& x,const at::Tensor& w,int a,int b,bool halo) {
  at::Tensor v=halo?x.narrow(3,a,x.size(3)-1):x;
  return launch<256,256,2>(v,w,halo?0:1-a,1-b,halo?0:a,b);
}
