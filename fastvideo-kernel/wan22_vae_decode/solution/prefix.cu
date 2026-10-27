// Prefix ownership transpose: no arithmetic, only bf16 NDHWC packets.
#include "prefix.h"
#include "kernels.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {
// Blocks for destination bands are disjoint; each packet is [T,H_peer,W,C].
__global__ void pack_prefix_rows(const uint4* x,uint4* send,int T,int H,int WC8,int world) {
  const int peer=blockIdx.y;
  const int lo=H*peer/world, hi=H*(peer+1)/world;
  const int64_t n=(int64_t)T*(hi-lo)*WC8;
  const int64_t offset=(int64_t)T*lo*WC8;
  for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(int64_t)gridDim.x*blockDim.x) {
    const int64_t t=i/((int64_t)(hi-lo)*WC8);
    const int64_t rowcol=i%((int64_t)(hi-lo)*WC8);
    send[offset+i]=x[(t*H+lo)*WC8+rowcol];
  }
}
}

at::Tensor prefix_temporal_to_rows(const at::Tensor& local,int64_t total_frames,
    int rank,int world,const c10::intrusive_ptr<c10d::ProcessGroup>& pg) {
  const int C=local.size(1),T=local.size(2),H=local.size(3),W=local.size(4);
  TORCH_CHECK(C%8==0 && local.is_contiguous(at::MemoryFormat::ChannelsLast3d),"Prefix needs aligned NDHWC");
  TORCH_CHECK(T==total_frames*(rank+1)/world-total_frames*rank/world,"Prefix ownership mismatch");
  auto send=at::empty({local.numel()},local.options());
  const int band_height=H*(rank+1)/world-H*rank/world;
  auto output=nhwc_empty(C,total_frames,band_height,W,local);
  // ProcessGroup splits count bf16 elements, not bytes. Receive into the
  // physical NDHWC order: the source ranks own chronological frame intervals.
  auto physical=output.permute({0,2,3,4,1});
  TORCH_CHECK(physical.is_contiguous(),"Prefix receive must be dense NDHWC");
  auto recv=physical.view({-1});
  if(T) {
    dim3 grid(256,world);
    pack_prefix_rows<<<grid,256,0,at::cuda::getCurrentCUDAStream()>>>(
        (const uint4*)local.data_ptr(),(uint4*)send.data_ptr(),T,H,W*(C/8),world);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  std::vector<int64_t> in_split(world),out_split(world);
  int64_t sent=0,received=0;
  for(int p=0;p<world;++p) {
    in_split[p]=(int64_t)T*(H*(p+1)/world-H*p/world)*W*C;
    out_split[p]=(total_frames*(p+1)/world-total_frames*p/world)*band_height*W*C;
    sent+=in_split[p]; received+=out_split[p];
  }
  TORCH_CHECK(sent==send.numel() && received==recv.numel(),"Prefix packet size mismatch");
  // WorkNCCL::wait orders the caller's stream after the collective; it need
  // not block the host. ProcessGroup's tensor/allocator stream tracking keeps
  // send/recv storage live through that dependency. Do not drop this wait
  // when tail kernels consume output on the caller's current stream.
  pg->alltoall_base(recv,send,out_split,in_split)->wait();
  return output;
}
