// The output head, epilogue half: conv_out's 3x3 spatial stencil, its bias and the clamp to
// [-1,1], from the 112-channel plane conv.cu's `conv_n128` (a 3x1x1 convolution) writes to the
// 12-channel plane write_video consumes.
//
// Why the convolution writes 108 channels and this pass sums 9 of them. conv_out is 256x27 MACs
// and its per-position INPUT WINDOW is 6912*2 B (the weight is 12x that and stays resident), i.e.
// 12 FLOP/byte against the machine's ~375, so any form that reads the 27 taps per position is
// operand-bound. Moving the nine spatial taps into the output channel dimension of a 3x1x1
// (temporal-only) convolution makes the operand 768*2 B for 108 outputs instead of 13.8 KB for
// 12, and the nine shifted copies are summed here.
//
// The value path, so that the rounding is on the record: the three temporal taps are accumulated
// in the convolution's fp32 accumulator and rounded to bf16 in g; the nine spatial taps are then
// summed here in fp32 from nine bf16 g values, the bias is added in fp32, and the result is
// rounded once to bf16 on the store. Main's path is also two roundings (cuDNN's bf16 conv_out,
// then finish_scalar's bf16 of x+bias); the count is the same and what differs is WHERE the first
// one sits - after three taps instead of after twenty-seven - which is why max|d| moves and rms
// does not. The reference accumulates all 27 taps in one fp32 accumulator.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include "head.h"

// One thread per output position. Its 12 accumulators take the 12 channels of the same (i,j)
// tap from nine neighbouring positions; the nine shifted reads of each 216-byte channel vector
// are served from L1, so the pass moves each element of g about once.
static __global__ void head_stencil_kernel(const __nv_bfloat16* __restrict__ g,
                                    const __nv_bfloat16* __restrict__ bias,
                                    __nv_bfloat16* __restrict__ out,
                                    int D,int rows,int W,int H,int g_off,int lo) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=D*rows*W)return;
  int w=idx%W, t=idx/W, r=t%rows, d=t/rows;
  int gh0=lo+r;
  const int gH=rows+2*g_off;
  float acc[12];
#pragma unroll
  for(int c=0;c<12;++c)acc[c]=__bfloat162float(bias[c]);
  for(int i=0;i<3;++i) {
    int gh=gh0+i-1;
    if(gh<0||gh>=H)continue;                       // conv3d's zero padding, in the plane
    int gr=r+i-1+g_off;
    if(gr<0||gr>=gH)continue;
    for(int j=0;j<3;++j) {
      int ww=w+j-1;
      if(ww<0||ww>=W)continue;                     // conv3d's zero padding, across the row
      const __nv_bfloat162* p=(const __nv_bfloat162*)(g+((((size_t)d*gH+gr)*W)+ww)*112+(i*3+j)*12);
#pragma unroll
      for(int k=0;k<6;++k) {
        float2 f=__bfloat1622float2(p[k]);
        acc[2*k]+=f.x;
        acc[2*k+1]+=f.y;
      }
    }
  }
  __nv_bfloat162* o=(__nv_bfloat162*)(out+((((size_t)d*rows+r)*W)+w)*12);
#pragma unroll
  for(int k=0;k<6;++k)
    o[k]=__floats2bfloat162_rn(fminf(1.f,fmaxf(-1.f,acc[2*k])),fminf(1.f,fmaxf(-1.f,acc[2*k+1])));
}

void head_stencil(const at::Tensor& g,const at::Tensor& bias,const at::Tensor& out,
                  int64_t H_global,int64_t g_off,int64_t lo) {
  int D=out.size(2),rows=out.size(3),W=out.size(4);
  TORCH_CHECK(g.is_contiguous(at::MemoryFormat::ChannelsLast3d)&&g.size(1)==112,
              "head_stencil: g must be [1,112,D,rows+2*g_off,W] channels-last");
  TORCH_CHECK(out.is_contiguous(at::MemoryFormat::ChannelsLast3d)&&out.size(1)==12,
              "head_stencil: out must be [1,12,D,rows,W] channels-last");
  // The kernel strides g by gH = rows+2*g_off taken from `out` and reads D*rows*W positions of it,
  // so a g of the wrong depth or height reads out of bounds rather than failing here.
  TORCH_CHECK(g.size(2)==out.size(2)&&g.size(3)==out.size(3)+2*g_off&&g.size(4)==out.size(4),
              "head_stencil: g must be [1,112,D,rows+2*g_off,W] of out's [1,12,D,rows,W]");
  c10::cuda::CUDAGuard guard(g.device());
  int64_t total=(int64_t)D*rows*W;
  int threads=256;
  int blocks=(int)((total+threads-1)/threads);
  head_stencil_kernel<<<blocks,threads,0,at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)g.data_ptr(),(const __nv_bfloat16*)bias.data_ptr(),
      (__nv_bfloat16*)out.data_ptr(),D,rows,W,(int)H_global,(int)g_off,(int)lo);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
