#include "kernels.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <algorithm>
using B = __nv_bfloat16;
union Pack { uint4 v; B b[8]; __nv_bfloat162 h[4]; };
static cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }
static const B* ptr(const Tensor& t) { return t.defined() ? (const B*)t.data_ptr<at::BFloat16>() : nullptr; }
static B* ptr(Tensor& t) { return (B*)t.data_ptr<at::BFloat16>(); }
Tensor nhwc_empty(int64_t c,int64_t t,int64_t h,int64_t w,const Tensor& like) {
  return at::empty({1,c,t,h,w},like.options().memory_format(at::MemoryFormat::ChannelsLast3d));
}
Tensor last_frames(const Tensor& x,int n,int ph) {
  int C=x.size(1),T=x.size(2),H=x.size(3)-2*ph,W=x.size(4);
  Tensor y=nhwc_empty(C,n,H,W,x);
  size_t row=(size_t)H*W*C*sizeof(B),src_pitch=(size_t)(H+2*ph)*W*C*sizeof(B);
  const B* source=ptr(x)+((int64_t)(T-n)*(H+2*ph)+ph)*W*C;
  C10_CUDA_CHECK(cudaMemcpy2DAsync(ptr(y),row,source,src_pitch,row,n,cudaMemcpyDeviceToDevice,stream()));
  return y;
}
// The halo edge pack/scatter lives in comm.cu (halo_pack/halo_unpack, vectorised), with the transport.
__global__ void finish_vector(B* x,const B* b,const B* s,int64_t n,int C) {
  for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n/8;i+=(int64_t)gridDim.x*blockDim.x) {
    Pack v, bias, skip;
    v.v=((uint4*)x)[i];bias.v=((const uint4*)b)[i%(C/8)];
    if(s) skip.v=((const uint4*)s)[i];
    #pragma unroll
    for(int j=0;j<4;j++) { v.h[j]=__hadd2(v.h[j],bias.h[j]); if(s) v.h[j]=__hadd2(v.h[j],skip.h[j]); }
    ((uint4*)x)[i]=v.v;
  }
}
__global__ void finish_scalar(B* x,const B* b,const B* s,int64_t n,int C) {
  for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(int64_t)gridDim.x*blockDim.x) {
    B v=__float2bfloat16(__bfloat162float(x[i])+__bfloat162float(b[i%C]));
    if(s) v=__float2bfloat16(__bfloat162float(v)+__bfloat162float(s[i])); x[i]=v;
  }
}
void finish(Tensor& x,const Tensor& b,const Tensor& s) {
  int C=x.size(1);int64_t n=x.numel();
  if(C%8==0) finish_vector<<<std::min((int64_t)2048,(n+2047)/2048),256,0,stream()>>>(ptr(x),ptr(b),ptr(s),n,C);
  else finish_scalar<<<std::min((int64_t)2048,(n+255)/256),256,0,stream()>>>(ptr(x),ptr(b),ptr(s),n,C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void write_kernel(const B* x,B* out,int64_t n,int T,int H,int W,int fullT,int fullH,int offsetT,int offsetH) {
  int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) {
    int c=i%12;int64_t p=i/12;int w=p%W,h=(p/W)%H,t=p/((int64_t)W*H);
    int color=c/4,dy=c%2,dx=(c%4)/2;
    int64_t o=(((int64_t)color*fullT+t+offsetT)*fullH+offsetH+2*h+dy)*(2*W)+2*w+dx;
    out[o]=__float2bfloat16(fminf(1.f,fmaxf(-1.f,__bfloat162float(x[i]))));
  }
}
void write_video(const Tensor& x,Tensor& out,int frame_offset,int row_offset) {
  write_kernel<<<(x.numel()+255)/256,256,0,stream()>>>(ptr(x),ptr(out),x.numel(),x.size(2),x.size(3),x.size(4),out.size(2),out.size(3),frame_offset,row_offset);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// phases_kernel used to sit here: PR17 replaces it, together with finish_phases below, with one
// bandwidth-bound pass in resample.cu that writes the interleaved tensor directly.
// F(4x4,3x3), with fp32 spatial transforms and bf16 transformed operands.
__device__ __forceinline__ void bt6(float v0,float v1,float v2,float v3,float v4,float v5,float* y) {
  float a=v4-4*v2,b=v3-4*v1,c=v4-v2,d=2*(v3-v1);
  y[0]=4*v0-5*v2+v4;y[1]=a+b;y[2]=a-b;y[3]=c+d;y[4]=c-d;y[5]=4*v1-5*v3+v5;
}
__global__ void winograd_input_kernel(const B* x,B* v,int C,int D,int H,int W,int PH,int TH,int TW) {
  int HP=H+2*PH,CV=C/2;int64_t n=(int64_t)D*TH*TW*CV;
  for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(int64_t)gridDim.x*blockDim.x) {
    int c=i%CV,p=i/CV,tx=p%TW,ty=(p/TW)%TH,t=p/(TW*TH);
    float d[2][6][6];
    #pragma unroll
    for(int a=0;a<6;a++) {
      int y=4*ty-1+a+PH;
      #pragma unroll
      for(int b=0;b<6;b++) {int xx=4*tx-1+b;float2 f=(y>=0&&y<HP&&xx>=0&&xx<W)?__bfloat1622float2(((const __nv_bfloat162*)x)[(((int64_t)t*HP+y)*W+xx)*CV+c]):make_float2(0,0);d[0][a][b]=f.x;d[1][a][b]=f.y;}
    }
    #pragma unroll
    for(int lane=0;lane<2;lane++) {
      #pragma unroll
      for(int b=0;b<6;b++) {float q[6];bt6(d[lane][0][b],d[lane][1][b],d[lane][2][b],d[lane][3][b],d[lane][4][b],d[lane][5][b],q);
        #pragma unroll
        for(int a=0;a<6;a++)d[lane][a][b]=q[a];}
    }
    #pragma unroll
    for(int a=0;a<6;a++) {float q0[6],q1[6];bt6(d[0][a][0],d[0][a][1],d[0][a][2],d[0][a][3],d[0][a][4],d[0][a][5],q0);bt6(d[1][a][0],d[1][a][1],d[1][a][2],d[1][a][3],d[1][a][4],d[1][a][5],q1);
      #pragma unroll
      for(int b=0;b<6;b++)((__nv_bfloat162*)v)[(6*a+b)*n+i]=__floats2bfloat162_rn(q0[b],q1[b]);}
  }
}
Tensor winograd_input(const Tensor& x,int ph) {
  int C=x.size(1),D=x.size(2),H=x.size(3)-2*ph,W=x.size(4),TH=(H+3)/4,TW=(W+3)/4;
  auto v=at::empty({36,D,TH,TW,C},x.options());int64_t n=(int64_t)D*TH*TW*C;
  winograd_input_kernel<<<std::min((int64_t)2048,(n+255)/256),256,0,stream()>>>(ptr(x),ptr(v),C,D,H,W,ph,TH,TW);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return v;
}
// (WinoPtrs, at6, winograd_output_kernel and winograd_output are deleted: with the fused inverse
//  in wino_epilogue.cu nothing calls them. That file's a6v reads the same A^T out of shared memory
//  instead of 36 separate global pointers, and it is the only copy of the transform left there.)

