// Fused F(4x4,3x3) inverse transform for the causal 3x3x3 Winograd convolutions.
// One pass over the 36 component tensors M produces BOTH
//   y = A^T M A + bias + residual         (raw conv output: the resnet's residual operand)
//   n = silu(rmsnorm(y) * sqrt(C) * gain) (the next convolution's normalized, activated,
//                                          temporal+spatial padded input)
// so the normalized tensor is never written and read again by a separate norm pass.
// The RMS statistic needs every output channel of a position, so one block holds all of C
// for one 4x4 output tile: blockDim.x == C/2, one channel pair per thread.
//
// M layout: 36 separate allocations, each [D][TH][TW][C] bf16, C innermost, so a plane's
// position stride is 1 in C and its plane offset is (((t*TH+ty)*TW+tx)*C + ch).
// y is [D][H][W][C]; n is [D+PT][H+2PH][W][C].
//
// Two lessons from the guild, both measured elsewhere and both applied here:
//  - w01's J576 (input transform): the head carries no divisions, no predicates and no 64-bit
//    address chain. 3-D grid (tx,ty,t), a separate EXACT instantiation with the bounds test
//    removed, and a 32-bit plane byte offset. EXACT is dispatched per call: it holds at
//    88x156, 176x312 and 352x624, whose widths are multiples of 4 -- NOT at 44x78, where
//    W=78 leaves the last tile two columns short and the checked path is taken.
//  - w05's J627 (inverse transform, 5.5 TB/s at 64 registers): no thread loads 4 B of one plane.
//    The block stages the 36 components of its tile into shared memory cooperatively -- one
//    plane's C channels is C*2 contiguous bytes, so a warp pass is 16 B per lane and one fully
//    coalesced 512 B request -- and the consumer reads smem[comp][2c] as a 4-byte pair, which is
//    128 contiguous bytes per warp and therefore conflict-free. No transpose on either side.
//
// Per-element work follows w02's recipe (M137, PR15): bf16->f32 by an integer shift so the
// conversions leave the XU pipe, one bf16 rounding for the whole normalize*sqrt(C)*gain product
// instead of the reference's three, SiLU as u + u*tanh(u) with u = pre-activation/2 and
// tanh.approx.f32, and 0.5*sqrt(C)*rsqrt(sum) as one per-position scalar.
#include "kernels.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <algorithm>
using B=__nv_bfloat16;
static cudaStream_t e_stream(){return at::cuda::getCurrentCUDAStream();}
static const B* e_ptr(const Tensor& t){return t.defined()?(const B*)t.data_ptr<at::BFloat16>():nullptr;}
static B* e_mut(Tensor& t){return t.defined()?(B*)t.data_ptr<at::BFloat16>():nullptr;}

struct EpiPtrs{const B* p[36];};

// A of F(4x4,3x3) applied along one axis: out[i] = sum_v A[i][v] v[v], two fp32 lanes per
// instruction (a6v below is the only form this file uses). add/sub/fma.f32x2 need sm_100+; this
// tree is built for sm_100a only (solution.json), so no scalar fallback is kept beside it -- a
// scalar copy would be dead source here, which is why the scalar `a6` was removed rather than left.
// f32x2: PTX 8.6 gives add/sub/mul/fma on a .b64 operand holding two f32 lanes (sm_100a).
// This kernel already carries two independent fp32 lanes per thread -- the channel pair a thread
// owns -- so `ch` IS the SIMD lane and nothing crosses a lane. Each instruction performs exactly
// the arithmetic the scalar instruction performs, in the same association.
struct V2{
  float x,y;
  __device__ V2(){} __device__ V2(float a,float b):x(a),y(b){}
};
__device__ __forceinline__ unsigned long long pk(V2 a){
  return (unsigned long long)__float_as_uint(a.y)<<32|__float_as_uint(a.x);
}
__device__ __forceinline__ V2 upk(unsigned long long u){
  V2 r; r.x=__uint_as_float((unsigned)u); r.y=__uint_as_float((unsigned)(u>>32)); return r;
}
__device__ __forceinline__ V2 vadd(V2 a,V2 b){
  unsigned long long r; asm("add.rn.f32x2 %0,%1,%2;":"=l"(r):"l"(pk(a)),"l"(pk(b))); return upk(r);
}
__device__ __forceinline__ V2 vsub(V2 a,V2 b){
  unsigned long long r; asm("sub.rn.f32x2 %0,%1,%2;":"=l"(r):"l"(pk(a)),"l"(pk(b))); return upk(r);
}
__device__ __forceinline__ V2 vfmas(V2 a,float c,V2 acc){
  V2 k; k.x=k.y=c; unsigned long long r;
  asm("fma.rn.f32x2 %0,%1,%2,%3;":"=l"(r):"l"(pk(a)),"l"(pk(k)),"l"(pk(acc))); return upk(r);
}
__device__ __forceinline__ V2 b2v(uint32_t w){
  V2 r; r.x=__uint_as_float(w<<16); r.y=__uint_as_float(w&0xffff0000u); return r;
}
// The transform of A above, two lanes per instruction.
__device__ __forceinline__ void a6v(V2 v0,V2 v1,V2 v2,V2 v3,V2 v4,V2 v5,V2* y){
  V2 a=vadd(v1,v2), b=vsub(v1,v2), c=vadd(v3,v4), d=vsub(v3,v4);
  y[0]=vadd(vadd(v0,a),c); y[1]=vfmas(d,2.f,b); y[2]=vfmas(c,4.f,a); y[3]=vadd(vfmas(d,8.f,b),v5);
}
// Exact bf16 pair -> two floats without the XU pipe: a bf16 is the top half of an fp32.
__device__ __forceinline__ float2 b2f(uint32_t w){
  return make_float2(__uint_as_float(w<<16),__uint_as_float(w&0xffff0000u));
}
__device__ __forceinline__ float tanh_ap(float x){float r;asm("tanh.approx.f32 %0, %1;":"=f"(r):"f"(x));return r;}
// silu(x) with u = x/2: x*sigmoid(x) = u + u*tanh(u).
__device__ __forceinline__ float silu2(float u){return __fmaf_rn(u,tanh_ap(u),u);}
__device__ __forceinline__ uint32_t add2(uint32_t a,uint32_t b){
  __nv_bfloat162 x=*(const __nv_bfloat162*)&a,y=*(const __nv_bfloat162*)&b;
  __nv_bfloat162 r=__hadd2(x,y); return *(const uint32_t*)&r;
}
__device__ __forceinline__ uint32_t pack2(float x,float y){
  __nv_bfloat162 r=__floats2bfloat162_rn(x,y); return *(const uint32_t*)&r;
}
__device__ __forceinline__ float sq8(const uint4& v){
  float2 a=b2f(v.x),b=b2f(v.y),c=b2f(v.z),d=b2f(v.w);
  return (__fmaf_rn(a.x,a.x,a.y*a.y)+__fmaf_rn(b.x,b.x,b.y*b.y))
        +(__fmaf_rn(c.x,c.x,c.y*c.y)+__fmaf_rn(d.x,d.x,d.y*d.y));
}

// Stage the 36 components of one 4x4 tile into shared memory: 36*C/8 tasks of 16 bytes,
// 9 per thread at any C. Each warp's 32 lanes take 32 consecutive 16-byte slots, i.e. 512
// contiguous bytes of one plane.
//
// The copy is `cp.async.cg` (16 B, global->shared, L2-cached, NO register round trip) rather than
// the `uint4` load-then-store it replaces.  Measured reason (J3871, wino_epi_kernel<1024,1>, ncu,
// 3 passes): the binding FIXABLE stall is `mio_throttle 2.11` cycles per issued instruction --
// second only to `not_selected 2.73`, which is not a defect (it means the SM already has eligible
// warps to pass over) -- and `l1tex__throughput` is the highest pipe at 62.7-64.7 %.  The old form
// is a global load AND a shared store per 16 B, so it spends two MIO slots and a writeback
// register per slot; this form spends one slot and no register.  Per block at C=1024 that is
// 9*512 = 4,608 pairs, i.e. 9,216 MIO instructions becoming 4,608.
//
// Bytes moved, alignment and the resulting V are IDENTICAL: same comp/off decomposition, same
// 16-byte slot, same plane offset inside smem, so the transform reads what it read before and the
// output is bit-for-bit the old output.  This is a scheduling change, not a numerical one.
//
// The commit/wait pair sits here rather than at the call site because the transform consumes the
// staging immediately -- there is no independent work to overlap the transfer with inside one
// tile.  Overlap across tiles needs a persistent loop and is a separate change, not this one.
// The form above is measured, not preferred: `cp.async.cg` beats the `uint4` load+store it
// replaced by 12.7 ms at f121x1 (J4745, one job, one clock, A/A 0.2 ms).
template<int C>
__device__ __forceinline__ void stage36(B* sm,const EpiPtrs& mp,unsigned src_tile){
  constexpr int CV=C/2;
  constexpr int PER=C/8;                 // 16-byte slots per plane
  const unsigned smbase=(unsigned)__cvta_generic_to_shared(sm);
  #pragma unroll
  for(int i=0;i<9;i++){
    const int task=threadIdx.x+i*CV;
    const int comp=task/PER, off=task%PER;
    const unsigned dst=smbase+(unsigned)((comp*C+off*8)*2);
    const char* src=(const char*)mp.p[comp]+src_tile+off*16;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(dst),"l"(src));
  }
  asm volatile("cp.async.commit_group;\n"::);
  asm volatile("cp.async.wait_group 0;\n"::);
}

template<int C,bool EXACT>
__global__ __launch_bounds__(C/2) void wino_epi_kernel(
    EpiPtrs mp,B* __restrict__ y,B* __restrict__ n,
    const B* __restrict__ bias,const B* __restrict__ residual,const B* __restrict__ gain,
    int H,int W,int TH,int TW,int PT,int PH,bool wy){
  // `wy` is the CALLER's assertion that nobody reads this launch's y. It is only ever false where
  // the very next conv has idx == fnext and therefore takes the fused `prepared` edge (see
  // binding.cpp: resnet's first conv, whose consumer is its own second conv). The n tensor is
  // written either way, so this can never be false without a n to carry the activation.
  constexpr int CV=C/2;          // channel pairs per position
  constexpr int WP=CV/32;        // warps per block
  const int c=threadIdx.x;
  const int tx=blockIdx.x, ty=blockIdx.y, t=blockIdx.z;
  extern __shared__ __align__(16) B smem[];
  B* sm=smem;                                              // 36 planes x C, this tile
  // 32-bit plane byte offset: D*TH*TW*C*2 bytes, where D is the PIXEL frame count at this level
  // (4*chunk-3 at up block 1, not the latent chunk size -- the earlier "34" was the latent count).
  // chunk_size is NOW min(16*world, 61) -- this PR is what moved the rule off `8*world`, so chunk
  // 32 at world 4 no longer exists and the worst case moves with it. Over the shipping configs the
  // largest src_tile is world 2 / chunk 32 = 4.39e8 bytes and world 1 / chunk 16 = 4.29e8, both
  // ~10x inside UINT32_MAX and both enforced at run time by the guard further down this file.
  // The `8.8e8` that stood here was 4x too large and it described a configuration this PR retires.
  // A chunk size that more than doubles again, or a wider clip, must widen this to unsigned long long.
  const unsigned src_tile=(unsigned)((((t*TH+ty)*TW+tx)*C)*2u);

  stage36<C>(sm,mp,src_tile);
  __syncthreads();

  V2 acc[4][4];
  #pragma unroll
  for(int i=0;i<4;i++)
    #pragma unroll
    for(int b=0;b<4;b++) acc[i][b]=V2(0.f,0.f);

  const B* q0=sm+2*c;                     // this thread's channel pair, in each plane
  #pragma unroll
  for(int u=0;u<6;u++){
    V2 f[6];
    #pragma unroll
    for(int v=0;v<6;v++) f[v]=b2v(*(const uint32_t*)(q0+(6*u+v)*C));
    V2 q[4];
    a6v(f[0],f[1],f[2],f[3],f[4],f[5],q);
    #pragma unroll
    for(int b=0;b<4;b++){
      const V2 v=q[b];
      if(u==0) acc[0][b]=vadd(acc[0][b],v);
      else if(u==1){acc[0][b]=vadd(acc[0][b],v);acc[1][b]=vadd(acc[1][b],v);acc[2][b]=vadd(acc[2][b],v);acc[3][b]=vadd(acc[3][b],v);}
      else if(u==2){acc[0][b]=vadd(acc[0][b],v);acc[1][b]=vsub(acc[1][b],v);acc[2][b]=vadd(acc[2][b],v);acc[3][b]=vsub(acc[3][b],v);}
      else if(u==3){acc[0][b]=vadd(acc[0][b],v);acc[1][b]=vfmas(v,2.f,acc[1][b]);acc[2][b]=vfmas(v,4.f,acc[2][b]);acc[3][b]=vfmas(v,8.f,acc[3][b]);}
      else if(u==4){acc[0][b]=vadd(acc[0][b],v);acc[1][b]=vfmas(v,-2.f,acc[1][b]);acc[2][b]=vfmas(v,4.f,acc[2][b]);acc[3][b]=vfmas(v,-8.f,acc[3][b]);}
      else {acc[3][b]=vadd(acc[3][b],v);}
    }
  }

  const int Hp=H+2*PH;
  float2 biasv=make_float2(0.f,0.f),gainv=make_float2(0.f,0.f);
  if(bias) biasv=b2f(*(const uint32_t*)&((const __nv_bfloat162*)bias)[c]);
  if(gain) gainv=b2f(*(const uint32_t*)&((const __nv_bfloat162*)gain)[c]);
  const long yb=(((long)t*H+4*ty)*W+4*tx)*CV+c;

  const __nv_bfloat162 bv=__floats2bfloat162_rn(biasv.x,biasv.y);
  if(!n){
  // No prepared tensor: write the raw conv output in the natural per-thread layout.
  #pragma unroll
  for(int i=0;i<4;i++){
    const int yy=4*ty+i;
    #pragma unroll
    for(int b=0;b<4;b++){
      const int xx=4*tx+b;
      const long off=(long)(i*W+b)*CV;
      const bool ok=EXACT?true:(yy<H&&xx<W);
      __nv_bfloat162 v=__floats2bfloat162_rn(acc[i][b].x,acc[i][b].y);
      if(bias) v=__hadd2(v,bv);
      if(residual&&ok) v=__hadd2(v,((const __nv_bfloat162*)residual)[yb+off]);
      if(ok) ((__nv_bfloat162*)y)[yb+off]=v;
    }
  }
  }

  if(n){
    // Remapped epilogue (w01's M655, measured 1.58x on the standalone inverse): the thread that
    // computes the transform owns a channel pair, so it would write 4 bytes per output, which
    // caps the pass at the 4-byte row of the stream table. Instead the 16 outputs go to shared
    // memory with the bias already applied, one barrier, and the epilogue is re-laid so a warp
    // owns one position and each lane an octet of channels: the residual read and the y write are
    // one 512-byte transaction per warp instruction, and the sum of squares becomes a single-warp
    // butterfly because a warp now holds every channel of its position.
    __syncthreads();                       // the components are dead; their buffer is reused
    B* so=smem;                            // so[p*C + ch], bias applied
    #pragma unroll
    for(int i=0;i<4;i++)
      #pragma unroll
      for(int b=0;b<4;b++){
        __nv_bfloat162 v=__floats2bfloat162_rn(acc[i][b].x,acc[i][b].y);
        if(bias) v=__hadd2(v,bv);
        ((__nv_bfloat162*)so)[(i*4+b)*CV+c]=v;
      }
    __syncthreads();
    constexpr int NPW=16/WP;               // positions per warp (4, 2 or 1)
    constexpr int CW=C/32;                 // channels per lane (8, 16 or 32)
    constexpr int NU=CW/8;                 // 16-byte chunks per lane (1, 2 or 4)
    const int wp=c>>5, ln=c&31;
    const float hs=0.5f*sqrtf((float)C);
    uint4 gv[NU];
    // Chunk map: lane ln takes chunk (k*32+ln), i.e. channel (k*32+ln)*8, so the 32 lanes of one
    // warp instruction cover 32 consecutive chunks = 512 contiguous bytes. The previous map gave
    // lane ln the block [ln*CW, ln*CW+CW), which puts neighbours CW*2 bytes apart: contiguous at
    // C=256 (16 B), half a line at C=512 (32 B), a quarter at C=1024 (64 B) -- and at C=1024 that
    // is a 4x sector over-fetch on the residual read and on both stores. w21's J1096 gives it in
    // counters (L1 hit 5.32%, L2 hit 0.20%, DRAM 65.6%): no reuse, so every over-fetched sector
    // is paid at DRAM. Measured at D=8, min of three rounds: C=256 1.000x (identical expression at
    // NU=1), C=512 1.244x, C=1024 1.431x. It permutes which channel each k reads, so the 4-term
    // sum of squares accumulates in a different order and s rounds differently in its last bits --
    // f121x1 max|d| 0.1605 -> 0.147, rms unchanged to five digits.
    #pragma unroll
    for(int k=0;k<NU;k++)
      gv[k]=gain?*(const uint4*)((const char*)gain+((long)(k*32+ln)*8)*2)
               :make_uint4(0x3f803f80u,0x3f803f80u,0x3f803f80u,0x3f803f80u);
    #pragma unroll
    for(int j=0;j<NPW;j++){
      const int p=wp*NPW+j, yy=4*ty+(p>>2), xx=4*tx+(p&3);
      const bool ok=EXACT?true:(yy<H&&xx<W);
      uint4 yv4[NU]; float sq=0.f;
      #pragma unroll
      for(int k=0;k<NU;k++){
        const int off=(k*32+ln)*8;
        uint4 v=*(const uint4*)((const char*)so+((long)p*C+off)*2);
        if(residual&&ok){
          // (long)t first: the whole offset must be 64-bit, not the last factor. `H` here is this
          // RANK'S BAND height, not the full 352 -- chunk() narrows dim 3 of the latent grid (44
          // rows) to [44r/W,44(r+1)/W) before resnet15 and every up block, so block 3 carries 88
          // rows at world 4 and 44 at world 8. The band shrinks as the temporal chunk grows, which
          // is why a rank's working set is near-constant -- BUT THE CENSUS THAT STOOD HERE WAS
          // TRUE ONLY OF `chunk_size = 8*world`, AND THIS PR MOVES THE RULE TO `min(16*world, 61)`.
          // Post-change THREE cells pass INT_MAX and THE LARGEST IS WORLD 2, so it is listed
          // first: narrowing against world 4 because "chunk 61 looks biggest" is exactly the
          // mislabelled-maximum this paragraph exists to prevent.
          //   world 2 / chunk 32   D=125, rows 176, 624x256   = 3.51e9   <-- THE MAXIMUM
          //   world 1 / chunk 16   D=61,  rows 352            = 3.43e9
          //   world 4 / chunk 61   D=241, rows 88             = 3.39e9
          // World 2's chunk 32 is THIS PR's OWN value at world 2 (min(16*2, 61) = 32), not a
          // foreign or historical form. ALL THREE ARE PAST INT_MAX. This site is
          // correct because it chains `(long)` here and again on `nbase` below, so the 64-bit chain
          // is now LOAD-BEARING at those three cells rather than merely belt-and-braces. Do not
          // narrow it back on the strength of an `inside INT_MAX` reading from the old rule.
          const uint4 r=*(const uint4*)((const char*)residual
                        +((long)((((long)t*H+yy)*W+xx)*C+off))*2);
          v.x=add2(v.x,r.x); v.y=add2(v.y,r.y); v.z=add2(v.z,r.z); v.w=add2(v.w,r.w);
        }
        yv4[k]=v; sq+=sq8(v);
      }
      #pragma unroll
      for(int d=16;d;d>>=1) sq+=__shfl_xor_sync(0xffffffffu,sq,d);
      if(ok){
        const float s=hs*rsqrtf(fmaxf(sq,1e-24f));
        const long ybase=((((long)t*H+yy)*W+xx)*C)*2;
        const long nbase=(((((long)(t+PT)*Hp+PH+yy)*W+xx)*C))*2;
        #pragma unroll
        for(int k=0;k<NU;k++){
          const long cof=(long)((k*32+ln)*8)*2;
          if(wy) *(uint4*)((char*)y+ybase+cof)=yv4[k];
          uint4 u=yv4[k], g=gv[k], o;
          float2 f0=b2f(u.x), f1=b2f(u.y), f2=b2f(u.z), f3=b2f(u.w);
          float2 g0=b2f(g.x), g1=b2f(g.y), g2=b2f(g.z), g3=b2f(g.w);
          o.x=pack2(silu2(f0.x*s*g0.x),silu2(f0.y*s*g0.y));
          o.y=pack2(silu2(f1.x*s*g1.x),silu2(f1.y*s*g1.y));
          o.z=pack2(silu2(f2.x*s*g2.x),silu2(f2.y*s*g2.y));
          o.w=pack2(silu2(f3.x*s*g3.x),silu2(f3.y*s*g3.y));
          *(uint4*)((char*)n+nbase+cof)=o;
        }
      }
    }
  }
}

// Zero the spatial padding rows of every frame of the prepared tensor.
__global__ void epi_border(B* n,int C,int T,int Hp,int W){
  const int CV=C/2; const long per=(long)2*W*CV;
  const long total=(long)T*per;
  for(long i=(long)blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=(long)gridDim.x*blockDim.x){
    const int t=(int)(i/per); const long r=i%per; const int edge=(int)(r/((long)W*CV));
    const long r2=r%((long)W*CV); const int w=(int)(r2/CV),cc=(int)(r2%CV);
    const long idx=(((long)t*Hp+(edge?Hp-1:0))*W+w)*CV+cc;
    ((__nv_bfloat162*)n)[idx]=__floats2bfloat162_rn(0.f,0.f);
  }
}
__global__ void epi_zero(B* n,long pairs){
  const __nv_bfloat162 z=__floats2bfloat162_rn(0.f,0.f);
  for(long i=(long)blockIdx.x*blockDim.x+threadIdx.x;i<pairs;i+=(long)gridDim.x*blockDim.x)
    ((__nv_bfloat162*)n)[i]=z;
}

template<int C,bool EXACT>
static void launch_epi(const dim3& grid,const dim3& block,size_t smem,
                       const EpiPtrs& p,B* y,B* n,
                       const B* b,const B* r,const B* g,
                       int H,int W,int TH,int TW,int PT,int PH,bool wy,cudaStream_t s){
  auto k=wino_epi_kernel<C,EXACT>;
  if(smem>48*1024){
    static bool done[2]={false,false};
    if(!done[EXACT?1:0]){
      C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,36*C*2));
      done[EXACT?1:0]=true;
    }
  }
  k<<<grid,block,smem,s>>>(p,y,n,b,r,g,H,W,TH,TW,PT,PH,wy);
}

WinoOut wino_epilogue(const std::vector<Tensor>& m,int H,int W,const Tensor& bias,
                      const Tensor& residual,const Tensor& gamma,const Tensor& hist,int pt,int ph,
                      bool write_y){
  // `write_y=false` is admissible ONLY when the caller's next conv has idx == the `fnext` this call
  // prepares for, so that next conv takes the `prepared` edge and never reads the returned y. Out.y
  // is still a distinct allocation: its unread contents are undefined, not zero.
  TORCH_CHECK(m.size()==36&&m[0].dim()==5,"wino_epilogue: M list ",m.size()," rank ",m[0].dim());
  const int C=m[0].size(1),D=m[0].size(2),TH=m[0].size(3),TW=m[0].size(4);
  TORCH_CHECK(C==256||C==512||C==1024,"wino_epilogue: unsupported channel count");
  auto s=e_stream();
  EpiPtrs p; for(int i=0;i<36;i++) p.p[i]=(const B*)m[i].data_ptr<at::BFloat16>();
  WinoOut out;
  out.y=nhwc_empty(C,D,H,W,m[0]);
  const int Hp=H+2*ph;
  if(gamma.defined()){
    out.n=nhwc_empty(C,D+pt,Hp,W,m[0]);
    B* np=e_mut(out.n);
    const long pairs=(long)(D+pt)*Hp*W*(C/2);
    if(pt>0&&!hist.defined())
      epi_zero<<<(int)std::min((long)2048,((long)pt*Hp*W*(C/2)+4095)/4096),256,0,s>>>(np,(long)pt*Hp*W*(C/2));
    if(ph>0)
      epi_border<<<(int)std::min((long)2048,(pairs+4095)/4096),256,0,s>>>(np,C,D+pt,Hp,W);
    if(pt>0&&hist.defined()){
      // `hist` is a `last_frames` output: a COMPACT [C,pt,H,W] channels-last-3d tensor, so its
      // per-frame SOURCE pitch is H*W*C -- unpadded. Only the DESTINATION is padded: out.n is
      // [C,D+pt,Hp,W] with pitch Hp*W*C, and these frames land in its interior rows [ph,ph+H).
      // The two pitches are one of each and neither is a typo: passing Hp*W*C as the source pitch
      // would read pt frames of garbage, and passing H*W*C as the destination pitch would write
      // the pad rows away. If `last_frames` changes its shape or its pitch, this copy changes.
      const size_t row=(size_t)H*W*C*sizeof(B);
      C10_CUDA_CHECK(cudaMemcpy2DAsync(np+(long)ph*W*C,(size_t)Hp*W*C*sizeof(B),
                                       e_ptr(hist),row,row,pt,cudaMemcpyDeviceToDevice,s));
    }
  }
  // src_tile in the kernel is the only 32-bit index in this file, so assert here what it relies on
  // rather than trusting the kernel's comment. (The y and n offsets in the kernel body are formed in
  // 64-bit and are not bounded by this.) Checked against UINT32_MAX, not INT32_MAX: the index is
  // unsigned, and a check against the signed bound would refuse shapes that are in fact safe.
  { const unsigned long long off=(((unsigned long long)(D-1)*TH+(TH-1))*TW+(TW-1))*(unsigned long long)C*2ull;
    TORCH_CHECK(off<=0xFFFFFFFFull,"wino_epilogue: 32-bit tile offset ",off,
                " exceeds UINT32_MAX; widen src_tile in wino_epilogue.cu"); }
  const bool exact=(TH*4==H&&TW*4==W);
  const dim3 grid(TW,TH,D);
  const dim3 block(C/2);
  const size_t shmem=(size_t)36*C*2;
  B* yp=e_mut(out.y); B* np=e_mut(out.n);
  const B* bp=e_ptr(bias); const B* rp=e_ptr(residual); const B* gp=e_ptr(gamma);
  if(C==256){ if(exact) launch_epi<256,true>(grid,block,shmem,p,yp,np,bp,rp,gp,H,W,TH,TW,pt,ph,write_y,s);
              else launch_epi<256,false>(grid,block,shmem,p,yp,np,bp,rp,gp,H,W,TH,TW,pt,ph,write_y,s); }
  else if(C==512){ if(exact) launch_epi<512,true>(grid,block,shmem,p,yp,np,bp,rp,gp,H,W,TH,TW,pt,ph,write_y,s);
                   else launch_epi<512,false>(grid,block,shmem,p,yp,np,bp,rp,gp,H,W,TH,TW,pt,ph,write_y,s); }
  else { if(exact) launch_epi<1024,true>(grid,block,shmem,p,yp,np,bp,rp,gp,H,W,TH,TW,pt,ph,write_y,s);
         else launch_epi<1024,false>(grid,block,shmem,p,yp,np,bp,rp,gp,H,W,TH,TW,pt,ph,write_y,s); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
