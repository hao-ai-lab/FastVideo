// Wan2.2 VAE decoder seed. Derived from the reference architecture and
// indexing in diffusers' AutoencoderKLWan (Wan Team / Hugging Face Team,
// Apache-2.0; see WAN_LICENSE.txt). Native CUDA execution is implemented here.
// All invocation state, including temporal history, belongs to Decoder below.
#include <torch/extension.h>
#include <ATen/ops/cudnn_convolution.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/record_function.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include "kernels.h"
#include "prefix.h"
#include "comm.h"
#include "resample.h"
#include "attn.h"
#include "head.h"
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

at::Tensor custom_conv(const at::Tensor&,const at::Tensor&,int64_t,int64_t,int64_t);

// The temporal upsampler, writing the frame-interleaved video directly (solution/timeconv.cu).
at::Tensor time_upsample(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,int64_t,bool);

using PG = c10::intrusive_ptr<c10d::ProcessGroup>;
using at::Tensor;
using Weights = std::vector<Tensor>;

Weights prepare(const Weights& weights,int64_t rank,int64_t world,const PG& pg) {
  TORCH_CHECK(weights.size()==110,"Expected the 110 decoder tensors");
  at::NoGradGuard guard;
  Weights result;
  result.reserve(weights.size());
  for(const auto& w:weights) {
    // A 1x1x1 tensor is "trivially channels-last", so contiguous() leaves it with unit strides on
    // its size-1 spatial dims; CUTLASS's im2col TMA needs those strides to be 16-byte multiples.
    // The same bytes with dense strides: as_strided keeps the NDHWC order but gives every spatial
    // dim the channel-count stride.
    if(w.dim()==5 && w.size(2)==1 && w.size(3)==1 && w.size(4)==1) {
      auto wc=w.contiguous();int64_t C=wc.size(1);
      result.push_back(wc.as_strided({wc.size(0),C,1,1,1},{C,1,C,C,C}));
    }
    else if(w.dim()==5) result.push_back(w.contiguous(at::MemoryFormat::ChannelsLast3d));
    else if(w.dim()==4 && w.size(2)>0 && w.size(1)>1)
      result.push_back(w.unsqueeze(2).contiguous(at::MemoryFormat::ChannelsLast3d));
    else result.push_back(w.contiguous());
  }
  // The output head: conv_out's nine spatial taps become the output channels of a 3x1x1
  // (temporal-only) convolution, so its 112 outputs are (tap i,j) x (channel c) and head.cu's
  // epilogue sums the nine shifted copies. W'[(i*3+j)*12+c, ch, t] = W[c, ch, t, i, j]; the four
  // channels past 108 are zero and are never read. N=112 rather than 108 keeps the epilogue's
  // channel count a multiple of 16.
  {
    auto orig=result[108];
    auto t=orig.permute({3,4,0,1,2}).contiguous();          // [i,j,c,ch,t] = [3,3,12,256,3]
    auto filled=at::zeros({112,256,3},orig.options());      // [n,ch,t], n=(i*3+j)*12+c
    filled.narrow(0,0,108).copy_(t.reshape({108,256,3}));
    auto kt=filled.transpose(1,2).contiguous();             // [n,t,ch] contiguous
    // The layout a [K,C,3,3,3] weight has under ChannelsLast3d: C innermost with stride 1, every
    // other stride a multiple of 16 bytes. torch normalises a size-1 H/W away, hence as_strided.
    result[108]=kt.transpose(1,2).as_strided({112,256,3,1,1},{768,1,256,256,256});
  }
  // Nearest-x2 + 3x3 convolution equals four 2x2 low-resolution phases.
  // Form the sums in fp32 and store the transformed weights in bf16.
  for(int spatial : {39,61,85}) {
    auto orig=result[spatial].to(at::kFloat);
    int C=orig.size(0);
    for(int a=0;a<2;++a)for(int b=0;b<2;++b) {
      auto transformed=at::zeros({C,C,1,2,2},orig.options());
      for(int u=0;u<2;++u)for(int v=0;v<2;++v) {
        auto sum=at::zeros({C,C},orig.options());
        for(int r=0;r<3;++r)for(int c=0;c<3;++c) {
          int dy=a+r-1,dx=b+c-1;
          int iy=dy<0?-1:dy/2,ix=dx<0?-1:dx/2;
          if(iy==u+a-1&&ix==v+b-1)sum.add_(orig.select(2,0).select(2,r).select(2,c));
        }
        transformed.select(2,0).select(2,u).select(2,v).copy_(sum);
      }
      result.push_back(transformed.to(at::kBFloat16).contiguous(at::MemoryFormat::ChannelsLast3d));
    }
  }
  // F(4x4,3x3) Winograd weights for up blocks 1, 2, and 3.
  // The 36 components each retain the causal 3-tap temporal convolution.
  const float G[6][3]={{.25f,0,0},{-1.f/6,-1.f/6,-1.f/6},{-1.f/6,1.f/6,-1.f/6},{1.f/24,1.f/12,1.f/6},{1.f/24,-1.f/12,1.f/6},{0,0,1}};
  for(int idx : {44,47,50,53,56,59,66,69,74,77,80,83,88,91,96,99,102,105}) {
    auto orig=result[idx].to(at::kFloat);
    for(int a=0;a<6;a++)for(int b=0;b<6;b++) {
      auto sum=at::zeros({orig.size(0),orig.size(1),3},orig.options());
      for(int r=0;r<3;r++)for(int c=0;c<3;c++)if(G[a][r]*G[b][c]!=0)
        sum.add_(orig.select(3,r).select(3,c),G[a][r]*G[b][c]);
      result.push_back(sum.unsqueeze(3).unsqueeze(4).to(at::kBFloat16).contiguous(at::MemoryFormat::ChannelsLast3d));
    }
  }
  // The 1x1x1 shortcut of a channel-changing resnet is added to the second convolution's
  // bias + output, out = round(acc2 + b2) + round(acc1 + b1). Fold b1 into b2 instead (one more
  // bf16 rounding of the bias sum and none of the operand): the shortcut tensor then needs no
  // separate bias pass, which was a full read+write of it.
  for(auto pair : {std::pair<int,int>{72,70},std::pair<int,int>{94,92}}) {
    auto sum=result[pair.second].to(at::kFloat)+result[pair.first].to(at::kFloat);
    result[pair.second]=sum.to(at::kBFloat16).contiguous();
  }
  return result;
}

struct Decoder {
  // This must not implicitly bind to a fused-epilogue weight index when
  // independently extended conv() interfaces are merged.
  struct TemporalTrim {
    int frames;
    explicit constexpr TemporalTrim(int n) : frames(n) {}
  };

  const Weights& w;
  int rank,world;
  PG pg;
  std::array<Tensor,110> cache;
  bool sharded=false;
  // A convolution whose input was already normalized, activated and padded by the previous
  // Winograd convolution's fused epilogue: (tensor, the conv index it is prepared for, and the
  // sharding it was prepared under — a mismatch there would be silently wrong, not a crash).
  Tensor prepared; int prepared_for=-1; bool prepared_sharded=false;
  // The conv index whose y store was suppressed because this conv reserved it for `prepared`.
  // It is the RUNTIME half of the y_dead contract: the producer states an intent and this records
  // it, so a later edit that stops the consumer taking the `pre` edge aborts instead of handing a
  // never-written tensor to norm_pad. Cleared when the edge is taken, and asserted closed at the
  // end of every chunk beside prepared_for, because a conv index restarts at the same values.
  int y_suppressed_for=-1;
  Decoder(const Weights& weights,int r,int n,const PG& p):w(weights),rank(r),world(n),pg(p){}

  Tensor conv(Tensor x,int idx,int gamma=-1,bool activate=true,bool add_bias=true,const Tensor& residual={},TemporalTrim trim=TemporalTrim{0},int fg=-1,int fnext=-1,bool y_dead=false) {
    RECORD_FUNCTION("wan.conv."+std::to_string(idx), std::vector<c10::IValue>());
    const int temporal_trim=trim.frames;
    int kt=w[idx].size(2),kh=w[idx].size(3),kw=w[idx].size(4);
    TORCH_CHECK(temporal_trim>=0 && temporal_trim<kt,"Invalid causal prefix trim");
    int pt=kt-1-temporal_trim,ph=(sharded&&world>1&&kh==3)?1:0;
    Tensor p=x;
    bool pre=(prepared_for==idx);
    // The consumer asserts the producer's y_dead contract AT RUNTIME, not in prose. If a conv had
    // its y store suppressed for this idx and this conv does not take the fused `prepared` edge,
    // it is about to read a tensor that was never written -- so abort instead of producing a wrong
    // video. The existing metadata checks at `pre` cannot catch this, because they live inside the
    // branch that is not being taken.
    TORCH_CHECK(pre || y_suppressed_for!=idx,
                "conv ",idx,": its y store was suppressed FOR THIS CONV, which then did not take "
                "the prepared edge; the tensor it is about to read is unwritten");
    // Prefix pruning never uses the fused, full-length prepared-input edge.
    // A future caller must implement a trimmed prepared layout before mixing them.
    TORCH_CHECK(temporal_trim==0 || (!pre && fg<0 && fnext<0),
                "Causal prefix trim cannot use or produce a fused prepared input");
    if(pre) {
      TORCH_CHECK(prepared.defined(),"conv ",idx," is marked prepared but no tensor was stored");
      TORCH_CHECK(prepared_sharded==sharded,"prepared tensor crosses a change of sharding");
      // The graph edge checks itself. pt/ph here are the consumer's own, read from this conv's
      // weight, and the producer computed the same two from w[fnext] with fnext==idx -- so these
      // equalities are true by construction today and are a proof obligation on any later edit:
      // a wrong fnext would otherwise hand a wrong-shaped tensor to the convolution, and a
      // wrong-but-same-shaped one would pass every check the kernel makes.
      TORCH_CHECK(prepared.size(1)==x.size(1) && prepared.size(2)==x.size(2)+pt &&
                  prepared.size(3)==x.size(3)+2*ph && prepared.size(4)==x.size(4),
                  "conv ",idx,": prepared ",prepared.sizes()," does not fit input ",x.sizes(),
                  " (pt ",pt,", ph ",ph,", sharded ",(int)sharded,")");
      p=prepared; prepared=Tensor(); prepared_for=-1; prepared_sharded=false;
      y_suppressed_for=-1;   // the suppressed y was reserved for this edge and this edge took it
    }
    else if(pt||ph||gamma>=0) p=norm_pad(x,gamma>=0?w[gamma]:Tensor(),cache[idx],pt,ph,activate);
    // Memory: the unpadded input is dead once the padded copy exists (p holds it when no copy was made).
    x=Tensor();
    // The causal cache carries this convolution's input history to the next chunk. `p` may be the
    // tensor the previous fused epilogue produced and this conv has just consumed, so the copy is
    // ordered after that kernel by the stream they share -- a side stream here would race it.
    if(pt) cache[idx]=last_frames(p,pt,ph);
    if(ph) {
      RECORD_FUNCTION("wan.halo",std::vector<c10::IValue>());
      auto recv=halo_exchange(halo_pack(p),pg,rank,world);
      halo_unpack(p,recv,rank,world);
    }
    const int Ho=p.size(3)-2*ph,Wo=p.size(4);
    std::vector<int64_t> padding={0,ph?0:kh/2,kw/2};
    bool use_custom=(p.size(4)==78 && kt==3 && kh==3 && w[idx].size(0)==1024 && w[idx].size(1)==1024) || idx==61 || idx==85
                    || idx==71 || idx==93;
    // Only the expensive spatial 3x3 convolutions use Winograd. Low-resolution
    // convolutions remain direct implicit GEMMs; 1x1 and temporal-only layers
    // use cuDNN. Each transformed GEMM accumulates bf16 operands in fp32.
    int wi=-1;
    const int wids[18]={44,47,50,53,56,59,66,69,74,77,80,83,88,91,96,99,102,105};
    for(int i=0;i<18;i++)if(idx==wids[i])wi=i;
    Tensor y;
    if(wi>=0) {
      auto transformed=winograd_input_fast(p,ph);
      // Memory: the padded input is dead once transformed (its history went to the cache above), and the transform
      // (2.25x the input) once every component product is enqueued; the caching allocator reuses both in stream order.
      p=Tensor();
      std::vector<Tensor> products;
      for(int a=0;a<36;a++) {
        auto input=transformed.select(0,a).unsqueeze(0).permute({0,4,1,2,3});
        products.push_back(custom_conv(input,w[122+36*wi+a],0,0,3));
      }
      transformed=Tensor();
      // The SiLU this epilogue applies is the CONSUMER's norm activation, not this conv's
      // `activate`: fusing is valid only when the consumer would have activated. Every fused
      // consumer in the decoder (the two convs of a residual block) has activate=true, so the
      // test below holds by construction; a non-activating consumer would have to pass its own.
      bool fuse=(fg>=0&&fnext>=0&&activate);
      // write_y is false exactly where the caller asserts nobody reads this launch's y AND the
      // fused `prepared` edge exists to carry the activation instead. It is recorded so the
      // consumer can ASSERT the edge rather than trust it; see y_suppressed_for above.
      const bool write_y=!(y_dead && fuse);
      int pt2=pt,ph2=ph;
      if(fuse) { pt2=w[fnext].size(2)-1; ph2=(sharded&&world>1&&w[fnext].size(3)==3)?1:0; }
      auto out=wino_epilogue(products,Ho,Wo,add_bias?w[idx+1]:Tensor(),residual,
                             fuse?w[fg]:Tensor(),fuse?cache[fnext]:Tensor(),pt2,ph2,write_y);
      y=out.y;
      if(fuse) { prepared=out.n; prepared_for=fnext; prepared_sharded=sharded;
                 if(!write_y) y_suppressed_for=fnext; }
    } else y=use_custom?custom_conv(p,w[idx],padding[1],padding[2],3):at::cudnn_convolution(p,w[idx],padding,{1,1,1},{1,1,1},1,false,false,false);
    if(add_bias && wi<0) finish(y,w[idx+1],residual);
    return y;
  }

  Tensor resnet(Tensor x,int base,bool change=false,int nx=-1,int nxg=-1) {
    RECORD_FUNCTION("wan.resnet."+std::to_string(base),std::vector<c10::IValue>());
    Tensor skip=change?conv(x,base+6,-1,true,false):x;
    // Memory: moved into the first convolution, the block's input is released there once its norm has read it
    // (skip still holds it when it is the residual).
    auto y=conv(std::move(x),base+1,base,true,true,Tensor(),TemporalTrim{0},base+3,base+4, /*y_dead=*/true);
    return conv(y,base+4,base+3,true,true,skip,TemporalTrim{0},nxg,nx);
  }

  Tensor attention(const Tensor& x) {
    RECORD_FUNCTION("wan.attention",std::vector<c10::IValue>());
    int T=x.size(2),H=x.size(3),W=x.size(4),C=x.size(1);
    Tensor qkv=conv(x,5,4,false);
    auto p=qkv.permute({0,2,3,4,1}).view({T,1,H*W,3*C});
    Tensor y=attn_forward(p);
    y=y.view({1,T,H,W,C}).permute({0,4,1,2,3});
    return conv(y,7,-1,true,true,x);
  }

  // The output head: RMS norm + SiLU + the causal padding, the 3x1x1 convolution to 112 channels
  // (108 real) and the 3x3 stencil with the bias and clamp (head.cu). Replaces conv(x,108,107),
  // whose 12 output channels waste 244 of a 256-wide tile's columns.
  Tensor output_head(const Tensor& x) {
    RECORD_FUNCTION("wan.head",std::vector<c10::IValue>());
    const int pt=2;
    int ph=world>1?1:0;
    Tensor p=norm_pad(x,w[107],cache[108],pt,ph,true);
    if(ph) {
      RECORD_FUNCTION("wan.halo",std::vector<c10::IValue>());
      // The third halo site. PR204 deleted pack_edges/unpack_edges, and the resolution that
      // compiles-and-passes while silently dropping the vectorised pack is to re-add them:
      // this is the same two pad rows through comm.cu's halo_pack/halo_unpack, which is what
      // the other two sites (conv/resnet) already use. Census: post_edges == wait_edges == 3.
      auto recv=halo_exchange(halo_pack(p),pg,rank,world);
      halo_unpack(p,recv,rank,world);
    }
    cache[108]=last_frames(p,pt,ph);
    Tensor g=conv_n128(p,w[108]);
    int rows=x.size(3);
    auto out=at::empty({1,12,g.size(2),rows,g.size(4)},x.options().memory_format(at::MemoryFormat::ChannelsLast3d));
    // 352 is the pixel-plane height (44 latent rows x 8) and lo this rank's first pixel-plane row;
    // both are the fixed 704x1248 geometry, and lo must stay 8*(44*rank/world) because run()
    // passes lo*16 - the same row on the 704-tall video grid - to write_video.
    head_stencil(g,w[109],out,352,ph,8*(44*rank/world));
    return out;
  }

  Tensor up(Tensor input,int block,bool first) {
    RECORD_FUNCTION("wan.up_block."+std::to_string(block),std::vector<c10::IValue>());
    const int bases[4]={21,43,65,87};
    int base=bases[block];
    bool change=block>=2;
    int b2=base+(change?8:6),b3=base+(change?14:12);
    // Memory: only the resampler's DupUp3D shortcut reads the block's input again, and block 3 has none.
    auto x=resnet(block==3?std::move(input):input,base,change,b2+1,b2);
    x=resnet(x,b2,false,b3+1,b3);
    x=resnet(x,b3,false);
    if(block==3) return x;
    int spatial=base+(change?20:18);
    if(block<2) {
      // The 3x1x1 causal time convolution writes the frame-interleaved tensor directly (a stride-2 D epilogue
      // in solution/timeconv.cu), so `up_time` and `finish` are gone from this seam.  `activate=false` is
      // INERT: with no gain the dispatcher takes norm.cu's copy-only mode and never reads the flag.
      int start=first?1:0;
      Tensor head=first?x.narrow(2,0,1):Tensor();
      int nf=x.size(2)-start;
      Tensor p;
      if(nf>0) {
        p=norm_pad(x.narrow(2,start,nf),Tensor(),cache[spatial+2],2,0,false);
        cache[spatial+2]=last_frames(p,2,0);
      }
      x=time_upsample(p,head,w[spatial+2],w[spatial+3],x.size(2),nf,first);
    }
    if(world>1) {
      x=norm_pad(x,Tensor(),Tensor(),0,1,false);
      auto recv=halo_exchange(halo_pack(x),pg,rank,world);
      halo_unpack(x,recv,rank,world);
    }
    // Four-phase spatial upsampler: the four 2x2 phase convolutions, then one bandwidth-bound pass that
    // interleaves them with the bias and the DupUp3D shortcut (resample.cu).
    std::vector<Tensor> pw;
    for(int a=0;a<2;++a)for(int b=0;b<2;++b) pw.push_back(w[110+4*block+2*a+b]);
    return resample_upsample(x,pw,w[spatial+1],input,world>1,block<2,first);
  }

  // Stateless exact prefix interval. Three causal convolutions require six
  // preceding latent frames, but the support is pruned by two after each conv.
  Tensor prefix_interval(const Tensor& z,int begin,int count) {
    sharded=false;
    const int end=begin+count;
    const int lo6=std::max(0,begin-6),lo4=std::max(0,begin-4),lo2=std::max(0,begin-2);
    auto raw=z.narrow(2,lo6,end-lo6).contiguous(at::MemoryFormat::ChannelsLast3d);
    auto x=conv(raw,0);
    // Reducing left temporal padding prunes the invalid outputs without ever
    // computing them. Near global frame zero, the remaining padding is real.
    x=conv(x,2,-1,true,true,Tensor(),TemporalTrim{lo4-lo6});
    auto y=conv(x,10,9,true,true,Tensor(),TemporalTrim{lo2-lo4});
    // Batch is one and T is the outer physical NDHWC dimension, so this
    // frame slice remains dense for finish()'s linear residual indexing.
    auto skip=x.narrow(2,begin-lo4,count);
    TORCH_CHECK(skip.is_contiguous(at::MemoryFormat::ChannelsLast3d),"Prefix residual must be dense NDHWC");
    y=conv(y,13,12,true,true,skip,TemporalTrim{begin-lo2});
    return attention(y);
  }

  Tensor distributed_prefix(const Tensor& z,int start,int count) {
    // No persistent prefix history: temporal ownership changes from batch to
    // batch. The full latents on each rank provide exact predecessor context.
    Decoder prefix(w,rank,world,pg);
    Tensor x;
    if(count>=2*world) {
      const int lo=count*rank/world,hi=count*(rank+1)/world;
      x=prefix.prefix_interval(z,start+lo,hi-lo);
      x=prefix_temporal_to_rows(x,count,rank,world,pg);
    } else {
      // Tiny clips/tails do not amortize the transpose, and may have empty
      // temporal owners. A stateless replicated prefix remains exact.
      x=prefix.prefix_interval(z,start,count);
      const int lo=44*rank/world,hi=44*(rank+1)/world;
      x=x.narrow(3,lo,hi-lo).contiguous(at::MemoryFormat::ChannelsLast3d);
    }
    return x;
  }

  Tensor tail(Tensor x,bool first) {
    // Every prepared edge is consumed within its residual/up-block chain.
    // Tail history survives chunks, but no prepared current-chunk tensor may.
    TORCH_CHECK(prepared_for<0 && !prepared.defined() && y_suppressed_for<0,
                "Prepared input crossed a tail-chunk boundary");
    sharded=true;
    x=resnet(x,15);
    for(int i=0;i<4;++i) x=up(std::move(x),i,first);
    return output_head(x);
  }

  Tensor chunk(const Tensor& z,bool first) {
    sharded=false;
    // The invariant tail() checks must be CHECKED here too, not cleared. Clearing it first makes
    // tail()'s check vacuous on the only path world 1 takes, and a leaked fused tensor is not
    // merely retained: prepared_for is a conv INDEX and every chunk restarts at the same indices,
    // so the next chunk's conv of that index CONSUMES it (`pre`) and the video is silently wrong
    // at world 1 while world 4 errors out.
    //
    // The reset was dead on this tree, so the check costs one compare per chunk and nothing else,
    // but the closure argument is NOT "every fused pair lives inside one resnet()" -- that is
    // false. resnet(b)'s second conv is called with (nxg=b2, nx=b2+1) and produces for
    // fnext=nx=b2+1, which is the FIRST conv of the next resnet; the same edge appears at every
    // up block (47->50, 53->56 at block 1; 69->74, 77->80 at block 2). The chain closes at the
    // END OF THE UP BLOCK instead: up() runs resnet(base, nxg=b2, nx=b2+1), then
    // resnet(b2, nxg=b3, nx=b3+1), then resnet(b3) with the DEFAULT nx=-1, so the third resnet's
    // last conv has fg=fnext=-1, produces nothing, and no fused tensor leaves up(). up() is the
    // only Winograd caller and nothing after it is Winograd (output_head goes through conv_n128),
    // so the triple is clear at every chunk entry.
    // The predicate is tail()'s, EXACTLY, and that identity is the point rather than a
    // coincidence: chunk() must accept everything tail() accepts, or the subset property this
    // change exists to establish is not established. A third conjunct would also be unreachable --
    // all three of prepared / prepared_for / prepared_sharded are written together at the single
    // producer site (conv()'s `if(fuse)`) and cleared together at the single consumption site, so
    // `prepared_sharded` can never differ from `prepared_for` and cannot fail on its own.
    TORCH_CHECK(prepared_for<0 && !prepared.defined(),
                "Prepared input crossed a chunk boundary");
    auto x=conv(z,0);
    x=conv(x,2);
    x=resnet(x,9);
    x=attention(x);
    if(world>1) {
      int lo=44*rank/world,hi=44*(rank+1)/world;
      x=x.narrow(3,lo,hi-lo).contiguous(at::MemoryFormat::ChannelsLast3d);
    }
    return tail(x,first);
  }
};

void decode_video(const Tensor& z,const Weights& weights,int64_t rank,int64_t world,const PG& pg,Tensor out,
                  int64_t frames_per_pass) {
  at::NoGradGuard no_grad;
  c10::cuda::CUDAGuard device_guard(z.device());
  TORCH_CHECK(z.scalar_type()==at::kBFloat16&&z.dim()==5&&z.size(0)==1&&z.size(1)==48&&z.size(2)>=1&&z.size(3)==44&&z.size(4)==78,"Expected bf16 [1,48,T,44,78]");
  TORCH_CHECK(world>=1&&world<=8&&rank>=0&&rank<world,"Invalid rank/world");
  TORCH_CHECK(weights.size()==770,"Call prepare() once before run()");
  TORCH_CHECK(frames_per_pass>=1,"frames_per_pass must be at least 1, not ",frames_per_pass);
  halo_stats()=HaloStats{};  // per call: the debug tail below prints THIS run's exchange, not the process's
  Decoder decoder(weights,rank,world,pg);
  // A pass carries frames_per_pass latent frames per GPU, frames_per_pass * world in all. The working memory of a GPU
  // grows with it (3.5-4.2 GiB per latent frame at 704 x 1248), and so does the speed: a pass's GEMMs at 44 x 78 and
  // 88 x 156 fill the GPU only with enough frames, and above world 1 every pass also pays a fixed cost (its prefix
  // and its halo exchanges). The causal history crosses passes in `cache`, so the video does not depend on the size.
  // At most 61 latent frames: the largest pass this tree was measured with (wino_epilogue.cu's indices are 64-bit).
  static const int64_t kMaxChunk=61;
  const int chunk_size=(int)std::min<int64_t>(frames_per_pass*world,kMaxChunk);
  int T=z.size(2);
  for(int start=0;start<T;start+=chunk_size) {
    int count=std::min((int)chunk_size,T-start);
    Tensor pixels;
    if(world>1) pixels=decoder.tail(decoder.distributed_prefix(z,start,count),start==0);
    else {
      auto input=z.narrow(2,start,count).contiguous(at::MemoryFormat::ChannelsLast3d);
      pixels=decoder.chunk(input,start==0);
    }
    int frame_offset=start==0?0:4*start-3;
    if(world==1) write_video(pixels,out,frame_offset,0);
    else if(rank==0) {
      write_video(pixels,out,frame_offset,0);
      for(int peer=1;peer<world;++peer) {
        int lo=44*peer/world,hi=44*(peer+1)/world;
        auto part=nhwc_empty(12,pixels.size(2),(hi-lo)*8,624,pixels);
        std::vector<Tensor> buffers={part};
        pg->recv(buffers,peer,0)->wait();
        write_video(part,out,frame_offset,lo*16);
      }
    } else {
      std::vector<Tensor> buffers={pixels};
      pg->send(buffers,0,0)->wait();
    }
  }
  // Debug only, and never set by CI: cached once per process so the measured span runs no getenv.
  static const bool show_halo_stats=std::getenv("WAN_HALO_STATS")!=nullptr;
  if(show_halo_stats) {
    const HaloStats& s=halo_stats();
    printf("[halo rank %d/%d] exchanges %lld  sent %.1f MB  received %.1f MB\n",(int)rank,(int)world,
           (long long)s.exchanges,1e-6*s.bytes_sent,1e-6*s.bytes_received);
    fflush(stdout);
  }
}

// The default pass: 4 latent frames per GPU. On one GB200 at 704 x 1248 it peaks at 21-22 GiB allocated and takes
// 1.09x the time of 16 frames per GPU, which peaks at 60-61 GiB (8 to 20 s of video).
constexpr int64_t kFramesPerPass=4;

void run(const Tensor& z,const Weights& weights,int64_t rank,int64_t world,const PG& pg,Tensor out) {
  decode_video(z,weights,rank,world,pg,out,kFramesPerPass);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("prepare",&prepare);
  m.def("run",&run,py::call_guard<py::gil_scoped_release>());
  m.def("run_with",&decode_video,py::call_guard<py::gil_scoped_release>());
}
