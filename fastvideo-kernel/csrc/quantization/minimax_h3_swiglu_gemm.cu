// SPDX-License-Identifier: Apache-2.0
// Experimental full-K SM120 GEMM with value/gate-paired columns and FP4 output.
#include "minimax_h3_swiglu_quant.cuh"
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/util/packed_stride.hpp>

namespace h3_epilogue {
using namespace cute;
using namespace cutlass::epilogue::fusion;
struct Operation : LinearCombination<cutlass::bfloat16_t, float, void> {
  using ElementAux = cutlass::bfloat16_t;
  using GmemLayoutTagAux = cutlass::layout::RowMajor;
  static constexpr int AlignmentAux = 8;
};

template<class CtaTile, class EpiTile>
struct Store {
  static constexpr int EM = size<0>(EpiTile{});
  static constexpr int EN = size<1>(EpiTile{});
  static_assert(EN % 32 == 0);
  struct SharedStorage { cutlass::array_aligned<__nv_bfloat16, EM * EN / 2> values; };
  struct Arguments {
    const float* alpha_ptr = nullptr;
    const float* scale_ptr = nullptr;
    uint8_t* q = nullptr;
    uint8_t* scales = nullptr;
    float* amax = nullptr;
    __nv_bfloat16* y = nullptr;
  };
  using Params = Arguments;
  Params const* params;
  __nv_bfloat16* shared;
  CUTLASS_HOST_DEVICE Store() = default;
  CUTLASS_HOST_DEVICE Store(Params const& p, SharedStorage const& s)
      : params(&p), shared(const_cast<__nv_bfloat16*>(s.values.data())) {}
  template<class Shape>
  static Params to_underlying_arguments(Shape const&, Arguments const& a, void*) { return a; }
  template<class Shape> static bool can_implement(Shape const& s, Arguments const&) {
    return get<1>(s) % 128 == 0;
  }
  template<class Shape> static size_t get_workspace_size(Shape const&, Arguments const&) { return 0; }
  template<class Shape>
  static cutlass::Status initialize_workspace(Shape const&, Arguments const&, void*, cudaStream_t,
                                               cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }
  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }
  template<class... A> CUTLASS_DEVICE auto get_producer_load_callbacks(ProducerLoadArgs<A...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class Coords, int Threads>
  struct Consumer : EmptyConsumerStoreCallbacks {
    Coords coords;
    Params p;
    __nv_bfloat16* shared;
    int tid, rows, width, row_base, col_base;
    float alpha, global_sf;
    CUTLASS_DEVICE Consumer(Coords c, Params p_, __nv_bfloat16* sm, int t, int m, int n, int rb, int cb)
        : coords(c), p(p_), shared(sm), tid(t), rows(m), width(n/2), row_base(rb), col_base(cb),
          alpha(*p_.alpha_ptr), global_sf(*p_.scale_ptr) {}
    template<class Acc, int F>
    CUTLASS_DEVICE auto visit(cutlass::Array<Acc, F> const& acc, int v, int em, int en) {
      static_assert(F % 2 == 0);
      auto c = coords(_,_,_,em,en);
      CUTLASS_PRAGMA_UNROLL
      for (int i=0; i<F; i+=2) {
        auto xy = c(v*F+i);
        const int row = get<0>(xy), col = get<1>(xy);
        const float value = __bfloat162float(__float2bfloat16_rn(__fmul_rn(alpha, float(acc[i]))));
        const float gate = __bfloat162float(__float2bfloat16_rn(__fmul_rn(alpha, float(acc[i+1]))));
        const float y = row < rows ? swiglu_bf16(value,gate) : 0;
        shared[(row % EM) * (EN/2) + (col % EN)/2] = __float2bfloat16_rn(y);
      }
      return acc;
    }
    template<class Buffer, class Sync, class Values>
    CUTLASS_DEVICE void reduce(Buffer&&, Sync const& sync, int em, int en, bool, Values) {
      sync();
      CUTLASS_PRAGMA_UNROLL
      for (int i=tid; i<EM*EN/2; i+=Threads) {
        const int row = row_base + em*EM + i/(EN/2);
        const int col = col_base/2 + en*(EN/2) + i%(EN/2);
        const float y = __bfloat162float(shared[i]);
        float amax = fabsf(y);
        CUTLASS_PRAGMA_UNROLL
        for (int d=1; d<16; d*=2)
          amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,d,16));
        const uint8_t sf=scale_fp8(__fmul_rn(global_sf,__fmul_rn(amax,rcp_approx(6.0f))));
        const float inv=amax!=0 ? rcp_approx(__fmul_rn(scale_float(sf),rcp_approx(global_sf))) : 0;
        const float code=__fmul_rn(y,inv);
        const float hi=__shfl_xor_sync(0xffffffff,code,1);
        if (row<rows && col<width) {
          if ((col&1)==0) p.q[int64_t(row)*(width/2)+col/2]=fp4_pair(code,hi);
          if (p.y) p.y[int64_t(row)*width+col]=__float2bfloat16_rn(y);
        }
        if (row < (rows + 127) / 128 * 128 && col<width && (col&15)==0) {
          const int group=col/16;
          const int64_t idx=(((((int64_t(row)/128)*(width/64)+group/4)*32+row%32)*4+(row/32)%4)*4+group%4);
          p.scales[idx]=sf;
          if (p.amax) p.amax[int64_t(row)*(width/16)+group]=amax;
        }
      }
      sync();
    }
  };
  template<bool ReferenceSrc, class... A>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(ConsumerStoreArgs<A...> const& args) {
    auto [M,N,K,L]=args.problem_shape_mnkl;
    auto [m,n,k,l]=args.tile_coord_mnkl;
    auto identity=make_identity_tensor(make_shape(M,N));
    auto tile=local_tile(identity,take<0,2>(args.tile_shape_mnk),make_coord(m,n));
    auto coords=sm90_partition_for_epilogue<ReferenceSrc>(tile,args.epi_tile,args.tiled_copy,args.thread_idx);
    constexpr int threads=decltype(size(args.tiled_copy))::value;
    return Consumer<decltype(coords),threads>(coords,*params,shared,args.thread_idx,M,N,
                                              m*size<0>(CtaTile{}),n*size<1>(CtaTile{}));
  }
};
} // namespace h3_epilogue

namespace cutlass::epilogue::fusion {
template<int SC,int SD,int F,bool R,bool D,class CT,class ET,class... Extra>
struct FusionCallbacks<cutlass::epilogue::Sm90TmaWarpSpecialized<SC,SD,F,R,D>,
                       h3_epilogue::Operation,CT,ET,Extra...> : h3_epilogue::Store<CT,ET> {
  using Impl=h3_epilogue::Store<CT,ET>;
  using Impl::Impl;
  using Operation=h3_epilogue::Operation;
};
}

namespace h3_epilogue {
using Shape=cute::Shape<int,int,int,int>;
#ifndef H3_TILE_M
#define H3_TILE_M 128
#define H3_TILE_N 128
#define H3_TILE_K 256
#endif
#ifndef H3_PINGPONG
#define H3_PINGPONG 0
#endif
#ifndef H3_STAGES
#define H3_STAGES 0
#endif
using Tile=cute::Shape<cute::Int<H3_TILE_M>,cute::Int<H3_TILE_N>,cute::Int<H3_TILE_K>>;
using Cluster=cute::Shape<cute::_1,cute::_1,cute::_1>;
using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm120,cutlass::arch::OpClassTensorOp,Tile,Cluster,
  cutlass::epilogue::collective::EpilogueTileAuto,float,float,
  void,cutlass::layout::RowMajor,8,void,cutlass::layout::RowMajor,8,
  cutlass::epilogue::TmaWarpSpecialized,Operation>::CollectiveOp;
using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm120,cutlass::arch::OpClassBlockScaledTensorOp,
  cutlass::nv_float4_t<cutlass::float_e2m1_t>,cutlass::layout::RowMajor,32,
  cutlass::nv_float4_t<cutlass::float_e2m1_t>,cutlass::layout::ColumnMajor,32,
  float,Tile,Cluster,
  cute::conditional_t<H3_STAGES == 0,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
    cutlass::gemm::collective::StageCount<H3_STAGES>>, 
  cute::conditional_t<H3_PINGPONG != 0, cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                      cutlass::gemm::KernelTmaWarpSpecializedCooperative>>::CollectiveOp;
using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape,Mainloop,Epilogue,cutlass::gemm::StaticPersistentScheduler>;
using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
}
extern "C" int launch_gemm_swiglu(const void* a,const void* b,const void* sa,const void* sb,
                                 const float* alpha,const float* sf,void* q,void* scales,
                                 float* amax,void* y,int m,int n,int k,void* workspace,
                                 uint64_t workspace_bytes,int raster,int swizzle,void* raw_stream) {
  using namespace h3_epilogue;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  Gemm::Arguments args{};
  args.mode=cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape=cute::make_shape(m,n,k,1);
  args.mainloop.ptr_A=static_cast<const cutlass::float_e2m1_t*>(a);
  args.mainloop.ptr_B=static_cast<const cutlass::float_e2m1_t*>(b);
  args.mainloop.ptr_SFA=static_cast<const cutlass::float_ue4m3_t*>(sa);
  args.mainloop.ptr_SFB=static_cast<const cutlass::float_ue4m3_t*>(sb);
  args.mainloop.dA=cutlass::make_cute_packed_stride(Kernel::StrideA{}, {m,k,1});
  args.mainloop.dB=cutlass::make_cute_packed_stride(Kernel::StrideB{}, {n,k,1});
  using Config=Mainloop::Sm1xxBlkScaledConfig;
  args.mainloop.layout_SFA=Config::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB=Config::tile_atom_to_shape_SFB(args.problem_shape);
  args.epilogue.dC=cutlass::make_cute_packed_stride(Kernel::StrideC{}, {m,n,1});
  args.epilogue.dD=args.epilogue.dC;
  args.epilogue.thread={alpha,sf,static_cast<uint8_t*>(q),static_cast<uint8_t*>(scales),amax,
                       static_cast<__nv_bfloat16*>(y)};
  args.scheduler.max_swizzle_size=swizzle;
  args.scheduler.raster_order=static_cast<decltype(args.scheduler.raster_order)>(raster);
  args.hw_info.cluster_shape=dim3(1,1,1);
  args.hw_info.cluster_shape_fallback=dim3(1,1,1);
  Gemm gemm;
  if (gemm.get_workspace_size(args)>workspace_bytes) return -1;
  auto status=gemm.can_implement(args);
  if (status!=cutlass::Status::kSuccess) return 100+int(status);
  status=gemm.initialize(args,workspace,stream);
  if (status!=cutlass::Status::kSuccess) return 200+int(status);
  status=gemm.run(stream);
  return status==cutlass::Status::kSuccess ? int(cudaGetLastError()) : 300+int(status);
}
