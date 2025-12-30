/* 
 * Copyright (c) 2025 by TurboDiffusion team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * 
 * Citation (please cite if you use this code):
 * 
 * @article{zhang2025turbodiffusion,
 *   title={TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times},
 *   author={Zhang, Jintao and Zheng, Kaiwen and Jiang, Kai and Wang, Haoxu and Stoica, Ion and Gonzalez, Joseph E and Chen, Jianfei and Zhu, Jun},
 *   journal={arXiv preprint arXiv:2512.16093},
 *   year={2025}
 * }
 */

#pragma once

#include <cuda.h>
#include "cute/tensor.hpp"

#include "common/common.hpp"
#include "gemm/utils.hpp"

using namespace cute;

template <
  class OutputDtype_,
  bool IsEvenM,
  bool IsEvenN
>
struct GemmKernel {
  using ElementA = int8_t;
  using ElementB = int8_t;
  using OutputDtype = OutputDtype_;
  using AccumulatorDtype = int32_t;
  static constexpr int BlockSize = 128;
  static constexpr int TileM = 128;
  static constexpr int TileN = 128;
  static constexpr int TileK = 128;
  static constexpr int Stage = 3;
  static constexpr int EpiStage = 2;
  
  static_assert(
       BlockSize % TileM == 0
    && BlockSize % TileN == 0
    && BlockSize % TileK == 0
  );

  static constexpr int NumTilePerBlock = BlockSize / TileK;

  using SmemLayoutAtom = decltype(
    composition(
      Swizzle<3, 4, 3>{},
      make_layout(
        make_shape(Int<8>{}, Int<TileK>{}),
        make_stride(Int<TileK>{}, Int<1>{})
      )
    )
  );

  using SmemLayoutA = decltype(
    tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<TileM>{}, Int<TileK>{}, Int<Stage>{})
    )
  );

  using SmemLayoutB = decltype(
    tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<TileN>{}, Int<TileK>{}, Int<Stage>{})
    )
  );

  using MmaOP = cute::SM80_16x8x32_S32S8S8S32_TN;
  using TiledMma = decltype(
    make_tiled_mma(
      MMA_Atom<MMA_Traits<MmaOP>>{},
      make_layout(make_shape(
        _4{}, _2{}, _1{}
      )),
      make_tile(Int<64>{}, Int<32>{}, Int<32>{})
    )
  );

  using G2SCopyAtomA = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>>, ElementA>;
  using G2SCopyAtomB = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>>, ElementB>;
  using G2STiledCopyA = decltype(
    make_tiled_copy(
      G2SCopyAtomA{},
      make_layout(
        make_shape(Int<64>{}, Int<4>{}),
        make_stride(Int<4>{}, Int<1>{})
      ),
      make_layout(make_shape(Int<1>{}, Int<16>{}))
    )
  );
  using G2STiledCopyB = decltype(
    make_tiled_copy(
      G2SCopyAtomB{},
      make_layout(
        make_shape(Int<64>{}, Int<4>{}),
        make_stride(Int<4>{}, Int<1>{})
      ),
      make_layout(make_shape(Int<1>{}, Int<16>{}))
    )
  );

  using S2RCopyAtomA = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, ElementA>;
  using S2RCopyAtomB = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, ElementB>;
  using S2RTiledCopyA = decltype(make_tiled_copy_A(S2RCopyAtomA{}, TiledMma{}));
  using S2RTiledCopyB = decltype(make_tiled_copy_B(S2RCopyAtomB{}, TiledMma{}));

  // epilogue
  using SmemLayoutAtomD = decltype(
    composition(
      Swizzle<2, 3, 3>{},
      make_layout(
        make_shape(Int<32>{}, Int<32>{}),
        LayoutRight{}
      )
    )
  );

  using SmemLayoutD = decltype(
    tile_to_shape(
      SmemLayoutAtomD{},
      make_shape(Int<64>{}, Int<32>{}, Int<EpiStage>{})
    )
  );

  using R2SCopyAtomD = Copy_Atom<UniversalCopy<std::conditional_t<sizeof(OutputDtype) == 4, int32_t, int16_t>>, OutputDtype>;
  using R2STiledCopyD = decltype(make_tiled_copy_C(R2SCopyAtomD{}, TiledMma{}));

  using S2GCopyAtomD = Copy_Atom<UniversalCopy<uint128_t>, OutputDtype>;
  using S2GCopyD = decltype(make_tiled_copy(
    S2GCopyAtomD{},
    make_layout(Shape<_64, _4>{}),
    make_layout(Shape<_1, _8>{})
  ));

  using TileShape = decltype(make_shape(Int<TileM>{}, Int<TileN>{}, Int<TileK>{}));

  struct SharedStorageAB: cute::aligned_struct<128> {
    array_aligned<typename TiledMma::ValTypeA, cosize_v<SmemLayoutA>, 128> smem_A;
    array_aligned<typename TiledMma::ValTypeB, cosize_v<SmemLayoutB>, 128> smem_B;
    array_aligned<float, 1> smem_AS;
    array_aligned<float, 1> smem_BS;
    array_aligned<int32_t, 1> smem_AF;
  };

  struct SharedStorageD: cute::aligned_struct<128> {
    array_aligned<OutputDtype, cosize_v<SmemLayoutD>> smem_D;
  };

  union SharedStorage {
    SharedStorageAB storage_AB;
    SharedStorageD storage_D;
  };


  struct Params {
    void const* Aptr;
    void const* ASptr;
    void const* Bptr;
    void const* BSptr;
    void* Dptr;
    int64_t const m;
    int64_t const n;
    int64_t const k;
    int const swizzle_dir;
    int const swizzle_size;
  };

  using Arguments = Params;

  static constexpr int ThreadNum = size(TiledMma{});
  static constexpr int ShmSize = sizeof(SharedStorage);
  static constexpr bool FastInt2Float = false;

  static bool can_implement(int64_t m, int64_t n, int64_t k) {
    if (k % BlockSize != 0) return false;
    if ((n * sizeof(OutputDtype)) % 16 != 0)
      return false;
    return true;
  }

  static Params to_underlying_arguments(Arguments const& args) {
    return args;
  }

  static dim3 get_grid_size(int64_t m, int64_t n) {
    return dim3(cdiv(m, TileM) * cdiv(n, TileN));
  }

  CUTLASS_HOST_DEVICE 
  static auto get_block_coord(
    int64_t m_blocks, 
    int64_t n_blocks, 
    int const swizzle_dir,
    int64_t const swizzle_size_log
  ) {
    int64_t blk_m;
    int64_t blk_n;

    if (swizzle_dir == 1)
      std::swap(m_blocks, n_blocks);

    if (swizzle_size_log == 0) {
      blk_m = blockIdx.x % m_blocks;
      blk_n = blockIdx.x / m_blocks;
    } else {
      int64_t group_size = n_blocks << swizzle_size_log;
      int64_t num_groups = m_blocks >> swizzle_size_log;
      int64_t group_idx = blockIdx.x / group_size;
      int64_t local_idx = blockIdx.x % group_size;
      if (group_idx == num_groups) {
        blk_m = (num_groups << swizzle_size_log) + local_idx % (m_blocks - (num_groups << swizzle_size_log));
        blk_n = local_idx / (m_blocks - (num_groups << swizzle_size_log));
      } else {
        blk_m = (local_idx & ((1LL << swizzle_size_log) - 1)) + (group_idx << swizzle_size_log);
        blk_n = local_idx >> swizzle_size_log;
      }
    }

    if (swizzle_dir == 1)
      std::swap(blk_m, blk_n);

    return make_coord(blk_m, blk_n);
  }

  CUTLASS_DEVICE
  void operator()(
    Params const& params, char* smem_data
  ) {

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_data);

    auto t_idx = threadIdx.x;

    int64_t const m = params.m;
    int64_t const n = params.n;
    int64_t const k = params.k;
    int const swizzle_dir = params.swizzle_dir;
    int const swizzle_size = params.swizzle_size;

    Tensor A = make_tensor(
      make_gmem_ptr<ElementA>(params.Aptr),
      make_shape(m, k),
      make_stride(k, _1{})
    );
    Tensor B = make_tensor(
      make_gmem_ptr<ElementB>(params.Bptr),
      make_shape(m, k),
      make_stride(k, _1{})
    );
    Tensor AS = make_tensor(
      make_gmem_ptr<float>(params.ASptr),
      make_shape(cdiv(m, BlockSize), cdiv(k, BlockSize)),
      make_stride(cdiv(k, BlockSize), _1{})
    );
    Tensor BS = make_tensor(
      make_gmem_ptr<float>(params.BSptr),
      make_shape(cdiv(n, BlockSize), cdiv(k, BlockSize)),
      make_stride(cdiv(k, BlockSize), _1{})
    );
    Tensor D = make_tensor(
      make_gmem_ptr<OutputDtype>(params.Dptr),
      make_shape(m, n),
      LayoutRight{}
    );

    auto [m_coord, n_coord] = get_block_coord(
      cdiv(m, size<0>(TileShape{})),
      cdiv(n, size<1>(TileShape{})),
      swizzle_dir, swizzle_size
    );

    int32_t blk_m_coord = m_coord / (BlockSize / TileM);
    int32_t blk_n_coord = n_coord / (BlockSize / TileN);

    // local tile
    auto gA = local_tile(A, TileShape{}, make_coord(m_coord, n_coord, _), Step<_1, X, _1>{});
    auto gB = local_tile(B, TileShape{}, make_coord(m_coord, n_coord, _), Step<X, _1, _1>{});
    auto gD = local_tile(D, TileShape{}, make_coord(m_coord, n_coord, _), Step<_1, _1, X>{});

    // shared memory
    Tensor sA = make_tensor(
      make_smem_ptr<ElementA>(shared_storage.storage_AB.smem_A.data()),
      SmemLayoutA{}
    );

    Tensor sB = make_tensor(
      make_smem_ptr<ElementB>(shared_storage.storage_AB.smem_B.data()),
      SmemLayoutB{}
    );

    // register
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(t_idx);
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tDrC = thr_mma.partition_fragment_C(gD); // mma accumulator
    auto tDrD = make_tensor_like<float>(tDrC); // float accumulator

    if constexpr (FastInt2Float) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tDrC); ++i)
        tDrC(i) = 0x4B400000;
    } else {
      clear(tDrC);
    }

    clear(tDrD);


    // global to shared copy
    G2STiledCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(t_idx);
    auto tAgA = g2s_thr_copy_a.partition_S(gA);
    auto tAsA = g2s_thr_copy_a.partition_D(sA);
    auto cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
    auto tAcA = g2s_thr_copy_a.partition_S(cA);
    int const m_limit = m - TileM * m_coord;
    int const n_limit = n - TileN * n_coord;

    G2STiledCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(t_idx);
    auto tBgB = g2s_thr_copy_b.partition_S(gB);
    auto tBsB = g2s_thr_copy_b.partition_D(sB);
    auto cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));
    auto tBcB = g2s_thr_copy_a.partition_S(cB);


    // shared to register copy
    S2RTiledCopyA s2r_tiled_copy_a;
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(t_idx);
    auto tCsA = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

    S2RTiledCopyB s2r_tiled_copy_b;
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(t_idx);
    auto tCsB = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    // pipeline status
    int64_t g2s_a_tile = 0;
    int64_t g2s_b_tile = 0;
    int g2s_a_smem = 0;
    int g2s_b_smem = 0;

    int g2s_tile_in_block = 0;
    int g2s_block = 0; // b block idx

    int s2r_a_smem = 0;
    int s2r_b_smem = 0;
    int s2r_tile_in_block = 0;

    int mma_block_a = 0;
    int mma_block_b = 0;

    int ntile = k / TileK;
    // load scale and fallback
    // we assume all ptrs are 128bit aligned
    // auto smem_fallback_A = raw_pointer_cast(make_smem_ptr<int32_t>(shared_storage.storage_AB.smem_AF.data()));
    // auto smem_scale_A = raw_pointer_cast(make_smem_ptr<float>(shared_storage.storage_AB.smem_AS.data()));
    // auto smem_scale_B = raw_pointer_cast(make_smem_ptr<float>(shared_storage.storage_AB.smem_BS.data()));
    __syncthreads();


    int32_t fallbackA_load = 0;
    int32_t fallbackA_mma = 0;

    // copy first Stage - 1 tile
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0, _i = min(Stage - 1, ntile); i < _i; ++i) {
      if (g2s_b_tile < ntile) {
        g2s_tile_in_block = (g2s_tile_in_block + 1) % NumTilePerBlock;
        copy_AB<IsEvenM>(g2s_tiled_copy_a, tAgA, tAsA, tAcA, g2s_a_tile, g2s_a_smem, m_limit);
        copy_AB<IsEvenN>(g2s_tiled_copy_b, tBgB, tBsB, tBcB, g2s_b_tile, g2s_b_smem, n_limit);
        ++g2s_b_tile;
        ++g2s_b_smem;
        ++g2s_block;
        g2s_a_tile = g2s_block * NumTilePerBlock;
        ++g2s_a_smem;
      }
      cp_async_fence();
    }

    constexpr int nk = size<2>(tCrA);
    float scale_a = AS(blk_m_coord, 0);
    float scale_b = BS(blk_n_coord, 0);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int64_t mma_b_tile = 0; mma_b_tile < ntile; ++mma_b_tile) {
      s2r_tile_in_block = (s2r_tile_in_block + 1) % NumTilePerBlock;
      cp_async_wait<Stage - 2>();
      __syncthreads();

      // do mma first
      CUTLASS_PRAGMA_UNROLL
      for (int ik = 0; ik < nk; ++ik) {
        cute::copy(s2r_tiled_copy_a, tCsA(_, _, ik, s2r_a_smem),
                    tCrA_view(_, _, ik));
        cute::copy(s2r_tiled_copy_b, tCsB(_, _, ik, s2r_b_smem),
                    tCrB_view(_, _, ik));
        cute::gemm(tiled_mma, tDrC, tCrA(_, _, ik), tCrB(_, _, ik), tDrC);
      }

      // a s2r increase anyway
      s2r_a_smem = (s2r_a_smem + 1) % Stage;

      // get next s2r b tile int64_t
        // end of a block

        // dequant first
      dequant<AccumulatorDtype, TileM * TileN / ThreadNum, FastInt2Float>(
        tDrC.data(), tDrD.data(), scale_a * scale_b
      );

      s2r_b_smem = (s2r_b_smem + 1) % Stage;
      // b advance
      ++mma_block_b;
      if (mma_block_b < size<1>(BS)) scale_b = BS(blk_n_coord, mma_block_b);
      mma_block_a = mma_block_b;
      if (mma_block_a < size<1>(AS)) scale_a = AS(blk_m_coord, mma_block_a);

      // load next stage
      if (g2s_b_tile < ntile) {
        g2s_tile_in_block = (g2s_tile_in_block + 1) % NumTilePerBlock;
        copy_AB<IsEvenM>(g2s_tiled_copy_a, tAgA, tAsA, tAcA, g2s_a_tile, g2s_a_smem, m_limit);
        copy_AB<IsEvenN>(g2s_tiled_copy_b, tBgB, tBsB, tBcB, g2s_b_tile, g2s_b_smem, n_limit);
        ++g2s_b_tile;
        g2s_b_smem = (g2s_b_smem + 1) % Stage;
        ++g2s_block;
        g2s_a_tile = g2s_block * NumTilePerBlock;
        g2s_a_smem = (g2s_a_smem + 1) % Stage;
      }
      cp_async_fence();
    }


    // epilogue

    Tensor sD = make_tensor(
      make_smem_ptr<OutputDtype>(shared_storage.storage_D.smem_D.data()),
      SmemLayoutD{}
    );

    R2STiledCopyD r2s_tiled_copy_d;
    auto r2s_thr_copy_d = r2s_tiled_copy_d.get_slice(t_idx);
    auto tDrD_r2s = r2s_thr_copy_d.retile_S(tDrD);
    auto tDsD_r2s = r2s_thr_copy_d.partition_D(sD);

    S2GCopyD s2g_tiled_copy_d;
    auto s2g_thr_copy_d = s2g_tiled_copy_d.get_slice(t_idx);
    auto tDsD_s2g = s2g_thr_copy_d.partition_S(sD);
    auto tDgD_s2g = s2g_thr_copy_d.partition_D(gD);
    Tensor cD = make_identity_tensor(make_shape(Int<TileM>{}, Int<TileN>{}));
    auto tDcD_s2g = s2g_thr_copy_d.partition_D(cD);

    auto tDgD_s2gx = group_modes<1, 3>(tDgD_s2g);  // (CPY_, CPY_MN)
    auto tDrD_r2sx = group_modes<1, 3>(tDrD_r2s);  // (CPY_, CPY_MN)
    auto tDcD_s2gx = group_modes<1, 3>(tDcD_s2g);
    
    int32_t step = size<3>(tDsD_r2s);  // pipe
    CUTLASS_PRAGMA_UNROLL
    for (int32_t i = 0; i < size<1>(tDrD_r2sx); i += step) {
      CUTLASS_PRAGMA_UNROLL
      for (int32_t j = 0; j < step; ++j) {
        if constexpr (std::is_same<OutputDtype, float>::value) {
            cute::copy(r2s_tiled_copy_d, tDrD_r2sx(_, i + j), tDsD_r2s(_, 0, 0, j));
        } else {
            auto t = make_tensor_like<OutputDtype>(tDrD_r2sx(_, i + j));
            cute::copy(tDrD_r2sx(_, i + j), t);
            cute::copy(r2s_tiled_copy_d, t, tDsD_r2s(_, 0, 0, j));
        }
      }

      __syncthreads();

      // shm -> global
      if constexpr (IsEvenM && IsEvenN) {
        CUTLASS_PRAGMA_UNROLL
        for (int32_t j = 0; j < step; ++j)
          cute::copy(s2g_tiled_copy_d, tDsD_s2g(_, 0, 0, j), tDgD_s2gx(_, i + j));
      } else if constexpr (IsEvenN) {
          CUTLASS_PRAGMA_UNROLL
          for (int32_t j = 0; j < step; ++j) {
            if (get<0>(tDcD_s2gx(0, i + j)) < m_limit)
              cute::copy(s2g_tiled_copy_d, tDsD_s2g(_, 0, 0, j), tDgD_s2gx(_, i + j));
          }
      } else if constexpr (IsEvenM) {
          CUTLASS_PRAGMA_UNROLL
          for (int32_t j = 0; j < step; ++j)
            if (get<1>(tDcD_s2gx(size<0>(tDsD_s2g) - 1, i + j)) < n_limit) {
              cute::copy(s2g_tiled_copy_d, tDsD_s2g(_, 0, 0, j), tDgD_s2gx(_, i + j));
            } else {
              CUTLASS_PRAGMA_UNROLL
              for (int k = 0; k < size<0>(tDsD_s2g); ++k)
                if (get<1>(tDcD_s2gx(k, i + j)) < n_limit)
                  tDgD_s2gx(k, i + j) = tDsD_s2g(k, 0, 0, j);
            }
      } else {
          CUTLASS_PRAGMA_UNROLL
          for (int32_t j = 0; j < step; ++j)
            if (get<0>(tDcD_s2gx(0, i + j)) < m_limit) {
              if (get<1>(tDcD_s2gx(size<0>(tDsD_s2g) - 1, i + j)) < n_limit) {
                cute::copy(s2g_tiled_copy_d, tDsD_s2g(_, 0, 0, j), tDgD_s2gx(_, i + j));
              } else {
                for (int32_t k = 0; k < size<0>(tDsD_s2g); ++k)
                  if (get<1>(tDcD_s2gx(k, i + j)) < n_limit)
                    tDgD_s2gx(k, i + j) = tDsD_s2g(k, 0, 0, j);
              }
            }
      }
      __syncthreads();
    }
  }

};
  