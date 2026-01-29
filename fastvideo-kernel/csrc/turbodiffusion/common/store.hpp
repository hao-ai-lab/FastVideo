#pragma once
#include "common/common.hpp"


template <
  class OutputDtype_,
  int TileM_,
  int TileN_,
  int NumThrPerCta_,
  bool IsEvenM,
  bool IsEvenN,
  bool Round = true,
  bool SaveScale = true
>
class Saver {
public:
  using OutputDtype = OutputDtype_;

  static constexpr int TileM = TileM_;
  static constexpr int TileN = TileN_;
  static constexpr int NumThrPerCta = NumThrPerCta_;
  static constexpr int NumElementPerThread = TileM * TileN / NumThrPerCta;
  static constexpr int NumThrPerRow = TileN / NumElementPerThread;

  static_assert(TileM * TileN % NumThrPerCta == 0);
  static_assert(NumThrPerCta % TileM == 0);

  CUTLASS_DEVICE void
  store(void *Optr, void *OSptr, void *reg, float scale_inv, int64_t m, int64_t n, int blk_m, int blk_n, int tid) {
    int n_alignment = (n & 31) * sizeof(OutputDtype);
    int thr_m_offset = tid / NumThrPerRow;
    int thr_n_offset = (tid % NumThrPerRow) * NumElementPerThread;
    void *cta_output_ptr = (void*)((OutputDtype*)Optr + blk_m * TileM * (Round ? cdiv(n, TileN) * TileN : n) + blk_n * TileN);
    void *thr_output_ptr = (void*)((OutputDtype*)cta_output_ptr + thr_m_offset * (Round ? cdiv(n, TileN) * TileN : n) + thr_n_offset);
    bool pred = IsEvenM ? true : thr_m_offset + blk_m * TileM < m;
    int limit = IsEvenN ? NumElementPerThread : MIN(NumElementPerThread, n - (blk_n * TileN + thr_n_offset));
    if (n_alignment % 128 == 0)
      _store<int4, IsEvenN>(thr_output_ptr, reg, limit, pred);
    else if (n_alignment % 64 == 0)
      _store<int2, IsEvenN>(thr_output_ptr, reg, limit, pred);
    else
      _store<OutputDtype, IsEvenN>(thr_output_ptr, reg, limit, pred);

    if constexpr (SaveScale) {
      if (tid == 0) {
        *((float*)OSptr + blk_m * cdiv(n, TileN)+ blk_n) = scale_inv;
      }
    }
  }

private:
  template <class StoreDataType, bool IsEven>
  CUTLASS_DEVICE void
  _store(void *thr_output_ptr, void *reg, int limit, bool pred) {
    static constexpr int NumElementPerStore = sizeof(StoreDataType) / sizeof(OutputDtype);
    if (pred) {
      if constexpr (IsEven) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; i += NumElementPerStore) {
          *(StoreDataType*)((OutputDtype*)thr_output_ptr + i) = *(StoreDataType*)((OutputDtype*)reg + i);
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < limit; i += NumElementPerStore) {
          if (limit - i > NumElementPerStore)
            *(StoreDataType*)((OutputDtype*)thr_output_ptr + i) = *(StoreDataType*)((OutputDtype*)reg + i);
          else {
            for (int j = 0; j < limit - i; ++j) {
              *((OutputDtype*)thr_output_ptr + i + j) = *((OutputDtype*)reg + i + j);
            }
          }
        }
      }
    }
  }

};
