#pragma once

template <
  class InputDtype_,
  int TileM_,
  int TileN_,
  int NumThrPerCta_,
  bool IsEvenM,
  bool IsEvenN
>
class Loader {
public: 
  using InputDtype = InputDtype_;
  static constexpr int TileM = TileM_;
  static constexpr int TileN = TileN_;
  static constexpr int NumThrPerCta = NumThrPerCta_;
  static constexpr int NumElementPerThread = TileM * TileN / NumThrPerCta;
  static constexpr int NumThrPerRow = TileN / NumElementPerThread;

  static_assert(NumThrPerCta % TileM == 0);
  static_assert(TileM * TileN % NumThrPerCta == 0);


  CUTLASS_DEVICE void
  load(void const *input_ptr, void *thr_output_reg, int64_t m, int64_t n, int blk_m, int blk_n, int tid) {
    int n_alignment = (n & 31) * sizeof(InputDtype);
    int thr_m_offset = tid / NumThrPerRow;
    int thr_n_offset = (tid % NumThrPerRow) * NumElementPerThread;
    void const *cta_input_ptr = (void*)((InputDtype*)input_ptr + blk_m * TileM * n + blk_n * TileN);
    void const *thr_input_ptr = (void*)((InputDtype*)cta_input_ptr + thr_m_offset * n + thr_n_offset);
    InputDtype tmp_reg[NumElementPerThread];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NumElementPerThread; ++i)
      tmp_reg[i] = InputDtype(0.f);
    bool pred = IsEvenM ? true : thr_m_offset + blk_m * TileM < m;
    int limit = IsEvenN ? NumElementPerThread : MIN(NumElementPerThread, n - (blk_n * TileN + thr_n_offset));
    if (n_alignment % 128 == 0)
      _load<int4, IsEvenN>(thr_input_ptr, (void*)tmp_reg, limit, pred);
    else if (n_alignment % 64 == 0)
      _load<int2, IsEvenN>(thr_input_ptr, (void*)tmp_reg, limit, pred);
    else
      _load<InputDtype, IsEvenN>(thr_input_ptr, (void*)tmp_reg, limit, pred);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NumElementPerThread; ++i)
      *((float*)thr_output_reg + i) = static_cast<float>(reinterpret_cast<InputDtype const&>(tmp_reg[i]));
  }

private:
  template <class LoadDataType, bool IsEven>
  CUTLASS_DEVICE void
  _load(void const *thr_input_ptr, void *thr_output_reg, int limit, bool pred) {
    static constexpr int NumElementPerLoad = sizeof(LoadDataType) / sizeof(InputDtype);
    if (pred) {
      if constexpr (IsEven) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; i += NumElementPerLoad) {
          *(LoadDataType*)((InputDtype*)thr_output_reg + i) = *(LoadDataType*)((InputDtype*)thr_input_ptr + i);
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < limit; i += NumElementPerLoad) {
          if (limit - i > NumElementPerLoad)
            *(LoadDataType*)((InputDtype*)thr_output_reg + i) = *(LoadDataType*)((InputDtype*)thr_input_ptr + i);
          else {
            for (int j = 0; j < NumElementPerLoad; ++j) {
              if (i + j < limit)
                *((InputDtype*)thr_output_reg + i + j) = *((InputDtype*)thr_input_ptr + i + j);
            }
          }
        }
      }
    }
  }
};
