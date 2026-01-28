/*
 * Copyright (c) 2025 by SageAttention team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cmath>
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "utils.h"

namespace flash {

using namespace cute;

/*
 * Two-Level Quantization Architecture:
 * 
 * For P (attention probabilities, computed in this file):
 *   - Level 1: s_P1 = absmax_row(P̃)/(448×6) - per-row scale (P >= 0, so max == absmax)
 *   - Level 2: s_P2 = absmax_block(P̃/s_P1)/6 - per-16-element scale
 *   - Controlled by: single_level_p_quant flag
 * 
 * For Q, K, V (computed in quantization kernels, see fp4_quantization_4d.cu):
 *   - Level 1: s_row = absmax_row(x)/(448×6) - per-row scale (stored in sfq_row, sfk_row, sfv_row)
 *   - Level 2: s_16 = absmax_block(x/s_row)/6 - per-16-element scale (stored in sfq, sfk, sfv)
 *   - Controlled by: two_level_qkv_quant flag (params.two_level_qkv_quant)
 *   - Note: Q, K, V can have negative values, so absmax (not max) is required
 * 
 * When two_level_qkv_quant is enabled:
 *   - The quantized values are: x_fp4 = round(x / (s_row * s_16))
 *   - Dequantization requires: x ≈ x_fp4 * s_16 * s_row
 *   - The row scales are passed via params.sfq_row_ptr, params.sfk_row_ptr, params.sfv_row_ptr
 *   - The kernel needs to apply both scale levels during MMA computation
 * 
 * The two-level approach provides better dynamic range by:
 *   1. First normalizing values to a standard range using s_row
 *   2. Then applying finer-grained per-block quantization
 */

template <int Rows>
struct SoftmaxFused{

    using TensorT = decltype(make_fragment_like<float>(Shape<Int<Rows>>{}));
    TensorT row_sum, row_max, scores_scale;
    static constexpr float fp8_scalexfp4_scale = 1.f / (448 * 6);
    static constexpr float fp8_scalexfp4_scale_log2 = -11.392317422778762f; //log2f(fp8_scalexfp4_scale)
    static constexpr float fp4_scale_log2 = -2.584962500721156f; // log2f(fp4_scale)
    static constexpr int RowReductionThr = 4;
    
    // If true, use single-level quantization: s_P2, P̂_2 = φ(P̃) directly (standard per-block FP4 quantization like V)
    // If false (default), use two-level quantization: s_P1 = rowmax(P̃)/(448×6), then s_P2, P̂_2 = φ(P̃/s_P1)
    bool single_level_p_quant;

    CUTLASS_DEVICE SoftmaxFused(bool single_level = false) : single_level_p_quant(single_level) {};

    template<bool FirstTile, bool InfCheck = false, typename TensorAcc, typename TensorMax>
    CUTLASS_DEVICE auto online_softmax_with_quant(
        TensorAcc& acc, 
        TensorMax& AbsMaxP,
        const float softmax_scale_log2
    ) {
        Tensor acc_reduction_view = make_tensor(acc.data(), flash::convert_to_reduction_layout(acc.layout()));
        Tensor acc_conversion_view = make_tensor(acc.data(), flash::convert_to_conversion_layout(acc.layout()));
        Tensor acc_conversion_flatten = group_modes<1, 5>(group_modes<0, 2>(flatten(acc_conversion_view)));
        
        if constexpr (FirstTile) {
            fill(row_max, -INFINITY);
            clear(row_sum);
            fill(scores_scale, 1.f);

            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), acc_reduction_view(mi, make_coord(ei, ni)));
                    }
                    float max_recv = __shfl_xor_sync(int32_t(-1), AbsMaxP(mi, ni), 1); // exchange max with neighbour thread of 8 elements
                    AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), max_recv);
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }
                
                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2); // exchange max in a quad in a row
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                // Two-level P quantization (default): s_P1 = absmax_row(P̃)/(448×6), then s_P2,P̂_2 = φ(P̃/s_P1)
                //   - Pre-scales P to [0, 448×6] range before φ, output scaled by s_P1
                //   - Since P is softmax output (P >= 0), absmax == max
                // Single-level P quantization: s_P2, P̂_2 = φ(P̃) directly (like V quantization)
                //   - No s_P1, just standard per-block FP4 quantization φ
                const float s_P1_offset = single_level_p_quant ? 0.f : fp8_scalexfp4_scale_log2;
                const float max_scaled = InfCheck
                                        ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + s_P1_offset))
                                        : (row_max(mi) * softmax_scale_log2 + s_P1_offset);
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                }
                // s_P2 = absmax(P_block)/6 — per-block scale factor from φ function (same formula for both modes)
                // The difference is in max_scaled: two-level includes 448×6 pre-scaling, single-level doesn't
                // Since P >= 0 after softmax, absmax == max
                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = flash::ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
            }
        }
        else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    float local_max = -INFINITY;
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        local_max = fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
                    }
                    float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1); // exchange max with neighbour thread of 8 elements
                    AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }
                
                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2); // exchange max in a quad in a row
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                float scores_max_cur = !InfCheck
                                        ? row_max(mi)
                                        : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = flash::ptx_exp2((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);

                // Two-level P quantization (default): s_P1 = absmax_row(P̃)/(448×6), then s_P2,P̂_2 = φ(P̃/s_P1)
                //   - Since P is softmax output (P >= 0), absmax == max
                // Single-level P quantization: s_P2, P̂_2 = φ(P̃) directly (like V quantization)
                const float s_P1_offset = single_level_p_quant ? 0.f : fp8_scalexfp4_scale_log2;
                const float max_scaled = InfCheck
                                        ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + s_P1_offset))
                                        : (row_max(mi) * softmax_scale_log2 + s_P1_offset);
                row_sum(mi) = row_sum(mi) * scores_scale(mi);
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
                // s_P2 = absmax(P_block)/6 — per-block scale factor from φ function (absmax == max since P >= 0)
                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = flash::ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
                }
                // scores_scale(mi) = max_scaled;
            }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(AbsMaxP); ++i) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<0>(acc_conversion_flatten); ++j)
                acc_conversion_flatten(j, i) /= AbsMaxP(i);
        }
    }

    template<typename TensorAcc>
    CUTLASS_DEVICE void finalize(TensorAcc& o_store) {
        Tensor o_store_reduction_view = make_tensor(o_store.data(), flash::convert_to_reduction_layout(o_store.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 1; i < RowReductionThr; i <<= 1) {
                float sum_recv = __shfl_xor_sync(int32_t(-1), row_sum(mi), i);
                row_sum(mi) += sum_recv;
            }
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1 / sum;
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) { 
                o_store_reduction_view(mi, ni) *= inv_sum;
             }
        }
    }

    template<typename TensorAcc>
    CUTLASS_DEVICE void rescale_o(TensorAcc& o_store, TensorAcc const& o_tmp) {
        Tensor o_store_reduction_view = make_tensor(o_store.data(), flash::convert_to_reduction_layout(o_store.layout()));
        Tensor o_tmp_reduction_view = make_tensor(o_tmp.data(), flash::convert_to_reduction_layout(o_tmp.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) { 
                o_store_reduction_view(mi, ni) = o_store_reduction_view(mi, ni) * scores_scale(mi) + o_tmp_reduction_view(mi, ni);
             }
        }

    }


};
} // namespace flash