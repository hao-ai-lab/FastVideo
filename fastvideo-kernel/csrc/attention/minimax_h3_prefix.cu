// Exact Omni prefix attention: Torch-pinned FlashAttention, larger query tile only.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include "flash_fwd_launch_template.h"

using namespace FLASH_NAMESPACE;

template<int M, int W, bool QRegs = false, bool Share = false>
struct Variant {
    using Traits = Flash_fwd_kernel_traits<128, M, 64, W, QRegs, Share, cutlass::bfloat16_t>;
    static auto kernel() {
        // Keep the original K tile and reduction order. All masking stays enabled.
        return flash_fwd_kernel<Traits, false, false, false, false, false, true, false, false>;
    }
    static void setup() {
        static const bool initialized = []() {
            if (Traits::kSmemSize >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   Traits::kSmemSize));
            }
            return true;
        }();
        (void) initialized;
    }
    static void launch(Flash_fwd_params &params, cudaStream_t stream) {
        setup();
        dim3 grid((params.seqlen_q + M - 1) / M, params.b, params.h);
        auto function = kernel();
        function<<<grid, W * 32, Traits::kSmemSize, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    static std::vector<int64_t> metadata() {
        setup();
        cudaFuncAttributes attributes;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel()));
        int blocks = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, kernel(), W * 32, Traits::kSmemSize));
        return {M, W, Traits::kSmemSize, attributes.numRegs,
                static_cast<int64_t>(attributes.localSizeBytes), blocks};
    }
};

torch::Tensor prefix_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda() && k.device() == q.device() && v.device() == q.device(), "CUDA operands required");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.sizes() == k.sizes(), "BHSD operands required");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                v.scalar_type() == at::kBFloat16, "BF16 operands required");
    TORCH_CHECK(q.size(3) == 128 && k.size(3) == 128 && q.stride(3) == 1 &&
                k.stride(3) == 1 && v.stride(3) == 1, "Contiguous 128-wide heads required");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1), "Batch/head mismatch");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 && k.size(2) > 0,
                "Nonempty attention required");
    c10::cuda::CUDAGuard guard(q.device());
    auto out = torch::empty({q.size(0), q.size(2), q.size(1), 128}, q.options()).transpose(1, 2);
    auto lse = torch::empty({q.size(0), q.size(1), q.size(2)}, q.options().dtype(at::kFloat));
    Flash_fwd_params params{};
    params.q_ptr = q.data_ptr(); params.k_ptr = k.data_ptr(); params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr(); params.softmax_lse_ptr = lse.data_ptr();
    params.q_batch_stride = q.stride(0); params.q_head_stride = q.stride(1); params.q_row_stride = q.stride(2);
    params.k_batch_stride = k.stride(0); params.k_head_stride = k.stride(1); params.k_row_stride = k.stride(2);
    params.v_batch_stride = v.stride(0); params.v_head_stride = v.stride(1); params.v_row_stride = v.stride(2);
    params.o_batch_stride = out.stride(0); params.o_head_stride = out.stride(1); params.o_row_stride = out.stride(2);
    params.b = q.size(0); params.h = q.size(1); params.h_k = k.size(1); params.h_h_k_ratio = 1;
    params.seqlen_q = q.size(2); params.seqlen_k = k.size(2); params.total_q = params.b * params.seqlen_q;
    params.seqlen_q_rounded = (params.seqlen_q + 127) / 128 * 128;
    params.seqlen_k_rounded = (params.seqlen_k + 127) / 128 * 128;
    params.d = 128; params.d_rounded = 128;
    params.scale_softmax = static_cast<float>(1.0 / std::sqrt(128.0));
    params.scale_softmax_log2 = params.scale_softmax * M_LOG2E;
    params.p_dropout = 1.0f; params.p_dropout_in_uint8_t = 255; params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.window_size_left = -1; params.window_size_right = -1;
    params.is_bf16 = true; params.is_seqlens_k_cumulative = true; params.num_splits = 1;
    auto stream = at::cuda::getCurrentCUDAStream();
    Variant<256, 8>::launch(params, stream);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, binding) {
    binding.def("forward", &prefix_forward);
}
