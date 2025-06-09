#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>

#ifdef TK_COMPILE_BLOCK_SPARSE
extern std::vector<torch::Tensor> block_sparse_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,  torch::Tensor q2k_block_sparse_index, torch::Tensor q2k_block_sparse_num
); 
extern std::vector<torch::Tensor> block_sparse_attention_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l_vec, torch::Tensor og, torch::Tensor k2q_block_sparse_index, torch::Tensor k2q_block_sparse_num
);
#endif

#ifdef TK_COMPILE_ST_ATTN

extern torch::Tensor sta_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, int kernel_t_size, int kernel_w_size, int kernel_h_size, int text_length, bool process_text, bool has_text, int kernel_aspect_ratio_flag
); 
#endif

#ifdef TK_COMPILE_MHA
extern std::vector<torch::Tensor> mha_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v
);
extern std::vector<torch::Tensor> mha_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l_vec, torch::Tensor og
);
#endif
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Sliding Block Attention Kernels"; // optional module docstring

#ifdef TK_COMPILE_BLOCK_SPARSE
    m.def("block_sparse_fwd",  torch::wrap_pybind_function(block_sparse_attention_forward), "block sparse attention");
    m.def("block_sparse_bwd",  torch::wrap_pybind_function(block_sparse_attention_backward), "block sparse attention backward");
#endif

#ifdef TK_COMPILE_ST_ATTN
    m.def("sta_fwd",  torch::wrap_pybind_function(sta_forward), "sliding tile attention, assuming tile size is (6,8,8)");
#endif

#ifdef TK_COMPILE_MHA
    m.def("mha_fwd",  torch::wrap_pybind_function(mha_forward), "mha forward");
    m.def("mha_bwd",  torch::wrap_pybind_function(mha_backward), "mha backward");
#endif
}
