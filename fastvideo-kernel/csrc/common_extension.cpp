#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// Forward declarations
#ifdef TK_COMPILE_ST_ATTN
extern torch::Tensor sta_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, 
    int kernel_t_size, int kernel_w_size, int kernel_h_size, 
    int text_length, bool process_text, bool has_text, int kernel_aspect_ratio_flag
); 
#endif

#ifdef TK_COMPILE_BLOCK_SPARSE
extern std::vector<torch::Tensor> block_sparse_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,  
    torch::Tensor q2k_block_sparse_index, torch::Tensor q2k_block_sparse_num, torch::Tensor block_size
); 
extern std::vector<torch::Tensor> block_sparse_attention_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l_vec, torch::Tensor og, 
    torch::Tensor k2q_block_sparse_index, torch::Tensor k2q_block_sparse_num, torch::Tensor block_size
);
#endif

// TurboDiffusion kernels
void register_quant(pybind11::module_ &);
void register_rms_norm(pybind11::module_ &);
void register_layer_norm(pybind11::module_ &);
void register_gemm(pybind11::module_ &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FastVideo CUDA Kernels";

#ifdef TK_COMPILE_ST_ATTN
    m.def("sta_fwd", torch::wrap_pybind_function(sta_forward), "sliding tile attention (Hopper)");
#endif

#ifdef TK_COMPILE_BLOCK_SPARSE
    m.def("block_sparse_fwd", torch::wrap_pybind_function(block_sparse_attention_forward), "block sparse attention forward (Hopper)");
    m.def("block_sparse_bwd", torch::wrap_pybind_function(block_sparse_attention_backward), "block sparse attention backward (Hopper)");
#endif

    // TurboDiffusion
    register_quant(m);
    register_rms_norm(m);
    register_layer_norm(m);
    register_gemm(m);
}
