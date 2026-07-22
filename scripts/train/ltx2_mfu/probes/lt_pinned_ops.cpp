// Minimal cuBLASLt pinned-algo GEMM registry for the LTX-2 packed shapes.
// Python owns tuning (probes/bench_ltx2_cublaslt_algo_sweep.py logic) and
// registers one case per (m, n, k, transa, transb) with the winning
// 64-byte cublasLtMatmulAlgo_t blob; lt_mm dispatches by case id with zero
// Python in the hot path. BF16 in/out, FP32 compute, no epilogue.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace {

struct PinnedCase {
  cublasLtMatmulDesc_t op_desc{};
  cublasLtMatrixLayout_t layout_a{};
  cublasLtMatrixLayout_t layout_b{};
  cublasLtMatrixLayout_t layout_d{};
  cublasLtMatmulAlgo_t algo{};
  int64_t m{}, n{}, k{};
  int64_t d_rows{}, d_cols{};
};

cublasLtHandle_t lt_handle() {
  static cublasLtHandle_t handle = [] {
    cublasLtHandle_t h;
    TORCH_CHECK(cublasLtCreate(&h) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate failed");
    return h;
  }();
  return handle;
}

std::vector<PinnedCase>& cases() {
  static std::vector<PinnedCase> registry;
  return registry;
}

torch::Tensor& workspace() {
  static torch::Tensor ws = torch::empty(
      {128 * 1024 * 1024},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
  return ws;
}

void check_lt(cublasStatus_t status, const char* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with status ", static_cast<int>(status));
}

int64_t register_case(int64_t m, int64_t n, int64_t k, bool transa, bool transb,
                      int64_t lda, int64_t ldb, int64_t d_rows, int64_t d_cols,
                      torch::Tensor algo_bytes) {
  TORCH_CHECK(algo_bytes.dtype() == torch::kUInt8 && algo_bytes.numel() == sizeof(cublasLtMatmulAlgo_t),
              "algo blob must be 64 uint8 bytes");
  PinnedCase entry;
  entry.m = m; entry.n = n; entry.k = k;
  entry.d_rows = d_rows; entry.d_cols = d_cols;
  check_lt(cublasLtMatmulDescCreate(&entry.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "descCreate");
  const cublasOperation_t opa = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t opb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  check_lt(cublasLtMatmulDescSetAttribute(entry.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opa, sizeof(opa)), "setA");
  check_lt(cublasLtMatmulDescSetAttribute(entry.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opb, sizeof(opb)), "setB");
  const int64_t a_rows = transa ? lda : m;
  const int64_t a_cols = transa ? m : k;
  const int64_t b_rows = transb ? ldb : k;
  const int64_t b_cols = transb ? k : n;
  check_lt(cublasLtMatrixLayoutCreate(&entry.layout_a, CUDA_R_16BF, a_rows, a_cols, lda), "layoutA");
  check_lt(cublasLtMatrixLayoutCreate(&entry.layout_b, CUDA_R_16BF, b_rows, b_cols, ldb), "layoutB");
  check_lt(cublasLtMatrixLayoutCreate(&entry.layout_d, CUDA_R_16BF, m, n, m), "layoutD");
  std::memcpy(&entry.algo, algo_bytes.data_ptr<uint8_t>(), sizeof(cublasLtMatmulAlgo_t));
  cases().push_back(entry);
  return static_cast<int64_t>(cases().size()) - 1;
}

torch::Tensor lt_mm(torch::Tensor a, torch::Tensor b, int64_t case_id) {
  TORCH_CHECK(case_id >= 0 && case_id < static_cast<int64_t>(cases().size()), "unknown lt case");
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.dtype() == torch::kBFloat16 && b.dtype() == torch::kBFloat16,
              "lt_mm expects CUDA bf16 tensors");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "lt_mm expects contiguous operands");
  const PinnedCase& entry = cases()[case_id];
  auto d = torch::empty({entry.d_rows, entry.d_cols}, a.options());
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_lt(cublasLtMatmul(lt_handle(), entry.op_desc, &alpha,
                          a.data_ptr(), entry.layout_a,
                          b.data_ptr(), entry.layout_b,
                          &beta,
                          d.data_ptr(), entry.layout_d,
                          d.data_ptr(), entry.layout_d,
                          &entry.algo,
                          workspace().data_ptr(), workspace().numel(),
                          at::cuda::getCurrentCUDAStream()), "ltMatmul");
  return d;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("register_case", &register_case, "register pinned cuBLASLt case");
  module.def("lt_mm", &lt_mm, "run pinned cuBLASLt matmul");
}
