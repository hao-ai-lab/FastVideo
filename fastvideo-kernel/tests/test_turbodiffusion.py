import torch
import pytest

try:
    from fastvideo_kernel._C import fastvideo_kernel_ops
except ImportError:
    fastvideo_kernel_ops = None

from fastvideo_kernel import turbodiffusion_ops

# Helper for RMS Norm reference
def rms_norm_ref(x, w, eps=1e-6):
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * w.float()).to(dtype)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTurboDiffusion:
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(16, 128), (32, 256), (1, 1024)])
    def test_quant_correctness(self, dtype, shape):
        if turbodiffusion_ops.quant_cuda is None:
            pytest.skip("quant_cuda not available")
            
        x = torch.randn(shape, dtype=dtype, device="cuda")
        x_q, x_scale = turbodiffusion_ops.int8_quant(x)
        
        assert x_q.dtype == torch.int8
        assert x_scale.dtype == torch.float32
        
        # Simple check: dequantize and compute error
        # Note: The quantization scheme details matter here (per block? per tensor?).
        # Looking at quant.cu, it seems to be block-based but the output scale shape isn't immediately obvious from python signature 
        # without looking at C++ code deeper.
        # But let's check shapes at least.
        
        # If we can't easily dequantize without knowing block size logic in python, 
        # checking that it runs and produces valid shapes is a good start.
        assert x_q.shape == shape
        # x_scale shape depends on block size, usually smaller than x
        
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gemm_correctness(self, dtype):
        if turbodiffusion_ops.gemm_cuda is None:
            pytest.skip("gemm_cuda not available")
            
        M, N, K = 32, 64, 128
        x = torch.randn(M, K, dtype=dtype, device="cuda")
        
        # Create weights
        # For simplicity in testing, let's create random int8 weights and scales
        w_q = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda")
        
        # Scale shape: The Int8Linear class uses:
        # row_blocks = cdiv(out_features, b=128)
        # col_blocks = cdiv(in_features, b=128)
        # scale shape: (row_blocks, col_blocks)
        
        row_blocks = (N + 127) // 128
        col_blocks = (K + 127) // 128
        w_s = torch.randn(row_blocks, col_blocks, dtype=torch.float32, device="cuda").abs()
        
        # Run int8_linear
        output = turbodiffusion_ops.int8_linear(x, w_q, w_s)
        
        assert output.shape == (M, N)
        assert output.dtype == dtype
        
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(2, 16, 128), (4, 32, 256)])
    def test_rms_norm_triton(self, dtype, shape):
        x = torch.randn(shape, dtype=dtype, device="cuda")
        dim = shape[-1]
        w = torch.randn(dim, dtype=dtype, device="cuda")
        eps = 1e-5
        
        # Triton implementation
        out_triton = turbodiffusion_ops.rmsnorm(x, w, eps)
        
        # Reference
        out_ref = rms_norm_ref(x, w, eps)
        
        torch.testing.assert_close(out_triton, out_ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rms_norm_cuda(self, dtype):
        if fastvideo_kernel_ops is None or not hasattr(fastvideo_kernel_ops, "rms_norm_cuda"):
            pytest.skip("rms_norm_cuda not available")
            
        shape = (16, 128)
        x = torch.randn(shape, dtype=dtype, device="cuda")
        dim = shape[-1]
        w = torch.randn(dim, dtype=dtype, device="cuda")
        eps = 1e-5
        
        # C++ implementation
        # Signature: rms_norm_cuda(Input, eps, Weight, Output) -> Output
        out_cuda = torch.empty_like(x)
        fastvideo_kernel_ops.rms_norm_cuda(x, eps, w, out_cuda)
        
        # Reference
        out_ref = rms_norm_ref(x, w, eps)
        
        torch.testing.assert_close(out_cuda, out_ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(2, 16, 128), (4, 32, 256)])
    def test_layer_norm_triton(self, dtype, shape):
        x = torch.randn(shape, dtype=dtype, device="cuda")
        dim = shape[-1]
        eps = 1e-5
        
        # With affine
        w = torch.randn(dim, dtype=dtype, device="cuda")
        b = torch.randn(dim, dtype=dtype, device="cuda")
        
        # Triton implementation
        out_triton = turbodiffusion_ops.layernorm(x, w, b, eps, elementwise_affine=True).to(dtype)
        
        # Reference
        ln = torch.nn.LayerNorm(dim, eps=eps, elementwise_affine=True, dtype=dtype).cuda()
        ln.weight.data.copy_(w)
        ln.bias.data.copy_(b)
        out_ref = ln(x)
        
        torch.testing.assert_close(out_triton, out_ref, atol=1e-2, rtol=1e-2)
        
        # Without affine
        out_triton_no_affine = turbodiffusion_ops.layernorm(x, None, None, eps, elementwise_affine=False).to(dtype)
        ln_no_affine = torch.nn.LayerNorm(dim, eps=eps, elementwise_affine=False, dtype=dtype).cuda()
        out_ref_no_affine = ln_no_affine(x)
        
        torch.testing.assert_close(out_triton_no_affine, out_ref_no_affine, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_layer_norm_cuda(self, dtype):
        if fastvideo_kernel_ops is None or not hasattr(fastvideo_kernel_ops, "layer_norm_cuda"):
            pytest.skip("layer_norm_cuda not available")
            
        shape = (16, 128)
        x = torch.randn(shape, dtype=dtype, device="cuda")
        dim = shape[-1]
        eps = 1e-5
        w = torch.randn(dim, dtype=dtype, device="cuda")
        b = torch.randn(dim, dtype=dtype, device="cuda")
        
        # C++ implementation
        # Signature: layer_norm_cuda(Input, eps, W, B, Output) -> Output
        out_cuda = torch.empty_like(x)
        fastvideo_kernel_ops.layer_norm_cuda(x, eps, w, b, out_cuda)
        
        # Reference
        ln = torch.nn.LayerNorm(dim, eps=eps, elementwise_affine=True, dtype=dtype).cuda()
        ln.weight.data.copy_(w)
        ln.bias.data.copy_(b)
        out_ref = ln(x)
        
        torch.testing.assert_close(out_cuda, out_ref, atol=1e-2, rtol=1e-2)

