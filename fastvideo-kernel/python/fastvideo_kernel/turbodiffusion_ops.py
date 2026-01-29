from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Try to load the C++ extension
try:
    from fastvideo_kernel._C import fastvideo_kernel_ops
    quant_cuda = getattr(fastvideo_kernel_ops, "quant_cuda", None)
    gemm_cuda = getattr(fastvideo_kernel_ops, "gemm_cuda", None)
    rms_norm_cuda = getattr(fastvideo_kernel_ops, "rms_norm_cuda", None)
    layer_norm_cuda = getattr(fastvideo_kernel_ops, "layer_norm_cuda", None)
except ImportError:
    quant_cuda = None
    gemm_cuda = None
    rms_norm_cuda = None
    layer_norm_cuda = None

def int8_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a floating-point tensor to int8 using a custom CUDA kernel.

    Args:
        x (torch.Tensor): Input tensor of type float16/bfloat16.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - x_q: Quantized int8 tensor.
            - x_scale: Per-block scale tensor used for quantization.
    """
    x_q, x_scale = quant_cuda(x, None, None)
    return x_q, x_scale


def int8_linear(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Perform an int8 GEMM (matrix multiplication) using quantized weights and a
    quantized version of the input. The underlying compute is performed by a
    custom CUDA kernel.

    Args:
        x (torch.Tensor): Input activation of shape (M, K) in float32.
        w_q (torch.Tensor): Quantized int8 weight tensor of shape (N, K).
        w_s (torch.Tensor): Scale tensor associated with w_q.
        **kwargs: Additional options (reserved for future use).

    Returns:
        torch.Tensor: Output tensor of shape (M, N) in float32.
    """
    assert w_q.dtype == torch.int8, "Weight tensor must be int8."
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    m = x.shape[0]
    n = w_q.shape[0]
    y = torch.zeros(m, n, dtype=x.dtype, device=x.device)

    x_q, x_s = int8_quant(x)
    gemm_cuda(x_q, x_s, w_q, w_s, y)
    return y.reshape(*shape[:-1], n)

def flatten_if_batched(*tensors):
    """
    Flattens all input tensors from (B, N, D_i) to (B * N, D_i) if they are batched (3D).

    Args:
        *tensors: Any number of input tensors, each must have shape (B, N, D_i) or (N, D_i)

    Returns:
        flat_tensors: List of flattened tensors
        batched: Boolean flag indicating whether inputs were batched
        batch_size: Batch size if batched, else None
    """
    if not tensors:
        raise ValueError("At least one tensor must be provided.")

    first = tensors[0]
    assert len(first.shape) in [
        2,
        3,
    ], "Input tensors must be batched (3D) or not batched (2D)"

    if len(first.shape) == 3:  # batched
        batched = True
        batch_size = first.shape[0]
        assert all(t.shape[0] == batch_size for t in tensors), "All input tensors must have the same batch size"
        assert all(
            t.shape[1] == first.shape[1] for t in tensors
        ), "All input tensors must have the same sequence length"
        flat_tensors = [t.reshape(-1, t.shape[-1]) for t in tensors]
    else:
        batched = False
        batch_size = None
        flat_tensors = list(tensors)

    return flat_tensors, batched, batch_size


@triton.jit
def _rms_norm_fwd_fused(
    X,
    Y,
    W,
    Rstd,
    x_stride,
    y_stride,
    N: tl.constexpr,  # number of columns in X,
    N2: tl.constexpr,
    eps,  # epsilon to avoid division by zero
    BLOCK_M: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    mask = cols < N

    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]

    x = tl.load(x_ptr, mask=mask[None, :], other=0.0).to(tl.float32)

    # Compute variance
    _var = x * x
    var = tl.sum(_var, axis=1) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Write mean / rstd
    tl.store(Rstd + rows, rstd)
    rstd = tl.reshape(rstd, (BLOCK_M, 1))

    # Normalize and apply linear transformation
    w = tl.load(W + cols)
    x_hat = x * rstd
    y = x_hat * w

    # Write output
    y = y.to(Y.type.element_ty)
    tl.store(y_ptr, y, mask=mask[None, :])


def rmsnorm(x, w, eps):
    """
    Forward pass of the RMSNorm.

    Args:
        x (torch.Tensor): Input tensor, High precision.
        w (torch.Tensor): RMSNorm weight tensor.
        eps (float): RMSNorm epsilon value.

    Returns:
        y (torch.Tensor): Output tensor, High precision.
        rstd (torch.Tensor): Inverse standard deviation, needed for backward.
    """
    assert x.is_contiguous(), "Input must be contiguous"
    # Change batched 3D input to 2D
    [x], batched, BS = flatten_if_batched(x)

    # allocate output
    M, N = x.shape
    y = torch.empty_like(x, dtype=x.dtype)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    # heuristics for number of warps
    num_warps = 8
    
    # Avoid illegal memory access
    N2 = triton.next_power_of_2(N)
    
    if N <= 512:
        BLOCK_M = 32
    else:
        BLOCK_M = 1

    # Call the triton kernel
    _rms_norm_fwd_fused[(triton.cdiv(M, BLOCK_M),)](  #
        x,
        y,
        w,
        rstd,  #
        x.stride(0),
        y.stride(0),
        N,
        N2,
        eps,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    # Recover 2D to 3D
    if batched:
        y = y.reshape(BS, -1, y.shape[-1])

    return y, rstd

@triton.jit
def _layer_norm_param_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    x_stride,  # how much to increase the pointer when moving by 1 row
    y_stride,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X,
    N2: tl.constexpr,  # number of columns in X,
    eps,  # epsilon to avoid division by zero
    BLOCK_M: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    mask = cols < N

    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]

    x = tl.load(x_ptr, mask=mask[None, :], other=0.0).to(tl.float32)

    # Compute mean and Variance
    mean = tl.sum(x, axis=1, keep_dims=True) / N
    # Compute variance
    _var = (x - mean) * (x - mean)
    var = tl.sum(_var, axis=1, keep_dims=True) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Write mean / rstd
    _mean = tl.reshape(mean, (BLOCK_M))
    _rstd = tl.reshape(rstd, (BLOCK_M))
    tl.store(Mean + rows, _mean)
    tl.store(Rstd + rows, _rstd)

    # Normalize and apply linear transformation
    x_hat = (x - mean) * rstd

    w = tl.load(W + cols)
    b = tl.load(B + cols)
    
    x_hat = x_hat * w + b

    # Write output
    x_hat = x_hat.to(Y.type.element_ty)
    tl.store(y_ptr, x_hat, mask=mask[None, :])


def layernorm_param(x, w, b, eps):
    # Change batched 3D input to 2D
    [x], batched, BS = flatten_if_batched(x)

    # allocate output
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # heuristics for number of warps
    num_warps = 8

    N2 = triton.next_power_of_2(N)
    
    if N <= 512:
        BLOCK_M = 32
    else:
        BLOCK_M = 1

    # enqueue kernel
    _layer_norm_param_fwd_fused[(triton.cdiv(M, BLOCK_M),)](  #
        x,
        y,
        w,
        b,
        mean,
        rstd,  #
        x.stride(0),
        y.stride(0),
        N,
        N2,
        eps,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    # Recover 2D to 3D
    if batched:
        y = y.reshape(BS, -1, y.shape[-1])

    return y, mean, rstd


########################################################
# Elementwise_affine=False
########################################################


@triton.jit
def _layer_norm_noparam_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    x_stride,  # how much to increase the pointer when moving by 1 row
    y_stride,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X,
    N2: tl.constexpr,  # number of columns in X,
    eps,  # epsilon to avoid division by zero
    BLOCK_M: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    mask = cols < N

    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]

    x = tl.load(x_ptr, mask=mask[None, :], other=0.0).to(tl.float32)

    # Compute mean and Variance
    mean = tl.sum(x, axis=1, keep_dims=True) / N
    # Compute variance
    _var = (x - mean) * (x - mean)
    var = tl.sum(_var, axis=1, keep_dims=True) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Write mean / rstd
    _mean = tl.reshape(mean, (BLOCK_M))
    _rstd = tl.reshape(rstd, (BLOCK_M))
    tl.store(Mean + rows, _mean)
    tl.store(Rstd + rows, _rstd)

    # Normalize and apply linear transformation
    x_hat = (x - mean) * rstd

    # Write output
    x_hat = x_hat.to(Y.type.element_ty)
    tl.store(y_ptr, x_hat, mask=mask[None, :])


def layernorm_noparam(x, eps):
    assert x.is_contiguous(), "Input must be contiguous"

    # Change batched 3D input to 2D
    [x], batched, BS = flatten_if_batched(x)

    # allocate output
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # heuristics for number of warps
    num_warps = 8

    N2 = triton.next_power_of_2(N)
    
    if N <= 512:
        BLOCK_M = 32
    else:
        BLOCK_M = 1

    # enqueue kernel
    _layer_norm_noparam_fwd_fused[(triton.cdiv(M, BLOCK_M),)](  #
        x,
        y,
        mean,
        rstd,  #
        x.stride(0),
        y.stride(0),
        N,
        N2,
        eps,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    # Recover 2D to 3D
    if batched:
        y = y.reshape(BS, -1, y.shape[-1])

    return y, mean, rstd

def layernorm(x, w, b, eps, elementwise_affine=True):
    if elementwise_affine:
        assert w is not None and b is not None
        return layernorm_param(x, w, b, eps)
    else:
        assert w is None and b is None
        return layernorm_noparam(x, eps)

def cdiv(a: int, b: int):
    return (a + b - 1) // b

def dequantize_weight(w_q, w_s, dtype):
    # w_q: (N, K) int8
    # w_s: (NB, KB) float32
    # Block size 128
    BLOCK = 128
    N, K = w_q.shape
    # Expand w_s
    # Repeat interleave
    w_s_exp = w_s.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    # Crop
    w_s_exp = w_s_exp[:N, :K]
    return w_q.to(dtype) * w_s_exp.to(dtype)

class Int8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_q, w_s, bias=None):
        ctx.save_for_backward(x, w_q, w_s, bias)
        ctx.bias_requires_grad = bias.requires_grad if bias is not None else False
        
        out = int8_linear(x, w_q, w_s)
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w_q, w_s, bias = ctx.saved_tensors
        
        grad_input = None
        grad_weight = None 
        grad_scale = None
        grad_bias = None
        
        if ctx.needs_input_grad[0]: 
            # grad_input = grad_output @ W
            w_float = dequantize_weight(w_q, w_s, grad_output.dtype)
            grad_input = torch.matmul(grad_output, w_float)
            
        if ctx.bias_requires_grad:
            dim_to_sum = list(range(grad_output.dim() - 1))
            grad_bias = grad_output.sum(dim=dim_to_sum)
            
        return grad_input, grad_weight, grad_scale, grad_bias

class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        y, rstd = rmsnorm(x, w, eps)
        ctx.save_for_backward(x, w, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # x, w, rstd are saved
        # grad_output is dL/dy
        # We can implement backward using torch ops for correctness
        x, w, rstd = ctx.saved_tensors
        eps = ctx.eps
        N = x.shape[-1]
        
        # dL/dy * w
        dx = grad_output * w
        
        # Expand rstd
        rstd = rstd.unsqueeze(-1) # (B, 1) or (M, 1)
        if x.dim() == 3:
            # Flatten if necessary or handle dimensions. 
            # forward flattens x to (M, N) internally but saved x might be 3D?
            # rmsnorm takes x, flattens it. But does it modify x in place? No.
            # But ctx.save_for_backward saves the original x (3D if input was 3D).
            # rstd is (M,).
            # We should flatten x and grad_output to match logic
            x_flat = x.reshape(-1, N)
            grad_output_flat = grad_output.reshape(-1, N)
            dx = dx.reshape(-1, N)
        else:
            x_flat = x
            grad_output_flat = grad_output
            dx = dx
            
        # Standard RMSNorm backward
        # dy = grad_output_flat
        # x_hat = x * rstd
        # w * dy
        # c1 = mean(dy * w * x) * rstd^2
        # dx = (dy * w - c1 * x) * rstd
        
        # More precise:
        # y = x * rstd * w
        # dy = grad_output
        # dw = sum(dy * x * rstd, dim=0)
        
        grad_w = None
        if ctx.needs_input_grad[1]: # w
             # grad_w = (grad_output_flat * x_flat * rstd).sum(dim=0)
             # More accurate gradient for weight when considering rstd was computed from x
             # Actually, for Affine part (w), it is just sum(dL/dy * x_hat).
             # y = x_hat * w
             # dL/dw = sum(dL/dy * x_hat)
             x_hat = x_flat * rstd
             grad_w = (grad_output_flat * x_hat).sum(dim=0)

        grad_input = None
        if ctx.needs_input_grad[0]: # x
            # x_hat = x * rstd
            # dx = rstd * (w * dy - mean(w * dy * x_hat) * x_hat) 
            # but for RMSNorm:
            # dx = rstd * (w * dy - (x * rstd^2) * mean(w * dy * x)) -> check formula
            
            # Using PyTorch autograd for reference logic:
            # y = x / sqrt(mean(x^2) + eps) * w
            # Let sigma = sqrt(...)
            # dL/dx = dL/dy * w * (1/sigma) + dL/dsigma * dsigma/dx
            # dsigma/dx = 1/(2*sigma) * 2x/N = x / (N * sigma)
            # dL/dsigma = sum(dL/dy * w * x * (-1/sigma^2))
            # dL/dx = (dL/dy * w)/sigma - sum(dL/dy * w * x) * x / (N * sigma^3)
            #       = (1/sigma) * [ (dL/dy * w) - x * sum(dL/dy * w * x) / (N * sigma^2) ]
            #       = rstd * [ (dL/dy * w) - x * rstd^2 * mean(dL/dy * w * x) ] # Wait, sum/N is mean
            
            # Let's compute:
            dy_w = grad_output_flat * w
            # term2 = (dy_w * x_flat).sum(dim=-1, keepdim=True) / N # mean(dy*w*x)
            # grad_input = rstd * (dy_w - x_flat * (rstd ** 2) * term2)
            
            # Re-derivation:
            # x_hat = x * rstd
            # y = x_hat * w
            # dL/dx = dL/dy * dy/dx
            # dy/dx = w * dx_hat/dx
            # dx_hat/dx = rstd * (I - x * x^T * rstd^2 / N) ?? No.
            # dx_hat_i / dx_j = rstd * (delta_ij - x_i * x_j * rstd^2 / N)
            
            # dL/dx_hat = dL/dy * w
            # dL/dx = rstd * (dL/dx_hat - x * rstd^2 * mean(dL/dx_hat * x))
            
            dx_hat = grad_output_flat * w
            dx_hat_x_mean = (dx_hat * x_flat).mean(dim=-1, keepdim=True)
            grad_input = rstd * (dx_hat - x_flat * (rstd ** 2) * dx_hat_x_mean)
            
            if x.dim() == 3:
                grad_input = grad_input.reshape(x.shape)
            
        return grad_input, grad_w, None

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, eps, elementwise_affine):
        if elementwise_affine:
            y, mean, rstd = layernorm_param(x, w, b, eps)
            ctx.save_for_backward(x, w, b, mean, rstd)
        else:
            y, mean, rstd = layernorm_noparam(x, eps)
            ctx.save_for_backward(x, None, None, mean, rstd)
            
        ctx.eps = eps
        ctx.elementwise_affine = elementwise_affine
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, w, b, mean, rstd = ctx.saved_tensors
        N = x.shape[-1]
        
        if x.dim() == 3:
            x_flat = x.reshape(-1, N)
            grad_output_flat = grad_output.reshape(-1, N)
        else:
            x_flat = x
            grad_output_flat = grad_output
            
        if w is None:
             w_eff = 1.0
        else:
             w_eff = w
        
        # dx calculation
        # x_hat = (x - mean) * rstd
        # y = x_hat * w + b
        # dL/dx_hat = dL/dy * w
        
        dy = grad_output_flat
        dx_hat = dy * w_eff
        
        # dL/dvar = sum(dL/dx_hat * (x-mean) * (-0.5) * (var+eps)^(-1.5))
        #         = -0.5 * rstd^3 * sum(dx_hat * (x-mean))
        # dL/dmean = sum(dL/dx_hat * (-rstd)) + dL/dvar * (-2/N) * sum(x-mean)
        # term sum(x-mean) is 0. So second part vanishes.
        # dL/dmean = -rstd * sum(dx_hat)
        
        # dL/dx = dL/dx_hat * rstd + dL/dvar * 2(x-mean)/N + dL/dmean * 1/N
        #       = dx_hat * rstd + (-0.5 * rstd^3 * sum(dx_hat * (x-mean))) * 2(x-mean)/N + (-rstd * sum(dx_hat))/N
        #       = rstd * [ dx_hat - mean(dx_hat) - (x-mean)*rstd^2 * mean(dx_hat * (x-mean)) ]
        
        x_centered = x_flat - mean.unsqueeze(-1)
        dx_hat_mean = dx_hat.mean(dim=-1, keepdim=True)
        dx_hat_x_centered_mean = (dx_hat * x_centered).mean(dim=-1, keepdim=True)
        
        grad_input = rstd.unsqueeze(-1) * (dx_hat - dx_hat_mean - x_centered * (rstd.unsqueeze(-1)**2) * dx_hat_x_centered_mean)
        
        if x.dim() == 3:
            grad_input = grad_input.reshape(x.shape)

        grad_w = None
        grad_b = None
        
        if ctx.elementwise_affine:
            if ctx.needs_input_grad[1]:
                 x_hat = x_centered * rstd.unsqueeze(-1)
                 grad_w = (dy * x_hat).sum(dim=0)
            if ctx.needs_input_grad[2]:
                 grad_b = dy.sum(dim=0)
                 
        return grad_input, grad_w, grad_b, None, None


class Int8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        row_blocks = cdiv(out_features, b=128)
        col_blocks = cdiv(in_features, b=128)
        
        self.register_buffer("int8_weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.empty((row_blocks, col_blocks), dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=dtype))
        else:
            self.bias = None
        

    def forward(self, x):
        return Int8LinearFunction.apply(x, self.int8_weight, self.scale, self.bias)

    @classmethod
    def from_linear(cls, original_linear: nn.Linear, quantize: bool = True):
    
        int8_layer = cls(
            original_linear.in_features,
            original_linear.out_features,
            bias=original_linear.bias is not None,
            dtype=original_linear.weight.dtype
        )
        if quantize:
            w_data = original_linear.weight.data.cuda()
            int8_w, scale = int8_quant(w_data)

            int8_layer.int8_weight.copy_(int8_w)
            int8_layer.scale.copy_(scale)
            if original_linear.bias is not None:
                int8_layer.bias.data.copy_(original_linear.bias.data.cuda())
            
        return int8_layer
    
class FastRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("weight", torch.ones(dim))

    def forward(self, x):
        return RMSNormFunction.apply(x.float(), self.weight, self.eps).to(x.dtype)

    @classmethod
    def from_rmsnorm(cls, original_rmsnorm):
        rmsnorm_layer = cls(
            dim=original_rmsnorm.dim,
            eps=original_rmsnorm.eps
        )
        if original_rmsnorm.weight.device != torch.device('meta'):
            rmsnorm_layer.weight.data.copy_(original_rmsnorm.weight.float().data)
        return rmsnorm_layer
    
class FastLayerNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = False,
        bias: bool = True
    ) :
        super().__init__()
        self.dim = dim  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.register_buffer("weight", torch.empty(self.dim))
            if bias:
                self.register_buffer("bias", torch.empty(self.dim))
            else:
                self.bias = None
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        return LayerNormFunction.apply(x.float(), self.weight, self.bias, self.eps, self.elementwise_affine).to(x.dtype)
    
    @classmethod
    def from_layernorm(cls, original_layernorm):
        layernorm_layer = cls(
            dim=original_layernorm.normalized_shape[0],
            eps=original_layernorm.eps,
            elementwise_affine=False if original_layernorm.weight is None else True,
            bias=original_layernorm.bias is not None
        )
        if original_layernorm.weight is not None and original_layernorm.weight.device != torch.device('meta'):
            layernorm_layer.weight.data.copy_(original_layernorm.weight.data)
        if original_layernorm.bias is not None and original_layernorm.bias.device != torch.device('meta'):
            layernorm_layer.bias.data.copy_(original_layernorm.bias.data)
        return layernorm_layer
