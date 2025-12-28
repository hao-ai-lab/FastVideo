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
    if quant_cuda is None:
        raise ImportError("fastvideo_kernel_ops not compiled or loaded, quant_cuda unavailable")
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
    if gemm_cuda is None:
        raise ImportError("fastvideo_kernel_ops not compiled or loaded, gemm_cuda unavailable")
        
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
        (w, rstd, num_warps) (tuple): RMSNorm weight tensor, rstd tensor, and number of warps.
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

    return y

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

    return y


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

    return y

def layernorm(x, w, b, eps, elementwise_affine=True):
    if elementwise_affine:
        assert w is not None and b is not None
        return layernorm_param(x, w, b, eps)
    else:
        assert w is None and b is None
        return layernorm_noparam(x, eps)

def cdiv(a: int, b: int):
    return (a + b - 1) // b

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
        out = int8_linear(x, self.int8_weight, self.scale)
        if self.bias is not None:
            out = out + self.bias
        return out

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
        return rmsnorm(x.float(), self.weight, self.eps).to(x.dtype)

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
        return layernorm(x.float(), self.weight, self.bias, self.eps, self.elementwise_affine).to(x.dtype)
    
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

