# modified from https://github.com/neuralmagic/AutoFP8/blob/main/auto_fp8/quantize.py
import gc
from typing import Tuple
import copy
import torch
import tqdm
import triton
import triton.language as tl



def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()


def per_tensor_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize a tensor using per-tensor static scaling factor.
    Args:
        tensor: The input tensor.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    # Calculate the scale as dtype max divided by absmax.
    # Since .abs() creates a new tensor, we use aminmax to get
    # the min and max first and then calculate the absmax.
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        min_val, max_val = (
            torch.tensor(-16.0, dtype=tensor.dtype),
            torch.tensor(16.0, dtype=tensor.dtype),
        )
    else:
        min_val, max_val = tensor.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = finfo.max / amax.clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float().reciprocal()
    return qweight, scale


fp8_gemm_configs = [
    triton.Config({'BLOCK_SIZE_M': block_m, 
                   'BLOCK_SIZE_N': block_n, 
                   'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]
@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_scale, b_scale,  # 改为单个scale值
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
        
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for i in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i, other=0.0)
        
        accumulator += tl.dot(a, b) * a_scale * b_scale
        
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def triton_fp8_gemm(a: torch.Tensor, 
                    b: torch.Tensor, 
                    a_scale: float, 
                    b_scale: float, 
                    out_dtype=torch.bfloat16, 
                    bias=None) -> torch.Tensor:
    """
    Perform a matrix multiplication using FP8 precision with per-tensor quantization.
    """
    assert a.is_contiguous() and b.is_contiguous()
    
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    if isinstance(a_scale, torch.Tensor):
        a_scale = a_scale.item()
    if isinstance(b_scale, torch.Tensor):
        b_scale = b_scale.item()
    # import pdb; pdb.set_trace()
    fp8_gemm_kernel[grid](a, b, c, a_scale, b_scale, M, N, K)
    if bias is not None:
        
        c += bias
    
    return c


def fp8_gemm(A, A_scale, B, B_scale, bias, out_dtype, native_fp8_support=False):
    """
    Optimized FP8 GEMM implementation, supports both native FP8 and Triton paths, 
    and automatically handles 3D input and bias.
    """
    if A.numel() == 0:
        # Handle empty tensor (e.g., when MoE expert is empty)
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)

    # Check if reshape is needed (support for 3D input)
    need_reshape = (A.dim() == 3)
    batch_size = A.shape[0] if need_reshape else None
    A_input = A.reshape(-1, A.shape[-1]).contiguous() if need_reshape else A

    if native_fp8_support:
        # Native FP8 support
        output = torch._scaled_mm(
            A_input,
            B.t(),
            out_dtype=out_dtype,
            scale_a=torch.tensor(A_scale) if not isinstance(A_scale, torch.Tensor) else A_scale,
            scale_b=torch.tensor(B_scale) if not isinstance(B_scale, torch.Tensor) else B_scale,
            bias=bias.to(out_dtype),
        )
    else:
        # Triton implementation
        output = triton_fp8_gemm(
            A_input,
            B.contiguous(),
            out_dtype=out_dtype,
            a_scale=A_scale,
            b_scale=B_scale,
            bias=None,
        )
        if bias is not None:
            output += bias

    if need_reshape:
        # Restore original batch dimension
        output = output.reshape(batch_size, -1, output.shape[-1])

    return output


# Class responsible for quantizing weights
class FP8DynamicLinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
        native_fp8_support: bool = False,
        name: str = ""
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias
        self.native_fp8_support = native_fp8_support
        self.name = name
    # @torch.compile
    def forward(self, x):
        if x.dtype != torch.float16 and x.dtype != torch.bfloat16:
            # print(f"Warning: {self.name}'s input is not quantized to float16 or bfloat16")
            # print(f"input dtype: {x.dtype}")
            x = x.to(torch.bfloat16)
        qinput, x_scale = per_tensor_quantize(x)
        # print("--------------")
        # print("layer_name:", self.name)
        # print("A_input.shape:", qinput.shape)
        # print("B.shape:", self.weight.shape)
        # print("--------------")
        output = fp8_gemm(
            A=qinput,
            A_scale=x_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
            native_fp8_support=self.native_fp8_support,
        )
        return output


def replace_module(model: torch.nn.Module, name: str, new_module: torch.nn.Module):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1 :]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model
        child_name = name
    setattr(parent, child_name, new_module)


def convert_fp8_linear(model: torch.nn.Module):
    # native_fp8_support = (
    #     torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)
    # )
    native_fp8_support = False
    named_modules = list(model.named_modules())
    for name, linear in tqdm.tqdm(named_modules, desc="Quantizing weights"):
        if not isinstance(linear, torch.nn.Linear):
            continue
        if "mod" in name:
            print(f"Warning: {name} is a mod module, skipping")
            continue
        if "block" not in name:
            print(f"Warning: {name} is not in a block module, skipping")
            continue
        quant_weight, weight_scale = per_tensor_quantize(linear.weight)
        bias = copy.deepcopy(linear.bias) if linear.bias is not None else None
        quant_linear = FP8DynamicLinear(
            weight=quant_weight, 
            weight_scale=weight_scale, 
            bias=bias, 
            native_fp8_support=native_fp8_support, 
            name = name
        )
        replace_module(model, name, quant_linear)
        del linear.weight
        del linear.bias
        del linear
    cleanup_memory()
