import vllm
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from quant_layer import FP8Linear

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import time


def quant_layer_refactor_(submodule,name,parent_module,quant_config,full_name):

    quant_layer_type = FP8Linear

    input_size = submodule.in_features
    output_size = submodule.out_features
    bias = True if submodule.bias is not None else False
    skip_bias_add = False if submodule.bias is not None else True
    params_dtype = submodule.weight.dtype
    prefix = full_name
    setattr(parent_module, name, quant_layer_type(input_size, output_size, bias=bias, skip_bias_add=skip_bias_add, params_dtype=params_dtype, quant_config=quant_config, prefix=prefix).cuda())
        
def apply_func_to_submodules(module, class_type, function, parent_name="", **kwargs):
    """
    Recursively iterates through all submodules of a PyTorch module and applies a hook function
    if the submodule matches the specified class type. The parent name is appended to the submodule name.

    Args:
        module (torch.nn.Module): The PyTorch module to iterate through.
        class_type (type): The class type to match against submodules.
        function (callable): The function to apply if a submodule matches the class type.
        parent_name (str): The name of the parent module (used for recursion).
    """

    for name, submodule in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        parent_module = module

        # INFO: pass from the parent call into func
        if 'name' in kwargs:
            kwargs['name']=name
        if 'full_name' in kwargs:
            kwargs['full_name'] = full_name
        if 'parent_module' in kwargs:
            kwargs['parent_module'] = module
        if isinstance(submodule, class_type):
            function(submodule, **kwargs)

        # Recursively apply the function to submodules
        apply_func_to_submodules(submodule, class_type, function, parent_name=full_name, **kwargs)
        
class LLaMA3FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        """
        Implements the feedforward network (FFN) used in LLaMA 3, which includes:
        - Gated Linear Unit (GLU)
        - SwiGLU activation
        - Projected down to input dim

        Args:
        - dim (int): The input dimension (e.g., 4096 for LLaMA 3 8B).
        - hidden_dim (int): Expanded dimension for FFN (typically 4x of dim, rounded to multiple_of).
        - multiple_of (int): Hidden dim is rounded to the nearest multiple of this (default 256).
        """
        super().__init__()

        # Ensure hidden_dim is a multiple of `multiple_of`
        hidden_dim = ((hidden_dim + multiple_of - 1) // multiple_of) * multiple_of
        
        # First projection (GLU mechanism with SwiGLU activation)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Gating mechanism

    def forward(self, x):
        """
        Forward pass of the LLaMA 3 FFN.
        """
        w1_out, _ = self.w1(x)
        w3_out, _ = self.w3(x)
        out, _ = self.w2(F.silu(w1_out) * w3_out)
        return out

from fastvideo.models.stepvideo.modules.blocks import FeedForward
class MultiLayerLLaMA3FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_layers: int, multiple_of: int = 256):
        """
        Implements multiple stacked FFN layers.

        Args:
        - dim (int): Input dimension.
        - hidden_dim (int): Hidden layer dimension (expanded dim).
        - num_layers (int): Number of FFN layers to stack.
        - multiple_of (int): Ensures hidden dim is a multiple of this.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            FeedForward(dim, hidden_dim, dim, False) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)  # Add normalization for stability

    def forward(self, x):
        for layer in self.layers:
            x = self.norm(x + layer(x))  # Residual connection
        return x

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    # Initialize model and input tensor
    dim = 6144  # Example input dimension for LLaMA 3 8B
    hidden_dim = 24576  # Hidden dim, rounded to nearest multiple of 256
    num_layers = 48
    ffn = MultiLayerLLaMA3FFN(dim, hidden_dim, num_layers).cuda() 
    quant_config = Fp8Config(activation_scheme="dynamic", ignored_layers=None)
    apply_func_to_submodules(ffn, class_type=nn.Linear, function=quant_layer_refactor_, name=None, parent_module=None, quant_config=quant_config, full_name=None)
    print(ffn)
    # ckpt = torch.load("fp_weight.pt")
    # missing, unexpected = ffn.load_state_dict(ckpt, strict=False)
    # print("missing: ", missing)
    # print("unexpected: ", unexpected)

    # Quantizing weight
    for name, submodule in ffn.named_modules():
        if isinstance(submodule, FP8Linear):
            submodule.quant_method.process_weights_after_loading(submodule)
    # Input
    x = torch.randn(2, 50592, 6144).cuda()  # (batch, seq_len, dim) - batch 32, seq 128
    
    # ------------------------ PROFILING ------------------------
    print("FP8")
    # Measure latency and throughput
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in tqdm(range(50)):  # Run multiple times for better estimates
            _ = ffn(x)
    torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / 50  # Average latency
    
    print(f"Latency per forward pass: {elapsed_time * 1000:.3f} ms")
    print(f"Throughput: {32*128 / elapsed_time:.3f} samples/sec")

    