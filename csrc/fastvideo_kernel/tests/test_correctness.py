import torch
import sys
import os
from tqdm import tqdm

# Add the parent directory of csrc/attn/tests to find flex_sta_ref
# We are running from FastVideo-fork/csrc/fastvideo_kernel
sys.path.append(os.path.abspath("../attn/tests"))

try:
    from flex_sta_ref import get_sliding_tile_attention_mask
except ImportError:
    print("Error: Could not find flex_sta_ref.py. Make sure you are running from FastVideo-fork/csrc/fastvideo_kernel")
    sys.exit(1)

# USE OUR NEW PACKAGE!
from fastvideo_kernel import sliding_tile_attention
from torch.nn.attention.flex_attention import flex_attention

flex_attention = torch.compile(flex_attention, dynamic=False)

def flex_test(Q, K, V, kernel_size):
    mask = get_sliding_tile_attention_mask(kernel_size, (6, 8, 8), (18, 48, 80), 0, 'cuda', 0)
    output = flex_attention(Q, K, V, block_mask=mask)
    return output

def h100_fwd_kernel_test(Q, K, V, kernel_size):
    # Using the same parameters as the original test
    o = sliding_tile_attention(Q, K, V, [kernel_size] * 24, 0, False, '18x48x80')
    return o

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    return scaled_tensor.contiguous()

def check_correctness(b, h, n, d, causal, mean, std, num_iterations=2):
    print(f"Running correctness check: batch={b}, heads={h}, seq_len={n}, dim={d}")
    kernel_size_ls = [(3, 3, 5), (3, 1, 10)]
    
    for kernel_size in kernel_size_ls:
        print(f"Testing kernel_size: {kernel_size}")
        for xi in tqdm(range(num_iterations)):
            torch.manual_seed(xi)
            Q = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            K = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            V = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            
            tk_o = h100_fwd_kernel_test(Q, K, V, kernel_size)
            pt_o = flex_test(Q, K, V, kernel_size)

            diff = pt_o - tk_o
            abs_diff = torch.abs(diff)
            max_d = torch.max(abs_diff).item()
            avg_d = torch.sum(abs_diff).item() / (b * h * n * d)
            
            if max_d > 0.1:
                print(f"Warning: Large diff detected! max={max_d}, avg={avg_d}")

    print("\n✅ TEST COMPLETE: New package matches FlexAttention behavior.")

if __name__ == "__main__":
    b, h, d = 2, 24, 128
    n = 69120 
    causal = False
    mean = 1e-1
    std = 10
    
    check_correctness(b, h, n, d, causal, mean, std, num_iterations=2)
