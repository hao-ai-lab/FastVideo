import torch
import argparse
from flash_attn.utils.benchmark import benchmark_forward
from st_attn import mha_forward, mha_backward

import numpy as np
import random

def set_seed(seed: int = 42):
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

def parse_arguments():
    parser = argparse.ArgumentParser(description='Benchmark MHA Attention')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of heads')
    parser.add_argument('--head_dim', type=int, default=64, help='Head dimension')
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[49152], help='Sequence lengths to benchmark')
    return parser.parse_args()

def create_input_tensors(batch, head, seq_len, headdim):
    """Create random input tensors for attention."""
    q = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    return q, k, v

def benchmark_mha(q, k, v, flops):
    """Benchmark MHA (Multi-Head Attention) forward and backward passes."""
    print("\n=== MHA BENCHMARK ===")
    
    # Forward pass
    # Warm-up run
    o, l_vec = mha_forward(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark forward
    _, fwd_time = benchmark_forward(
        mha_forward,
        q, k, v,
        repeats=20,
        verbose=False,
        desc='MHA Forward'
    )
    
    mha_tflops = flops / fwd_time.mean * 1e-12
    print(f"MHA Forward - TFLOPS: {mha_tflops:.2f}")
    
    # Backward pass
    grad_output = torch.randn_like(o)
    
    # Warm-up runs
    for _ in range(5):
        mha_backward(q, k, v, o, l_vec, grad_output)
    torch.cuda.synchronize()
    
    # Benchmark backward
    _, bwd_time = benchmark_forward(
        mha_backward,
        q, k, v, o, l_vec, grad_output,
        repeats=20,
        verbose=False,
        desc='MHA Backward'
    )
    bwd_flops = 2.5 * flops  # Approximation

    mha_bwd_tflops = bwd_flops / bwd_time.mean * 1e-12
    print(f"MHA Backward - TFLOPS: {mha_bwd_tflops:.2f}")
    
    return mha_tflops, mha_bwd_tflops

def main():
    args = parse_arguments()

    set_seed(42)
    
    # Extract parameters
    batch = args.batch_size
    head = args.num_heads
    headdim = args.head_dim
    
    print(f"MHA Benchmark")
    print(f"batch: {batch}, head: {head}, headdim: {headdim}")
    
    # Test with different sequence lengths
    for seq_len in args.seq_lengths:
        # Skip very long sequences if they might cause OOM
        if seq_len > 16384 and batch > 1:
            continue
            
        print("="*100)
        print(f"\nSequence length: {seq_len}")
        
        # Calculate theoretical FLOPs for attention
        flops = 4 * batch * head * headdim * seq_len * seq_len
        
        # Create input tensors
        q, k, v = create_input_tensors(batch, head, seq_len, headdim)
        
        # Benchmark MHA
        mha_fwd, mha_bwd = benchmark_mha(q, k, v, flops)
        
        # Print results
        print("\n=== PERFORMANCE RESULTS ===")
        print(f"MHA Forward - TFLOPS: {mha_fwd:.2f}")
        print(f"MHA Backward - TFLOPS: {mha_bwd:.2f}")

if __name__ == "__main__":
    main()