import sys
import traceback

import _bootstrap  # noqa: F401
import torch

from attn_qat_infer.api import sageattn_blackwell
from attn_qat_infer.quantization.bench.bench_utils import bench


def calculate_attention_flops(batch_size, num_heads, seq_len_q, seq_len_k, head_dim, is_causal=False):
    """Calculate FLOPs for attention (FlashAttention standard - matmuls only)."""
    f = 4 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
    if is_causal:
        f = f // 2
    return f


def benchmark_sageattn3(batch_size, num_heads, seq_len, head_dim, 
                        is_causal=False, dtype=torch.bfloat16, 
                        per_block_mean=True, single_level_p_quant=False,
                        num_warmups=100, num_tests=1000):
    """
    Benchmark SageAttention3 and return performance metrics.
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length (same for Q, K, V)
        head_dim: Head dimension
        is_causal: Whether to use causal masking
        dtype: Data type (torch.bfloat16 or torch.float16)
        per_block_mean: Whether to use per-block mean for Q smoothing
        single_level_p_quant: If True, use single-level quantization for P matrix
        num_warmups: Number of warmup iterations
        num_tests: Number of test iterations
    
    Returns:
        dict with performance metrics
    """
    device = 'cuda'
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a CUDA device.")
    
    # Create input tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    device=device, dtype=dtype)
    
    # Create closure for benchmarking (no extra stream needed - bench handles synchronization)
    def run_attention():
        return sageattn_blackwell(
            q, k, v, 
            is_causal=is_causal,
            per_block_mean=per_block_mean,
            single_level_p_quant=single_level_p_quant
        )
    
    # Benchmark using the bench utility (handles warmup and timing)
    avg_time_ms = bench(run_attention, num_warmups=num_warmups, num_tests=num_tests)
    avg_time_s = avg_time_ms / 1000.0
    
    # Calculate FLOPs
    total_flops = calculate_attention_flops(
        batch_size, num_heads, seq_len, seq_len, head_dim, is_causal
    )
    
    # Calculate TFLOPs
    tflops = total_flops / (avg_time_s * 1e12)
    
    # Calculate throughput (tokens/sec)
    tokens_per_second = (batch_size * seq_len) / avg_time_s
    
    return {
        'batch_size': batch_size,
        'num_heads': num_heads,
        'seq_len': seq_len,
        'head_dim': head_dim,
        'is_causal': is_causal,
        'dtype': str(dtype),
        'per_block_mean': per_block_mean,
        'single_level_p_quant': single_level_p_quant,
        'avg_time_ms': avg_time_ms,
        'avg_time_s': avg_time_s,
        'total_flops': total_flops,
        'tflops': tflops,
        'tokens_per_second': tokens_per_second,
    }


def print_results(results):
    """Print benchmark results in a formatted table."""
    print("\n" + "="*100)
    print("SageAttention3 Benchmark Results")
    print("="*100)
    print(f"Configuration:")
    print(f"  Batch Size:         {results['batch_size']}")
    print(f"  Num Heads:          {results['num_heads']}")
    print(f"  Sequence Length:    {results['seq_len']}")
    print(f"  Head Dimension:     {results['head_dim']}")
    print(f"  Causal:             {results['is_causal']}")
    print(f"  Data Type:          {results['dtype']}")
    print(f"  Per Block Mean:     {results['per_block_mean']}")
    print(f"  Single Level P Quant: {results['single_level_p_quant']}")
    print(f"\nPerformance:")
    print(f"  Average Time:       {results['avg_time_ms']:.3f} ms")
    print(f"  Total FLOPs:        {results['total_flops']/1e12:.4f} TFLOPs (theoretical)")
    print(f"  Throughput:         {results['tflops']:.4f} TFLOPs/s")
    print(f"  Tokens/sec:         {results['tokens_per_second']:,.0f}")
    print("="*100 + "\n")
    sys.stdout.flush()


def run_benchmark_suite():
    """Run a comprehensive benchmark suite with various configurations."""
    print("Starting SageAttention3 Benchmark Suite...")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}\n")
    sys.stdout.flush()
    
    # Default configurations to test
    # (batch_size, num_heads, seq_len, head_dim, is_causal, dtype)
    configs = [
        (1, 16, 1024, 64, False, torch.bfloat16),
        (1, 16, 2048, 64, False, torch.bfloat16),
        (1, 16, 4096, 64, False, torch.bfloat16),
        (1, 16, 8192, 64, False, torch.bfloat16),
        (1, 16, 16384, 64, False, torch.bfloat16),

        (1, 16, 1024, 128, False, torch.bfloat16),
        (1, 16, 2048, 128, False, torch.bfloat16),
        (1, 16, 4096, 128, False, torch.bfloat16),
        (1, 16, 8192, 128, False, torch.bfloat16),
        (1, 16, 16384, 128, False, torch.bfloat16),

        (1, 32, 1024, 64, False, torch.bfloat16),
        (1, 32, 2048, 64, False, torch.bfloat16),
        (1, 32, 4096, 64, False, torch.bfloat16),
        (1, 32, 8192, 64, False, torch.bfloat16),
        (1, 32, 16384, 64, False, torch.bfloat16),

        (1, 32, 1024, 128, False, torch.bfloat16),
        (1, 32, 2048, 128, False, torch.bfloat16),
        (1, 32, 4096, 128, False, torch.bfloat16),
        (1, 32, 8192, 128, False, torch.bfloat16),
        (1, 32, 16384, 128, False, torch.bfloat16),
    ]
    
    all_results = []
    
    for config in configs:
        batch_size, num_heads, seq_len, head_dim, is_causal, dtype = config
        
        print(f"\nBenchmarking: B={batch_size}, H={num_heads}, L={seq_len}, D={head_dim}, "
              f"Causal={is_causal}, dtype={dtype}...")
        sys.stdout.flush()
        
        try:
            results = benchmark_sageattn3(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                is_causal=is_causal,
                dtype=dtype,
                num_warmups=10,
                num_tests=50
            )
            
            print_results(results)
            all_results.append(results)
            
        except Exception as e:
            print(f"Error benchmarking configuration {config}:")
            print(f"  Exception: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            continue
    
    # Print summary table
    print("\n" + "="*120)
    print("Summary Table")
    print("="*120)
    print(f"{'B':<4} {'H':<4} {'L':<6} {'D':<4} {'Causal':<7} {'Time (ms)':<12} {'TFLOPs/s':<12} {'Tokens/s':<15}")
    print("-"*120)
    
    for r in all_results:
        print(f"{r['batch_size']:<4} {r['num_heads']:<4} {r['seq_len']:<6} {r['head_dim']:<4} "
              f"{str(r['is_causal']):<7} {r['avg_time_ms']:<12.3f} {r['tflops']:<12.4f} "
              f"{r['tokens_per_second']:<15,.0f}")
    
    print("="*120 + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark SageAttention3 in TFLOPs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--num-heads', type=int, default=None, help='Number of attention heads')
    parser.add_argument('--seq-len', type=int, default=None, help='Sequence length')
    parser.add_argument('--head-dim', type=int, default=None, help='Head dimension')
    parser.add_argument('--causal', action='store_true', help='Use causal attention')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'],
                        help='Data type')
    parser.add_argument('--per-block-mean', action='store_true', default=True,
                        help='Use per-block mean for Q smoothing (default: True)')
    parser.add_argument('--no-per-block-mean', action='store_false', dest='per_block_mean',
                        help='Disable per-block mean for Q smoothing')
    parser.add_argument('--single-level-p-quant', action='store_true', default=False,
                        help='Use single-level P quantization (default: True)')
    parser.add_argument('--two-level-p-quant', action='store_false', dest='single_level_p_quant',
                        help='Use two-level P quantization')
    parser.add_argument('--num-warmups', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--num-tests', type=int, default=50, help='Number of test iterations')
    parser.add_argument('--suite', action='store_true', help='Run full benchmark suite')
    
    args = parser.parse_args()
    
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }
    
    if args.suite:
        run_benchmark_suite()
    elif args.batch_size and args.num_heads and args.seq_len and args.head_dim:
        results = benchmark_sageattn3(
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            is_causal=args.causal,
            dtype=dtype_map[args.dtype],
            per_block_mean=args.per_block_mean,
            single_level_p_quant=args.single_level_p_quant,
            num_warmups=args.num_warmups,
            num_tests=args.num_tests
        )
        print_results(results)
    else:
        print("Running default benchmark suite. Use --suite for full suite or provide all parameters.")
        print(
            "Example: python benchmarks/benchmark_sageattn3.py "
            "--batch-size 1 --num-heads 16 --seq-len 4096 --head-dim 128"
        )
        sys.stdout.flush()
        run_benchmark_suite()
