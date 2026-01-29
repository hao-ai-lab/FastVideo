import torch
import sys
import traceback
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import numpy as np
from typing import Dict, List, Tuple, Optional

from flash_attn import flash_attn_func
from fastvideo.attention.backends.sageattn.quantization.bench.bench_utils import bench

# Import SageAttn components for direct control
from fastvideo.attention.backends.sageattn.api import (
    preprocess_qkv, 
    scale_and_quant_fp4, 
    scale_and_quant_fp4_permute, 
    scale_and_quant_fp4_transpose,
    blockscaled_fp4_attn
)


def calculate_attention_flops(batch_size, num_heads, seq_len_q, seq_len_k, head_dim, is_causal=False):
    """Calculate FLOPs for attention (FlashAttention standard - matmuls only)."""
    f = 4 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
    if is_causal:
        f = f // 2
    return f


def sageattn_blackwell_configurable(q, k, v, is_causal=False, per_block_mean=True, 
                                     single_level_p_quant=True, 
                                     enable_smoothing_q=False, enable_smoothing_k=False):
    """
    Configurable SageAttention3 Blackwell kernel with explicit smoothing control.
    
    Args:
        q: Query tensor [B, H, L, D]
        k: Key tensor [B, H, L, D]
        v: Value tensor [B, H, L, D]
        is_causal: Whether to use causal masking
        per_block_mean: Whether to use per-block mean for Q smoothing
        single_level_p_quant: If True, use single-level quantization for P matrix
        enable_smoothing_q: Enable Q smoothing
        enable_smoothing_k: Enable K smoothing
    
    Returns:
        Output tensor [B, H, L, D]
    """
    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16
    
    # Preprocess with explicit smoothing control
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean, enable_smoothing_q, enable_smoothing_k)
    
    qlist_from_cuda = scale_and_quant_fp4(q)
    klist_from_cuda = scale_and_quant_fp4_permute(k)
    vlist_from_cuda = scale_and_quant_fp4_transpose(v)
    
    o_fp4 = blockscaled_fp4_attn(
        qlist_from_cuda,
        klist_from_cuda, 
        vlist_from_cuda,
        delta_s,
        KL,
        is_causal,
        per_block_mean,
        is_bf16,
        single_level_p_quant
    )[0][:, :, :QL, :].contiguous()
    
    return o_fp4


def benchmark_flashattn2(batch_size, num_heads, seq_len, head_dim, 
                         is_causal=False, dtype=torch.bfloat16, 
                         num_warmups=10, num_tests=50):
    """Benchmark FlashAttention2."""
    device = 'cuda'
    
    # FlashAttention2 expects (batch, seq_len, num_heads, head_dim)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    def run_attention():
        return flash_attn_func(q, k, v, causal=is_causal)
    
    avg_time_ms = bench(run_attention, num_warmups=num_warmups, num_tests=num_tests)
    return avg_time_ms


def benchmark_sageattn3(batch_size, num_heads, seq_len, head_dim,
                        is_causal=False, dtype=torch.bfloat16,
                        per_block_mean=True, single_level_p_quant=False,
                        enable_smoothing_q=True, enable_smoothing_k=True,
                        num_warmups=10, num_tests=50):
    """Benchmark SageAttention3 with configurable smoothing."""
    device = 'cuda'
    
    # SageAttn expects (batch, num_heads, seq_len, head_dim)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    def run_attention():
        return sageattn_blackwell_configurable(
            q, k, v, 
            is_causal=is_causal,
            per_block_mean=per_block_mean,
            single_level_p_quant=single_level_p_quant,
            enable_smoothing_q=enable_smoothing_q,
            enable_smoothing_k=enable_smoothing_k
        )
    
    avg_time_ms = bench(run_attention, num_warmups=num_warmups, num_tests=num_tests)
    return avg_time_ms


def time_to_tops(time_ms, batch_size, num_heads, seq_len, head_dim, is_causal=False):
    """Convert time to TOPS (Tera Operations Per Second)."""
    total_flops = calculate_attention_flops(batch_size, num_heads, seq_len, seq_len, head_dim, is_causal)
    time_s = time_ms / 1000.0
    tops = total_flops / (time_s * 1e12)
    return tops


def run_benchmark_suite(head_dim=64, is_causal=False, num_heads=12, batch_size=1,
                        num_warmups=10, num_tests=50, 
                        seq_lens=None, output_file="benchmark_attention.png"):
    """
    Run comprehensive benchmark suite and generate plot.
    
    Args:
        head_dim: Head dimension (64 or 128)
        is_causal: Whether to use causal attention
        num_heads: Number of attention heads
        batch_size: Batch size
        num_warmups: Number of warmup iterations
        num_tests: Number of test iterations
        seq_lens: List of sequence lengths to test
        output_file: Output plot filename
    """
    if seq_lens is None:
        seq_lens = [1024, 2048, 4096, 8192, 16384, 32768]
    
    device_name = torch.cuda.get_device_name(0)
    # Extract short name (e.g., "RTX5090" from full name)
    short_name = device_name.split()[-1] if 'RTX' in device_name or 'A100' in device_name else device_name[:20]
    
    print(f"Starting Combined Attention Benchmark Suite...")
    print(f"CUDA Device: {device_name}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Head Dim: {head_dim}, Causal: {is_causal}, Num Heads: {num_heads}, Batch Size: {batch_size}")
    print("="*80)
    sys.stdout.flush()
    
    # Results storage: {method_name: {seq_len: tops}}
    results: Dict[str, Dict[int, Optional[float]]] = {
        'FlashAttn': {},
        'SageAttn3': {},
        'FP4': {},
    }
    
    dtype = torch.bfloat16
    
    for seq_len in seq_lens:
        print(f"\n--- Sequence Length: {seq_len} ---")
        sys.stdout.flush()
        
        # FlashAttention2
        print(f"  Benchmarking FlashAttn2...", end=" ")
        sys.stdout.flush()
        try:
            time_ms = benchmark_flashattn2(
                batch_size, num_heads, seq_len, head_dim,
                is_causal=is_causal, dtype=dtype,
                num_warmups=num_warmups, num_tests=num_tests
            )
            tops = time_to_tops(time_ms, batch_size, num_heads, seq_len, head_dim, is_causal)
            results['FlashAttn'][seq_len] = tops
            print(f"{tops:.0f} TOPS ({time_ms:.3f} ms)")
        except Exception as e:
            print(f"OOM or Error: {e}")
            results['FlashAttn'][seq_len] = None
        sys.stdout.flush()
        
        # SageAttn3 (with smoothing: single_level_p_quant=False, enable_smoothing_q=True, enable_smoothing_k=True)
        print(f"  Benchmarking SageAttn3 (smoothing ON)...", end=" ")
        sys.stdout.flush()
        try:
            time_ms = benchmark_sageattn3(
                batch_size, num_heads, seq_len, head_dim,
                is_causal=is_causal, dtype=dtype,
                per_block_mean=True,
                single_level_p_quant=False,
                enable_smoothing_q=True,
                enable_smoothing_k=True,
                num_warmups=num_warmups, num_tests=num_tests
            )
            tops = time_to_tops(time_ms, batch_size, num_heads, seq_len, head_dim, is_causal)
            results['SageAttn3'][seq_len] = tops
            print(f"{tops:.0f} TOPS ({time_ms:.3f} ms)")
        except Exception as e:
            print(f"OOM or Error: {e}")
            results['SageAttn3'][seq_len] = None
        sys.stdout.flush()
        
        # FP4 (no smoothing: single_level_p_quant=True, enable_smoothing_q=False, enable_smoothing_k=False)
        print(f"  Benchmarking FP4 (smoothing OFF)...", end=" ")
        sys.stdout.flush()
        try:
            time_ms = benchmark_sageattn3(
                batch_size, num_heads, seq_len, head_dim,
                is_causal=is_causal, dtype=dtype,
                per_block_mean=True,
                single_level_p_quant=True,
                enable_smoothing_q=False,
                enable_smoothing_k=False,
                num_warmups=num_warmups, num_tests=num_tests
            )
            tops = time_to_tops(time_ms, batch_size, num_heads, seq_len, head_dim, is_causal)
            results['FP4'][seq_len] = tops
            print(f"{tops:.0f} TOPS ({time_ms:.3f} ms)")
        except Exception as e:
            print(f"OOM or Error: {e}")
            results['FP4'][seq_len] = None
        sys.stdout.flush()
    
    # Print summary table
    print("\n" + "="*100)
    print("Summary Table (TOPS)")
    print("="*100)
    header = f"{'SeqLen':<10}"
    for method in results.keys():
        header += f"{method:<15}"
    print(header)
    print("-"*100)
    
    for seq_len in seq_lens:
        row = f"{seq_len:<10}"
        for method in results.keys():
            val = results[method].get(seq_len)
            if val is not None:
                row += f"{val:<15.0f}"
            else:
                row += f"{'OOM':<15}"
        print(row)
    print("="*100)
    sys.stdout.flush()
    
    # Generate plot
    generate_plot(results, seq_lens, head_dim, is_causal, short_name, output_file)
    
    return results


def generate_plot(results: Dict[str, Dict[int, Optional[float]]], 
                  seq_lens: List[int], 
                  head_dim: int, 
                  is_causal: bool,
                  device_name: str,
                  output_file: str):
    """Generate bar plot comparing attention implementations."""
    
    # Prepare data
    methods = list(results.keys())
    x_labels = [f"{sl//1024}K" for sl in seq_lens]
    
    # Colors for each method (matching the reference image style)
    colors = {
        'FlashAttn': '#FFA500',   # Orange
        'SageAttn3': '#228B22',   # Forest Green
        'FP4': '#32CD32',         # Lime Green (lighter than SageAttn3)
    }
    
    # Number of methods and positions
    n_methods = len(methods)
    n_positions = len(seq_lens)
    
    # Bar width and positions
    bar_width = 0.25
    x = np.arange(n_positions)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        values = []
        for seq_len in seq_lens:
            val = results[method].get(seq_len)
            values.append(val if val is not None else 0)
        
        offset = (i - n_methods/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width, 
                      label=method, color=colors.get(method, f'C{i}'),
                      edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        for bar, val, seq_len in zip(bars, values, seq_lens):
            if results[method].get(seq_len) is None:
                label = 'OOM'
            else:
                label = f'{int(val)}'
            
            height = bar.get_height()
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8, rotation=0)
    
    # Customize plot
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Speed (TOPS)', fontsize=12)
    ax.set_title(f'{device_name}, (Head dim = {head_dim}, causal = {is_causal})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc='upper left', ncol=len(methods), fontsize=10)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Also save as PDF for high quality
    pdf_file = output_file.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"PDF saved to: {pdf_file}")
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combined Attention Benchmark (FlashAttn2 vs SageAttn3 vs FP4)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--head-dim', type=int, default=64, choices=[64, 128], help='Head dimension')
    parser.add_argument('--causal', action='store_true', help='Use causal attention')
    parser.add_argument('--num-warmups', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--num-tests', type=int, default=50, help='Number of test iterations')
    parser.add_argument('--seq-lens', type=int, nargs='+', 
                        default=[1024, 2048, 4096, 8192, 16384, 32768],
                        help='Sequence lengths to benchmark')
    parser.add_argument('--output', type=str, default='benchmark_attention.png',
                        help='Output plot filename')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a CUDA device.")
    
    run_benchmark_suite(
        head_dim=args.head_dim,
        is_causal=args.causal,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        num_warmups=args.num_warmups,
        num_tests=args.num_tests,
        seq_lens=args.seq_lens,
        output_file=args.output
    )

