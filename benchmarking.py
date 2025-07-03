import os
import time
import torch
import numpy as np
from fastvideo.v1.attention import LocalAttention
from fastvideo.v1.platforms import AttentionBackendEnum
from fastvideo.v1.attention.selector import global_force_attn_backend
from fastvideo.v1.forward_context import set_forward_context

def test_attention_correctness():
    """Test that all attention backends produce the same results."""
    
    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("MPS is not available on this system")
        return
    
    print("Testing Attention Backend Correctness")
    print("=" * 50)
    
    # Set device
    device = torch.device("mps")
    
    # Test parameters
    batch_size = 2
    seq_len = 256  # Smaller for faster testing
    num_heads = 8
    head_dim = 64
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {head_dim}")
    
    # Create test tensors with fixed seed for reproducibility
    torch.manual_seed(42)
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    
    # Define backends to test
    backends = {
        "TORCH_SDPA": AttentionBackendEnum.TORCH_SDPA, 
    }
    
    results = {}
    
    for backend_name, backend_enum in backends.items():
        print(f"\nTesting {backend_name}...")
        
        # Force the specific backend
        global_force_attn_backend(backend_enum)
        
        # Create attention layer with the forced backend
        attention_layer = LocalAttention(
            num_heads=num_heads,
            head_size=head_dim,
            causal=False,
            supported_attention_backends=(backend_enum,)
        ).to(device)
        
        # Use forward context as a context manager
        with set_forward_context(current_timestep=0, attn_metadata=None):
            # Warm up
            for _ in range(2):
                with torch.no_grad():
                    _ = attention_layer(query, key, value)
            
            # Run attention
            with torch.no_grad():
                output = attention_layer(query, key, value)
        
        results[backend_name] = output
        print(f"{backend_name} output shape: {output.shape}")
        
        # Reset to auto-selection
        global_force_attn_backend(None)
    
    # Compare all results
    print(f"\nComparing backend outputs...")
    print("=" * 40)
    
    # Use MFA_ATTN as reference
    reference_output = results["MFA_ATTN"]
    
    for backend_name, output in results.items():
        if backend_name == "MFA_ATTN":
            continue
            
        diff = torch.abs(output - reference_output).max().item()
        print(f"{backend_name:12} vs MFA_ATTN: max diff = {diff:.8f}")
        
        if diff < 1e-3:
            print(f"✅ {backend_name:12}: Correct (within tolerance)")
        else:
            print(f"❌ {backend_name:12}: Incorrect (significant difference)")
    
    return results

def benchmark_attention_backends():
    """Benchmark SDPA attention backend on CPU vs MPS."""
    
    # Test parameters
    batch_size_q = 1
    seq_len_q = 768
    num_heads_q = 12
    head_dim_q = 128

    warmup_runs = 50  # More warmup runs
    benchmark_runs = 1000  # Actual benchmark runs
    
    print(f"\nBenchmarking SDPA attention backend: CPU vs MPS")
    print("=" * 60)
    print(f"Batch size query: {batch_size_q}")
    print(f"Sequence length query: {seq_len_q}")
    print(f"Number of heads query: {num_heads_q}")
    print(f"Head dimension query: {head_dim_q}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Benchmark runs: {benchmark_runs}")
    
    # Define devices to test
    devices = {
        "CPU": torch.device("cpu"),
        "MPS": torch.device("mps") if torch.backends.mps.is_available() else None
    }
    
    # Remove MPS if not available
    if devices["MPS"] is None:
        print("MPS is not available on this system, only testing CPU")
        devices.pop("MPS")
    
    results = {}
    
    for device_name, device in devices.items():
        print(f"\nTesting SDPA on {device_name}...")
        
        # Force SDPA backend
        global_force_attn_backend(AttentionBackendEnum.TORCH_SDPA)
        
        # Create test tensors on the specific device
        query = torch.randn(batch_size_q, seq_len_q, num_heads_q, head_dim_q, device=device, dtype=torch.float16)
        key = torch.randn(batch_size_q, seq_len_q, num_heads_q, head_dim_q, device=device, dtype=torch.float16)
        value = torch.randn(batch_size_q, seq_len_q, num_heads_q, head_dim_q, device=device, dtype=torch.float16)
        
        # Create attention layer with SDPA backend
        attention_layer = LocalAttention(
            num_heads=num_heads_q,
            head_size=head_dim_q,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA,)
        ).to(device)
        
        # Use forward context as a context manager
        with set_forward_context(current_timestep=0, attn_metadata=None):
            # Comprehensive warmup phase
            print(f"  Warming up with {warmup_runs} runs...")
            
            # Sync if using MPS
            if device.type == "mps":
                torch.mps.synchronize()
            
            for i in range(warmup_runs):
                with torch.no_grad():
                    _ = attention_layer(query, key, value)
                # Sync every 10 runs during warmup to ensure proper GPU utilization
                if device.type == "mps" and (i + 1) % 10 == 0:
                    torch.mps.synchronize()
            
            # Final sync after warmup
            if device.type == "mps":
                torch.mps.synchronize()
            
            # Actual benchmark phase
            print(f"  Running benchmark with {benchmark_runs} runs...")
            start_time = time.time()
            
            for i in range(benchmark_runs):
                with torch.no_grad():
                    _ = attention_layer(query, key, value)
            
            # Sync if using MPS
            if device.type == "mps":
                torch.mps.synchronize()
            end_time = time.time()
        
        # Calculate statistics
        total_time = (end_time - start_time) * 1000  # Convert to ms
        avg_time = total_time / benchmark_runs
        throughput = benchmark_runs / (total_time / 1000)  # ops per second
        
        results[device_name] = {
            'avg_time': avg_time,
            'total_time': total_time,
            'throughput': throughput
        }
        
        print(f"  {device_name} results:")
        print(f"    Average time: {avg_time:.4f} ms")
        print(f"    Total time: {total_time:.2f} ms")
        print(f"    Throughput: {throughput:.0f} ops/sec")
    
    # Reset to auto-selection
    global_force_attn_backend(None)
    
    # Print comparison
    print(f"\nBenchmark Results Summary:")
    print("=" * 40)
    fastest_avg = min(result['avg_time'] for result in results.values())
    
    print(f"{'Device':<8} {'Avg Time':<10} {'Speedup':<8} {'Throughput':<12}")
    print("-" * 45)
    
    for device, result in results.items():
        speedup = fastest_avg / result['avg_time']
        print(f"{device:<8} {result['avg_time']:<10.4f} {speedup:<8.2f}x {result['throughput']:<12.0f}")
    
    # Find the fastest device
    fastest_device = min(results.items(), key=lambda x: x[1]['avg_time'])[0]
    print(f"\nFastest device: {fastest_device}")
    
    return results

if __name__ == "__main__":
    print("SDPA Attention Benchmark: CPU vs MPS")
    print("=" * 50)
    
    # Benchmark SDPA on different devices
    benchmark_attention_backends()
    
    print("\nBenchmark completed!") 