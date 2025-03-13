import torch
import time
import math
from einops import rearrange
from functools import partial, lru_cache
from torch.nn.attention.flex_attention import flex_attention
from csrc.sliding_tile_attention.test.sba import get_sliding_block_attention_mask
import tk_4090_cuda as tk
from fastvideo.models.flash_attn_no_pad import flash_attn_no_pad
import torch.nn.functional as F
# Import benchmark utilities from flash_attn
from flash_attn.utils.benchmark import benchmark_forward
torch._dynamo.config.cache_size_limit = 64

@lru_cache(maxsize=32)
def get_compiled_flex_attention(strategy, tile_size, image_size, text_length, device):
    """
    Create and compile flex attention with a specific sliding block mask.
    This function is cached to avoid recompiling for the same parameters.
    
    Args:
        strategy (tuple): A tuple (t, h, w) defining the strategy
        tile_size (tuple): A tuple (ts_t, ts_h, ts_w) defining the tile size
        image_size (tuple): A tuple (n_t, n_h, n_w) defining the image size
        text_length (int): The text length
        device (str): The device to use
        
    Returns:
        function: A compiled flex attention function with the specified mask
    """
    # Convert strategy to the required format (ceil(t*3/2), h*2, w)
    adjusted_strategy = strategy
    
    # Get the sliding block attention mask
    mask = get_sliding_block_attention_mask(
        adjusted_strategy, 
        tile_size, 
        image_size, 
        text_length, 
        device
    )
    
    # Compile the flex attention with the mask
    compiled_flex_attn = torch.compile(
        partial(flex_attention, block_mask=mask),
        dynamic=False
    )
    
    return compiled_flex_attn

def sliding_tile_attention(q_all, k_all, v_all, strategy, tile_size, 
                           image_size, text_length, scale=None):
    device = q_all.device
    
    # Get the compiled flex attention function (cached if called with same parameters)
    compiled_flex_attn = get_compiled_flex_attention(
        strategy, 
        tile_size,
        image_size, 
        text_length, 
        device
    )
    
    
    # Apply the compiled flex attention
    output = compiled_flex_attn(q_all, k_all, v_all, scale=scale)

        
    return output


def tile(x, sp_size):
    x = rearrange(x, "b (sp t h w) head d -> b (t sp h w) head d", sp=sp_size, t=12 // sp_size, h=48, w=80)
    return rearrange(x,
                    "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
                    n_t=2,
                    n_h=6,
                    n_w=10,
                    ts_t=6,
                    ts_h=8,
                    ts_w=8)

def flops_attention_forward(batch, seqlen, headdim, nheads):
    """
    Calculate FLOPs for attention forward pass operation.
    """
    return 4 * batch * seqlen**2 * nheads * headdim

def efficiency(flop, time):
    """
    Calculate efficiency in TFLOPs/s
    """
    return (flop / time / 10**12) if not math.isnan(time) and time > 0 else 0.

def benchmark_tk_attention(func, q, k, v, o, text_length, repeats=100, desc=""):
    time_f, m = benchmark_forward(func, q, k, v, o, text_length, repeats=repeats, desc=desc, verbose=False)
    return m.mean

def benchmark_baseline_attention(func, qkv, attn_mask, causal, dropout_p, softmax_sacle, repeats=100, desc=""):
    time_f, m = benchmark_forward(func, qkv, attn_mask, causal, dropout_p, softmax_sacle, repeats=repeats, desc=desc, verbose=False)
    return m.mean

def benchmark_compiled_attention(func, q, k, v, strategy, tile_size, image_size, text_length, repeats=100, desc=""):
    time_f, m = benchmark_forward(func, q, k, v, strategy, tile_size, image_size, text_length, repeats=repeats, desc=desc, verbose=False)
    return m.mean
    
    
def benchmark_attention(func, q, k, v, repeats=100, desc=""):
    """
    Benchmark forward pass of attention function
    """
    time_f, m = benchmark_forward(func, q, k, v, repeats=repeats, desc=desc, verbose=False)
    return m.mean

def main():
    debug = False

    # Load tensors
    print("Loading tensors...")
    if debug:
        debug_seq_len = 46080
        debug_batch_size = 1
        debug_nheads = 24
        debug_headdim = 128
        q = torch.randn((debug_batch_size, debug_seq_len+256, debug_nheads, debug_headdim), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((debug_batch_size, debug_seq_len+256, debug_nheads, debug_headdim), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((debug_batch_size, debug_seq_len+256, debug_nheads, debug_headdim), device="cuda", dtype=torch.bfloat16)
    else:
        q = torch.load("query.pt").to("cuda")
        k = torch.load("key.pt").to("cuda")
        v = torch.load("value.pt").to("cuda")
    text_mask = torch.load("text_mask.pt")
    text_length = text_mask.sum()
    
    batch_size = q.shape[0]
    nheads = q.shape[2]
    seqlen = q.shape[1]
    headdim = q.shape[3]
    warmup = 50
    repeats = 100  # Number of repeats for reliable timing
    
    # Baseline measurement (standard flex attention without masking)
    baseline_attn = flash_attn_no_pad
    
    # Perform warmup for baseline
    print("\nWarming up baseline...")
    if debug:
        attn_mask = F.pad(text_mask, (debug_seq_len, 0), value=True)
    else:
        attn_mask = F.pad(text_mask, (seqlen - 256, 0), value=True)
    for _ in range(warmup):
        flash_attn_no_pad(torch.stack([q, k, v], dim=2), attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)
    
    # Benchmark baseline
    print("\nBenchmarking baseline flex attention...")
    torch.cuda.synchronize()
    baseline_time = benchmark_baseline_attention(baseline_attn, torch.stack([q, k, v], dim=2), attn_mask, causal=False, dropout_p=0.0, softmax_sacle=None, repeats=repeats, desc="Baseline")
    
    # benchmark tk_4090_cuda
    # tk.attention_fwd_4090(query, key, value, result, text_length)
    print("Benchmarking tk_4090_cuda...")
    result = torch.empty_like(q)
    for _ in range(warmup):
        tk.attention_fwd_4090(q, k, v, result, text_length)
    torch.cuda.synchronize()
    
    tk_time = benchmark_tk_attention(tk.attention_fwd_4090, q, k, v, result, text_length, repeats=repeats, desc="tk_4090_cuda")
    

    # Prepare tensors
    strategies = []
    if not debug:
        query, encoder_query = q.split_with_sizes((q.shape[1] - 256, 256), dim=1)
        key, encoder_key = k.split_with_sizes((k.shape[1] - 256, 256), dim=1)
        value, encoder_value = v.split_with_sizes((v.shape[1] - 256, 256), dim=1)

        q = torch.cat([tile(query, 1), encoder_query], dim=1).transpose(1, 2)
        k = torch.cat([tile(key, 1), encoder_key], dim=1).transpose(1, 2)
        v = torch.cat([tile(value, 1), encoder_value], dim=1).transpose(1, 2)

        # Strategy definitions
        strategies = [
            (2, 6, 1),   # (t, h, w)
            (1, 6, 10),
            (2, 3, 3),
            (2, 6, 10),
            (2, 1, 10),
            (2, 3, 5)
        ]
        
        # Strategy names for better reporting
        strategy_names = [f"Strategy_{t}_{h}_{w}" for t, h, w in strategies]
        
        print(f"Text length: {text_length}")
        
        # flex full baseline
        print("Benchmarking flex full baseline...")
        flex_baseline = torch.compile(
            partial(flex_attention, block_mask=None),
        )
        
        for _ in range(warmup):
            flex_baseline(q, k, v)
        torch.cuda.synchronize()
        flex_baseline_time = benchmark_attention(flex_baseline, q, k, v, repeats=repeats, desc="FA2 Baseline")
        
        
        # Create attention processors for each strategy
        attn_processors = []
        for strategy in strategies:
            mask = get_sliding_block_attention_mask(strategy, (6, 8, 8), (12, 48, 80), text_length, "cuda")
            attn_processor = torch.compile(
                partial(flex_attention, block_mask=mask), 
            )
            attn_processors.append(attn_processor)
    
    # Setup benchmarking parameters
    
    print(f"Tensor shapes - Q/K/V: {q.shape}")
    print(f"Benchmarking with: batch_size={batch_size}, nheads={nheads}, seqlen={seqlen}, headdim={headdim}")
    
    # Calculate theoretical FLOPS for forward pass
    flops = flops_attention_forward(batch_size, seqlen, headdim, nheads)
    print(f"Theoretical FLOPs: {flops / 10**12:.2f} TFLOPs")
    
    tk_efficiency = efficiency(flops, tk_time)
    print(f"tk_4090_cuda forward: {tk_time:.6f}s, {tk_efficiency:.2f} TFLOPs/s")
    
    baseline_efficiency = efficiency(flops, baseline_time)
    
    print(f"Baseline forward: {baseline_time:.6f}s, {baseline_efficiency:.2f} TFLOPs/s")
    
    # Create result containers
    times = {}
    speeds = {}
    speedups = {}
    
    # Benchmark each attention strategy
    print("\nBenchmarking attention strategies...")
    if not debug:
        for i, (processor, strategy, name) in enumerate(zip(attn_processors, strategies, strategy_names)):
            t, h, w = strategy
            
            # Warmup
            print(f"\nWarming up {name}...")
            for _ in range(warmup):
                processor(q, k, v)
                sliding_tile_attention(q, k, v, strategy, (6, 8, 8), (12, 48, 80), text_length)
            
            # Benchmark
            print(f"Benchmarking {name} (t={t}, h={h}, w={w})...")
            torch.cuda.synchronize()
            fwd_time = benchmark_attention(processor, q, k, v, repeats=repeats, desc=name)
            print(f"processor {name} (t={t}, h={h}, w={w}): {fwd_time:.6f}s")
            
            compiled_fwd_time = benchmark_compiled_attention(sliding_tile_attention, q, k, v, strategy, (6, 8, 8), (12, 48, 80), text_length, repeats=repeats, desc=name)
            
            
            print(f"Compiled {name} (t={t}, h={h}, w={w}): {compiled_fwd_time:.6f}s")
            
            # Save results
            times[name] = fwd_time
            speeds[name] = efficiency(flops, fwd_time)
            
            # Calculate theoretical and actual speedup
            theoretical_speedup = (2*6*10)/(t*h*w)
            actual_speedup = baseline_time / fwd_time
            speedups[name] = (theoretical_speedup, actual_speedup)
            
            # Print results
            print(f"Strategy: {name} (t={t}, h={h}, w={w})")
            print(f"Forward: {fwd_time:.6f}s, {speeds[name]:.2f} TFLOPs/s")
            print(f"Theoretical speedup: {theoretical_speedup:.2f}x")
            print(f"Actual speedup: {actual_speedup:.2f}x")
            print(f"Efficiency ratio (actual/theoretical): {actual_speedup/theoretical_speedup:.2f}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'Strategy':<15} {'Time (ms)':<15} {'TFLOPs/s':<10} {'Actual Speedup':<15} {'Theoretical':<15} {'Efficiency':<10}")
    print("-"*80)
    
    # Baseline row
    print(f"{'FA2 Baseline':<15} {baseline_time*1000:13.2f} {baseline_efficiency:10.2f} {1.00:15.2f} {1.00:15.2f} {1.00:10.2f}")
    if not debug:
        print(f"{'FA2 Flex':<15} {flex_baseline_time*1000:13.2f} {efficiency(flops, flex_baseline_time):10.2f} {baseline_time/flex_baseline_time:15.2f} {1.00:15.2f} {baseline_time/flex_baseline_time:10.2f}")
    print(f"{'Tk_4090_cuda':<15} {tk_time*1000:13.2f} {tk_efficiency:10.2f} {baseline_time/tk_time:15.2f} {1.00:15.2f} {baseline_time/tk_time:10.2f}")
    
    # Strategy rows
    if not debug:
        for name in strategy_names:
            theoretical, actual = speedups[name]
            print(f"{name:<15} {times[name]*1000:13.2f} {speeds[name]:10.2f} {actual:15.2f} {theoretical:15.2f} {actual/theoretical:10.2f}")

if __name__ == "__main__":
    main()