import torch
import thunderkittens_cuda as tk
import time

def test_attention_kernel():
    # Create small test tensors
    batch_size = 1
    seq_len = 1  # Start small
    n_heads = 4
    head_dim = 64  # Must match your kernel's supported dimensions (64 or 128)
    
    # Create random inputs
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, 
                   device='cuda', dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    print("Input shapes:", q.shape)
    print("Input tensors created")
    
    o = torch.empty_like(q)
    try:
        # Add timing
        start = time.time()
        print("Calling thunderkittens kernel...")
        _ = tk.attention_fwd_4090(q, k, v, o, False)
        print("yes")
        torch.cuda.synchronize()
        end = time.time()
        print(f"Kernel completed in {(end-start)*1000:.2f} ms")
        print("Output shape:", o.shape)
        
        # Run again with different parameters
        _ = tk.attention_fwd_4090(q, k, v, o, True)  # Try with causal=True
        print("Causal version completed")
        
        # Validate results (compare with PyTorch implementation)
        print("Computing reference result...")
        q_scaled = q * (1.0 / torch.sqrt(torch.tensor(head_dim, dtype=torch.float)))
        attn = torch.matmul(q_scaled, k.transpose(-1, -2))
        attn = torch.softmax(attn, dim=-1)
        ref_out = torch.matmul(attn, v)
        
        # Check if results are reasonable
        error = (o - ref_out).abs().mean().item()
        print(f"Mean absolute error: {error}")
        
        return True
    except Exception as e:
        print(f"Error calling kernel: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing attention kernel...")
    success = test_attention_kernel()
    print(f"Test {'succeeded' if success else 'failed'}")