#!/usr/bin/env python3

import torch
import math
from vsa.block_sparse_attn_triton import attention_sparse

def test_sparse_attention():
    """Test the sparse attention implementation with a simple example."""
    
    # Small test case
    B, H, T, D = 2, 4, 256, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("CUDA not available, skipping test")
        return
    
    # Create test tensors
    q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    
    # Create simple sparse pattern - each Q block attends to itself and a few neighbors
    BLOCK_SIZE = 64
    num_blocks = T // BLOCK_SIZE
    max_kv_blks = 4  # Each Q block attends to at most 4 KV blocks
    max_q_blks = 4   # Each KV block is attended to by at most 4 Q blocks
    
    # Initialize sparse indices
    q2k_index = torch.zeros(B, H, num_blocks, max_kv_blks, device=device, dtype=torch.int32)
    q2k_num = torch.zeros(B, H, num_blocks, device=device, dtype=torch.int32)
    k2q_index = torch.zeros(B, H, num_blocks, max_q_blks, device=device, dtype=torch.int32)
    k2q_num = torch.zeros(B, H, num_blocks, device=device, dtype=torch.int32)
    
    # Simple pattern: each block attends to itself and next block (cyclically)
    for b in range(B):
        for h in range(H):
            for q_blk in range(num_blocks):
                # Q block attends to itself and next block
                attend_to = [(q_blk) % num_blocks, (q_blk + 1) % num_blocks]
                q2k_num[b, h, q_blk] = len(attend_to)
                for i, kv_blk in enumerate(attend_to):
                    q2k_index[b, h, q_blk, i] = kv_blk
                
                # Reverse mapping: which Q blocks attend to this KV block
                attended_by = [(q_blk - 1) % num_blocks, q_blk]
                k2q_num[b, h, q_blk] = len(attended_by)
                for i, q_attending in enumerate(attended_by):
                    k2q_index[b, h, q_blk, i] = q_attending
    
    print(f"Testing sparse attention with shape: {q.shape}")
    print(f"Block size: {BLOCK_SIZE}, Num blocks: {num_blocks}")
    print(f"Max KV blocks per Q: {max_kv_blks}, Max Q blocks per KV: {max_q_blks}")
    
    try:
        # Forward pass
        output = attention_sparse(q, k, v, q2k_index, q2k_num, k2q_index, k2q_num)
        print(f"Forward pass successful! Output shape: {output.shape}")
        
        # Backward pass test
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        
        print("Backward pass successful!")
        print(f"Q grad shape: {q.grad.shape if q.grad is not None else 'None'}")
        print(f"K grad shape: {k.grad.shape if k.grad is not None else 'None'}")
        print(f"V grad shape: {v.grad.shape if v.grad is not None else 'None'}")
        
        print("✅ Sparse attention test passed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sparse_attention() 