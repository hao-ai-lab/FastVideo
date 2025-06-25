import torch
import argparse
from flash_attn.utils.benchmark import benchmark_forward
from flash_attn import flash_attn_func
from vsa import block_sparse_attention_fwd, block_sparse_attention_backward, BlockSparseAttentionFunction
from vsa import BLOCK_M, BLOCK_N

import numpy as np
import random
import gc

def set_seed(seed: int = 42):
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU


@torch.no_grad
def precision_metric(quant_o, fa2_o): 
    x, xx = quant_o.float(), fa2_o.float() 
    sim = torch.nn.functional.cosine_similarity(x.reshape(1, -1), xx.reshape(1, -1)).item()
    l1 =   ((x - xx).abs().sum() / xx.abs().sum() ).item()
    rmse = torch.sqrt(torch.mean((x -xx) ** 2)).item()

    return sim, l1, rmse

def create_input_tensors(batch, head, seq_len, headdim):
    """Create random input tensors for attention."""
    q = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    return q, k, v

def generate_block_sparse_pattern(bs, h, num_q_blocks, num_kv_blocks, k, device="cuda"):
    """
    Generate a block sparse pattern where each q block attends to exactly k kv blocks.
    
    Args:
        bs: batch size
        h: number of heads
        num_q_blocks: number of query blocks
        num_kv_blocks: number of key-value blocks
        k: number of kv blocks each q block attends to
        device: device to create tensors on
        
    Returns:
        q2k_block_sparse_index: [bs, h, num_q_blocks, k]
            Contains the indices of kv blocks that each q block attends to.
        q2k_block_sparse_num: [bs, h, num_q_blocks]
            Contains the number of kv blocks that each q block attends to (all equal to k).
        k2q_block_sparse_index: [bs, h, num_kv_blocks, num_q_blocks]
            Contains the indices of q blocks that attend to each kv block.
        k2q_block_sparse_num: [bs, h, num_kv_blocks]
            Contains the number of q blocks that attend to each kv block.
        block_sparse_mask: [bs, h, num_q_blocks, num_kv_blocks]
            Binary mask where 1 indicates attention connection.
    """
    # Ensure k is not larger than num_kv_blocks
    k = min(k, num_kv_blocks)
    
    # Create random scores for sampling
    scores = torch.rand(bs, h, num_q_blocks, num_kv_blocks, device=device)
    
    # Get top-k indices for each q block
    _, q2k_block_sparse_index = torch.topk(scores, k, dim=-1)
    q2k_block_sparse_index = q2k_block_sparse_index.to(torch.int32)

    # sort q2k_block_sparse_index
    q2k_block_sparse_index, _ = torch.sort(q2k_block_sparse_index, dim=-1)

    # All q blocks attend to exactly k kv blocks
    q2k_block_sparse_num = torch.full((bs, h, num_q_blocks), k, dtype=torch.int32, device=device)
    
    # Create the corresponding mask
    block_sparse_mask = torch.zeros(bs, h, num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device)
    
    # Fill in the mask based on the indices
    for b in range(bs):
        for head in range(h):
            for q_idx in range(num_q_blocks):
                kv_indices = q2k_block_sparse_index[b, head, q_idx]
                block_sparse_mask[b, head, q_idx, kv_indices] = True
    
    # Create the reverse mapping (k2q)
    # First, initialize lists to collect q indices for each kv block
    k2q_indices_list = [[[] for _ in range(num_kv_blocks)] for _ in range(bs * h)]
    
    # Populate the lists based on q2k mapping
    for b in range(bs):
        for head in range(h):
            flat_idx = b * h + head
            for q_idx in range(num_q_blocks):
                kv_indices = q2k_block_sparse_index[b, head, q_idx].tolist()
                for kv_idx in kv_indices:
                    k2q_indices_list[flat_idx][kv_idx].append(q_idx)
    
    # Find the maximum number of q blocks that attend to any kv block
    max_q_per_kv = 0
    for flat_idx in range(bs * h):
        for kv_idx in range(num_kv_blocks):
            max_q_per_kv = max(max_q_per_kv, len(k2q_indices_list[flat_idx][kv_idx]))
    
    # Create tensors for k2q mapping
    k2q_block_sparse_index = torch.full((bs, h, num_kv_blocks, max_q_per_kv), -1, 
                                        dtype=torch.int32, device=device)
    k2q_block_sparse_num = torch.zeros((bs, h, num_kv_blocks), 
                                       dtype=torch.int32, device=device)
    
    # Fill the tensors
    for b in range(bs):
        for head in range(h):
            flat_idx = b * h + head
            for kv_idx in range(num_kv_blocks):
                q_indices = k2q_indices_list[flat_idx][kv_idx]
                num_q = len(q_indices)
                k2q_block_sparse_num[b, head, kv_idx] = num_q
                if num_q > 0:
                    k2q_block_sparse_index[b, head, kv_idx, :num_q] = torch.tensor(
                        q_indices, dtype=torch.int32, device=device)
                
    return q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num, block_sparse_mask

def main(args):
    set_seed(42)
    
    # Extract parameters
    batch = args.batch_size
    head = args.num_heads
    headdim = args.head_dim
    num_iterations = args.num_iterations
    
    print(f"Block Sparse Attention Benchmark")
    print(f"batch: {batch}, head: {head}, headdim: {headdim}, iterations: {num_iterations}")
    
    # Test with different sequence lengths
    for seq_len in args.seq_lengths:
        # Skip very long sequences if they might cause OOM
        # if seq_len > 16384 and batch > 1:
        #     continue
            
        print("="*100)
        print(f"\nSequence length: {seq_len}")
        
        # Collect metrics across iterations
        forward_metrics = {'sim': [], 'l1': [], 'rmse': []}
        grad_q_metrics = {'sim': [], 'l1': [], 'rmse': []}
        grad_k_metrics = {'sim': [], 'l1': [], 'rmse': []}
        grad_v_metrics = {'sim': [], 'l1': [], 'rmse': []}
        
        for iter_idx in range(num_iterations):
            if num_iterations > 1:
                print(f"\nIteration {iter_idx+1}/{num_iterations}")
            
            # Create input tensors
            q, k, v = create_input_tensors(batch, head, seq_len, headdim)
            
            # Setup block sparse parameters
            num_q_blocks = seq_len // BLOCK_M
            num_kv_blocks = seq_len // BLOCK_N
            
            # Determine k value (number of kv blocks per q block)
            topk = args.topk
            if topk is None:
                topk = num_kv_blocks // 10  # Default to ~90% sparsity if k is not specified
            topk = max(1, topk)
            if iter_idx == 0:  # Only print this once
                print(f"Using topk={topk} kv blocks per q block (out of {num_kv_blocks} total kv blocks)")

            # Generate block sparse pattern
            q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num, block_sparse_mask = generate_block_sparse_pattern(
                batch, head, num_q_blocks, num_kv_blocks, topk, device="cuda")

            # expand block_sparse_mask to full mask
            block_mask_expanded = block_sparse_mask.unsqueeze(-1).unsqueeze(-2)  # [b, h, num_q_blocks, num_kv_blocks, 1, 1]
            block_mask_expanded = block_mask_expanded.expand(-1, -1, -1, -1, BLOCK_M, BLOCK_N)  # [b, h, num_q_blocks, num_kv_blocks, BLOCK_M, BLOCK_N]
            full_mask = block_mask_expanded.permute(0, 1, 2, 4, 3, 5).reshape(batch, head, seq_len, seq_len)

            q.requires_grad = True
            k.requires_grad = True
            v.requires_grad = True


            # testing forward
            o = BlockSparseAttentionFunction.apply(q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num)
            del q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num, block_sparse_mask, block_mask_expanded
            grad_o = torch.randn_like(o)
            o.backward(grad_o)
            # clear memory
            q_sdpa = q.detach().clone()
            k_sdpa = k.detach().clone()
            v_sdpa = v.detach().clone()
            q_sdpa.requires_grad = True
            k_sdpa.requires_grad = True
            v_sdpa.requires_grad = True
            q.data = torch.empty(0, device=q.device)
            k.data = torch.empty(0, device=k.device)
            v.data = torch.empty(0, device=v.device)
            torch.cuda.empty_cache()

            o_sdpa = torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=full_mask)


            sim, l1, rmse = precision_metric(o, o_sdpa)
            assert sim > 0.9999, f"SSIM too low: {sim}"
            assert l1 < 8e-5, f"l1 too large: {l1}"
            assert rmse < 2e-5, f"RMSE too large: {rmse}"
            forward_metrics['sim'].append(sim)
            forward_metrics['l1'].append(l1)
            forward_metrics['rmse'].append(rmse)

            print(f"block_sparse_attention_fwd vs torch.nn.functional.scaled_dot_product_attention:\nsim: {sim}, l1: {l1}, rmse: {rmse}")

            # test backward
            o_sdpa.backward(grad_o)

            sim, l1, rmse = precision_metric(q.grad, q_sdpa.grad)
            # Error bounds collected on H100
            assert sim > 0.9999, f"SSIM too low: {sim}"
            assert l1 < 4e-3, f"l1 too large: {l1}"
            assert rmse < 3e-4, f"RMSE too large: {rmse}"
            grad_q_metrics['sim'].append(sim)
            grad_q_metrics['l1'].append(l1)
            grad_q_metrics['rmse'].append(rmse)
            print(f"block_sparse_attention_bwd vs torch.nn.functional.scaled_dot_product_attention grad_q:\nsim: {sim}, l1: {l1}, rmse: {rmse}")        
            
            sim, l1, rmse = precision_metric(k.grad, k_sdpa.grad)
            assert sim > 0.9999, f"SSIM too low: {sim}"
            assert l1 < 4e-3, f"l1 too large: {l1}"
            assert rmse < 2e-4, f"RMSE too large: {rmse}"
            grad_k_metrics['sim'].append(sim)
            grad_k_metrics['l1'].append(l1)
            grad_k_metrics['rmse'].append(rmse)
            print(f"block_sparse_attention_bwd vs torch.nn.functional.scaled_dot_product_attention grad_k:\nsim: {sim}, l1: {l1}, rmse: {rmse}")
            
            sim, l1, rmse = precision_metric(v.grad, v_sdpa.grad)
            assert sim > 0.9999, f"SSIM too low: {sim}"
            assert l1 < 1e-4, f"l1 too large: {l1}"
            assert rmse < 2e-5, f"RMSE too large: {rmse}"
            grad_v_metrics['sim'].append(sim)
            grad_v_metrics['l1'].append(l1)
            grad_v_metrics['rmse'].append(rmse)
            print(f"block_sparse_attention_bwd vs torch.nn.functional.scaled_dot_product_attention grad_v:\nsim: {sim}, l1: {l1}, rmse: {rmse}")
            
            del o, o_sdpa, grad_o, q_sdpa, k_sdpa, v_sdpa
            gc.collect()
            torch.cuda.empty_cache()

        # Print summary statistics if multiple iterations were run
        if num_iterations > 1:
            print("\n" + "="*50)
            print(f"Summary Statistics (over {num_iterations} iterations):")
            
            print("\nForward metrics:")
            print(f"Similarity: mean={np.mean(forward_metrics['sim']):.6f}, std={np.std(forward_metrics['sim']):.6f}, min={np.min(forward_metrics['sim']):.6f}")
            print(f"L1 error: mean={np.mean(forward_metrics['l1']):.6f}, std={np.std(forward_metrics['l1']):.6f}, max={np.max(forward_metrics['l1']):.6f}")
            print(f"RMSE: mean={np.mean(forward_metrics['rmse']):.6f}, std={np.std(forward_metrics['rmse']):.6f}, max={np.max(forward_metrics['rmse']):.6f}")
            
            print("\nGradient Q metrics:")
            print(f"Similarity: mean={np.mean(grad_q_metrics['sim']):.6f}, std={np.std(grad_q_metrics['sim']):.6f}, min={np.min(grad_q_metrics['sim']):.6f}")
            print(f"L1 error: mean={np.mean(grad_q_metrics['l1']):.6f}, std={np.std(grad_q_metrics['l1']):.6f}, max={np.max(grad_q_metrics['l1']):.6f}")
            print(f"RMSE: mean={np.mean(grad_q_metrics['rmse']):.6f}, std={np.std(grad_q_metrics['rmse']):.6f}, max={np.max(grad_q_metrics['rmse']):.6f}")
            
            print("\nGradient K metrics:")
            print(f"Similarity: mean={np.mean(grad_k_metrics['sim']):.6f}, std={np.std(grad_k_metrics['sim']):.6f}, min={np.min(grad_k_metrics['sim']):.6f}")
            print(f"L1 error: mean={np.mean(grad_k_metrics['l1']):.6f}, std={np.std(grad_k_metrics['l1']):.6f}, max={np.max(grad_k_metrics['l1']):.6f}")
            print(f"RMSE: mean={np.mean(grad_k_metrics['rmse']):.6f}, std={np.std(grad_k_metrics['rmse']):.6f}, max={np.max(grad_k_metrics['rmse']):.6f}")
            
            print("\nGradient V metrics:")
            print(f"Similarity: mean={np.mean(grad_v_metrics['sim']):.6f}, std={np.std(grad_v_metrics['sim']):.6f}, min={np.min(grad_v_metrics['sim']):.6f}")
            print(f"L1 error: mean={np.mean(grad_v_metrics['l1']):.6f}, std={np.std(grad_v_metrics['l1']):.6f}, max={np.max(grad_v_metrics['l1']):.6f}")
            print(f"RMSE: mean={np.mean(grad_v_metrics['rmse']):.6f}, std={np.std(grad_v_metrics['rmse']):.6f}, max={np.max(grad_v_metrics['rmse']):.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Block Sparse Attention')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of heads')
    parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--topk', type=int, default=64, help='Number of kv blocks each q block attends to')
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[29120], help='Sequence lengths to benchmark')
    parser.add_argument('--num_iterations', type=int, default=50, help='Number of test iterations to run')
    args = parser.parse_args()
    main(args)