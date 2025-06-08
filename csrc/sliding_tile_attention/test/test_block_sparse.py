import torch
from flash_attn_interface import flash_attn_func
from st_attn import block_sparse_attention_fwd, block_sparse_attention_backward, BLOCK_M, BLOCK_N
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def pytorch_test(Q, K, V, dO, block_sparse_mask):
    q_ = Q.to(torch.float64).requires_grad_()
    k_ = K.to(torch.float64).requires_grad_()
    v_ = V.to(torch.float64).requires_grad_()
    dO_ = dO.to(torch.float64)
    
    # manual pytorch implementation of scaled dot product attention with block sparsity
    QK = torch.matmul(q_, k_.transpose(-2, -1))
    QK /= (q_.size(-1) ** 0.5)
    
    # Apply block sparse mask
    bs, h, n, d = Q.size()
    num_q_blocks = n // BLOCK_M
    num_kv_blocks = n // BLOCK_N
    
    # Create a full mask initialized to -inf
    sparse_mask = torch.full((bs, h, n, n), float('-inf'), device=QK.device, dtype=QK.dtype)
    
    # Fill in the allowed blocks based on block_sparse_mask
    for b in range(bs):
        for head in range(h):
            for q_block in range(num_q_blocks):
                for kv_block in range(num_kv_blocks):
                    if block_sparse_mask[b, head, q_block, kv_block]:
                        q_start, q_end = q_block * BLOCK_M, (q_block + 1) * BLOCK_M
                        kv_start, kv_end = kv_block * BLOCK_N, (kv_block + 1) * BLOCK_N
                        sparse_mask[b, head, q_start:q_end, kv_start:kv_end] = 0.0
    
    # Apply the sparse mask
    QK = QK + sparse_mask
    
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v_)
    
    output.backward(dO_)
    
    q_grad = q_.grad
    k_grad = k_.grad
    v_grad = v_.grad
    
    return output, q_grad, k_grad, v_grad


def h100_fwd_kernel_test(Q, K, V, dO, q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num, mode): 
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    o, l_vec = block_sparse_attention_fwd(Q, K, V, q2k_block_sparse_index, q2k_block_sparse_num)
    if mode == 'forward_only':
        return o, None, None, None
    else:  # 'forward_backward'
        qg, kg, vg = block_sparse_attention_backward(Q, K, V, o, l_vec, dO, k2q_block_sparse_index, k2q_block_sparse_num)
        return o, qg, kg, vg

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)

    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    
    return scaled_tensor.contiguous()


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

def check_correctness(b, h, n, d, mean, std, k=None, num_iterations=100, error_mode='all', test_mode='forward_backward'):
    results = {
        'TK vs PT': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
    }

    for _ in range(num_iterations):
        torch.manual_seed(0)
        
        Q  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        K  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        V  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        
        # Setup block sparse parameters
        num_q_blocks = n // BLOCK_M
        num_kv_blocks = n // BLOCK_N
        
        # Use the provided k value directly
        if k is None:
            k = num_kv_blocks  # Default to full attention if k is not specified
        k = min(k, num_kv_blocks)  # Ensure k is not larger than num_kv_blocks
        
        # Generate block sparse pattern
        q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num, block_sparse_mask = generate_block_sparse_pattern(
            b, h, num_q_blocks, num_kv_blocks, k, device="cuda")
        pt_o, pt_qg, pt_kg, pt_vg = pytorch_test(Q, K, V, dO, block_sparse_mask)
        
        if test_mode == 'forward_only':
            tk_o, _, _, _ = h100_fwd_kernel_test(Q, K, V, dO, q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num, 'forward_only')
            tensors = [(pt_o, tk_o)]
        else:  # 'forward_backward'
            tk_o, tk_qg, tk_kg, tk_vg = h100_fwd_kernel_test(Q, K, V, dO, q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num, 'forward_backward')
            
            if error_mode == 'output':
                tensors = [(pt_o, tk_o)]
            elif error_mode == 'backward':
                tensors = [(pt_qg, tk_qg),
                           (pt_kg, tk_kg),
                           (pt_vg, tk_vg)]
            else:  # 'all'
                tensors = [(pt_o, tk_o),
                           (pt_qg, tk_qg),
                           (pt_kg, tk_kg),
                           (pt_vg, tk_vg)]
        for pt, tk in tensors:
            diff = pt - tk
            abs_diff = torch.abs(diff)
            results['TK vs PT']['sum_diff'] += torch.sum(abs_diff).item()
            results['TK vs PT']['sum_abs'] += torch.sum(torch.abs(pt)).item()
            results['TK vs PT']['max_diff'] = max(results['TK vs PT']['max_diff'], torch.max(abs_diff).item())
                
        torch.cuda.empty_cache()

    # Calculate total elements based on test mode and error mode
    if test_mode == 'forward_only':
        total_elements = b * h * n * d * num_iterations
    else:  # 'forward_backward'
        total_elements = b * h * n * d * num_iterations * (1 if error_mode == 'output' else 3 if error_mode == 'backward' else 4)
    
    for name, data in results.items():
        avg_diff = data['sum_diff'] / total_elements
        max_diff = data['max_diff']
        results[name] = {'avg_diff': avg_diff, 'max_diff': max_diff}

    return results

def generate_error_tables(b, h, d, mean, std, k=None, error_mode='all', test_mode='forward_backward'):
    seq_lengths = [768 * (2**i) for i in range(1)]

    print(f"\n{'='*80}")
    print(f"BLOCK SPARSE ERROR COMPARISON TABLE (b={b}, h={h}, d={d}, mean={mean}, std={std}, k={k})")
    print(f"Mode: {error_mode}, Test: {test_mode}")
    print(f"{'='*80}")
    
    # Print header
    print(f"{'Seq Length':<12} | {'TK Avg':<12} | {'TK Max':<12}")
    print(f"{'-'*12} | {'-'*12} | {'-'*12}")
    
    for n in seq_lengths:
        results = check_correctness(b, h, n, d, mean, std, k, error_mode=error_mode, test_mode=test_mode)
        
        tk_avg = results['TK vs PT']['avg_diff']
        tk_max = results['TK vs PT']['max_diff']
        
        # Print row
        print(f"{n:<12} | {tk_avg:<12.6e} | {tk_max:<12.6e}")
    
    print(f"{'='*80}\n")

# fix random seed
torch.manual_seed(0)

# Example usage
b, h, d = 2, 2, 64
mean = 1e-1
std = 10
k = 4  # Directly specify k instead of sparsity

# Test forward only
# generate_error_tables(b, h, d, mean, std, k, error_mode='output', test_mode='forward_only')

# Test forward and backward
generate_error_tables(b, h, d, mean, std, k, error_mode='all', test_mode='forward_backward')

print("Block sparse attention error comparison completed.")