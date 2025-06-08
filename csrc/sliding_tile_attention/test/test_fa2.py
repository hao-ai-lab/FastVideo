import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def pytorch_test(Q, K, V, dO):
    q_ = Q.to(torch.float64).requires_grad_()
    k_ = K.to(torch.float64).requires_grad_()
    v_ = V.to(torch.float64).requires_grad_()
    dO_ = dO.to(torch.float64)
    
    # manual pytorch implementation of scaled dot product attention
    QK = torch.matmul(q_, k_.transpose(-2, -1))
    QK /= (q_.size(-1) ** 0.5)
    
    # Causal mask removed since causal is always false
    
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v_)
    
    output.backward(dO_)
    
    q_grad = q_.grad
    k_grad = k_.grad
    v_grad = v_.grad
    
    return output, q_grad, k_grad, v_grad

def fa2_test(Q, K, V, dO):
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
    output.backward(dO)
    
    return output, Q.grad, K.grad, V.grad

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)

    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    
    return scaled_tensor.contiguous()

def check_correctness(b, h, n, d, mean, std, num_iterations=100, error_mode='all', test_mode='forward_backward'):
    results = {
        'FA2 vs PT': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
    }

    for _ in range(num_iterations):
        torch.manual_seed(0)
        
        Q  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        K  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        V  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        
        pt_o, pt_qg, pt_kg, pt_vg = pytorch_test(Q, K, V, dO)
        fa2_o, fa2_qg, fa2_kg, fa2_vg = fa2_test(Q, K, V, dO)
        
        if test_mode == 'forward_only':
            tensors_fa2_pt = [(pt_o, fa2_o)]
        else:  # 'forward_backward'
            if error_mode == 'output':
                tensors_fa2_pt = [(pt_o, fa2_o)]
            elif error_mode == 'backward':
                tensors_fa2_pt = [(pt_qg, fa2_qg),
                               (pt_kg, fa2_kg),
                               (pt_vg, fa2_vg)]
            else:  # 'all'
                tensors_fa2_pt = [(pt_o, fa2_o),
                               (pt_qg, fa2_qg),
                               (pt_kg, fa2_kg),
                               (pt_vg, fa2_vg)]
                
        for pt, fa2 in tensors_fa2_pt:
            diff = pt - fa2
            abs_diff = torch.abs(diff)
            results['FA2 vs PT']['sum_diff'] += torch.sum(abs_diff).item()
            results['FA2 vs PT']['sum_abs'] += torch.sum(torch.abs(pt)).item()
            results['FA2 vs PT']['max_diff'] = max(results['FA2 vs PT']['max_diff'], torch.max(abs_diff).item())
                
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

def generate_error_tables(b, h, d, mean, std, error_mode='all', test_mode='forward_backward'):
    seq_lengths = [768 * (2**i) for i in range(1)]

    print(f"\n{'='*80}")
    print(f"ATTENTION ERROR COMPARISON TABLE (b={b}, h={h}, d={d}, mean={mean}, std={std})")
    print(f"Mode: {error_mode}, Test: {test_mode}")
    print(f"{'='*80}")
    
    # Print header
    print(f"{'Seq Length':<12} | {'FA2 vs PT Avg':<15} | {'FA2 vs PT Max':<15}")
    print(f"{'-'*12} | {'-'*15} | {'-'*15}")
    
    for n in seq_lengths:
        results = check_correctness(b, h, n, d, mean, std, error_mode=error_mode, test_mode=test_mode)
        
        fa2_pt_avg = results['FA2 vs PT']['avg_diff']
        fa2_pt_max = results['FA2 vs PT']['max_diff']
        
        # Print row
        print(f"{n:<12} | {fa2_pt_avg:<15.6e} | {fa2_pt_max:<15.6e}")
    
    print(f"{'='*80}\n")

# fix random seed
torch.manual_seed(0)

# Example usage
b, h, d = 2, 2, 64
mean = 1e-1
std = 10

# Test forward only
generate_error_tables(b, h, d, mean, std, error_mode='output', test_mode='forward_only')

# Test forward and backward
generate_error_tables(b, h, d, mean, std, error_mode='all', test_mode='forward_backward')

print("Attention error comparison completed.")