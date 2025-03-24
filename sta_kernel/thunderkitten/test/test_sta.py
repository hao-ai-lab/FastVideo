import torch
from flex_sta_ref import get_sliding_tile_attention_mask
from st_attn import sliding_tile_attention
from torch.nn.attention.flex_attention import flex_attention
# from flash_attn_interface import flash_attn_func
from tqdm import tqdm

flex_attention = torch.compile(flex_attention, dynamic=False)


def flex_test(Q, K, V, kernel_size):
    mask = get_sliding_tile_attention_mask(kernel_size, (6, 8, 8), (36, 48, 48), 39, 'cuda', 0)
    output = flex_attention(Q, K, V, block_mask=mask)

    return output


def h100_fwd_kernel_test(Q, K, V, kernel_size):
    o = sliding_tile_attention(Q, K, V, [kernel_size] * 24, 39, False)
    return o


def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)

    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude

    return scaled_tensor.contiguous()


def check_correctness(b, h, n, d, causal, mean, std, num_iterations=50, error_mode='all'):
    results = {
        'TK vs FLEX': {
            'sum_diff': 0,
            'sum_abs': 0,
            'max_diff': 0
        },
    }
    kernel_size_ls = [(6, 1, 6), (6, 6, 1)]
    from tqdm import tqdm
    for kernel_size in tqdm(kernel_size_ls):
        for _ in range(num_iterations):
            torch.manual_seed(0)

            Q = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            K = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            V = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            tk_o = h100_fwd_kernel_test(Q, K, V, kernel_size)
            pt_o = flex_test(Q, K, V, kernel_size)

            diff = pt_o - tk_o
            abs_diff = torch.abs(diff)
            results['TK vs FLEX']['sum_diff'] += torch.sum(abs_diff).item()
            results['TK vs FLEX']['max_diff'] = max(results['TK vs FLEX']['max_diff'], torch.max(abs_diff).item())

            torch.cuda.empty_cache()
        print("kernel_size", kernel_size)
        print("max_diff", torch.max(abs_diff).item())
        print(
            "avg_diff",
            torch.sum(abs_diff).item() / (b * h * n * d *
                                          (1 if error_mode == 'output' else 3 if error_mode == 'backward' else 4)))

    total_elements = b * h * n * d * num_iterations * (1 if error_mode == 'output' else
                                                       3 if error_mode == 'backward' else 4) * len(kernel_size_ls)
    for name, data in results.items():
        avg_diff = data['sum_diff'] / total_elements
        max_diff = data['max_diff']
        results[name] = {'avg_diff': avg_diff, 'max_diff': max_diff}

    return results


def generate_error_graphs(b, h, d, causal, mean, std, error_mode='all'):
    seq_lengths = [82944]

    tk_avg_errors, tk_max_errors = [], []

    for n in tqdm(seq_lengths, desc="Generating error data"):
        results = check_correctness(b, h, n, d, causal, mean, std, error_mode=error_mode)

        tk_avg_errors.append(results['TK vs FLEX']['avg_diff'])
        tk_max_errors.append(results['TK vs FLEX']['max_diff'])


# Example usage
b, h, d = 2, 24, 128
causal = False
mean = 1e-1
std = 10

for mode in ['output']:
    generate_error_graphs(b, h, d, causal, mean, std, error_mode=mode)

print("Error graphs generated and saved for all modes.")
