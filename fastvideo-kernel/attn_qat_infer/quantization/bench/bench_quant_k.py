"""
Copyright (c) 2025 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import fp4quant
from triton.tools.mxfp import MXFP4Tensor

from bench_utils import bench_kineto 
b = 1
h = 32
n = 16384
d = 128

def test():
    q = torch.randn((b, h, n, d), device="cuda", dtype=torch.float16)
    o = torch.empty((b, h, n, d // 2), device="cuda", dtype=torch.uint8)
    o_s = torch.empty((b, h, n, d // 16), device="cuda", dtype=torch.float8_e4m3fn)
    fp4quant.scaled_fp4_quant_permute(q, o, o_s, 1)

test()

t = bench_kineto(test, "scaled_fp4_quant_kernel", suppress_kineto_output=True)

IO = b * h * n * d * 2 + b * h * n * d * 0.5 + b * h * n * d // 16 * 1
throughput = IO / t * 1e-9

print(f"Throughput: {throughput:.2f} GB/s")

def scale_and_fp4_tensor(x: torch.Tensor, packed_dim: int = 3, all_ones: bool = False, permuted: bool = False):
    assert x.is_contiguous() and x.ndim == 4 and x.shape[-1] % 16 == 0
    B, H, M, N = x.shape
    x = x.view(B, H, M, N // 16, 16)
    scales = (x.abs().amax(dim=-1, keepdim=True) / 6).to(torch.float32)
    if all_ones:
       scales = torch.ones_like(scales)
    x_scaled = x / scales
    packed_fp4 = MXFP4Tensor(x_scaled.flatten(start_dim=-2)).to_packed_tensor(dim=packed_dim)
    dequant_x = (MXFP4Tensor(x_scaled).to(torch.float32) * scales.to(torch.float8_e4m3fn).to(torch.float32)).flatten(start_dim=-2)
    fp8_scale = scales.flatten(start_dim=-2).to(torch.float8_e4m3fn)
    permuted_fp8_scale = None
    if permuted:
        scales = scales.view(B, H // 64, 4, 16, M, N // 16).permute(0, 1, 3, 2, 4, 5).reshape(B, H, M, N // 16)
        permuted_fp8_scale = scales.view(B, H // 64, 64, M, N // 64, 4).permute(0, 1, 4, 3, 2, 5).reshape(B, H, M, N // 16).to(torch.float8_e4m3fn)
    return fp8_scale, packed_fp4, dequant_x, permuted_fp8_scale

b = 2
h = 4
n = 251
n_padded = (n + 127) // 128 * 128
d = 128

q = torch.randn(b, h, n, d, dtype=torch.float16, device='cuda')
o = torch.empty((b, h, n, d // 2), dtype=torch.uint8, device='cuda')
o_s = torch.empty((b, h, n, d // 16), dtype=torch.float8_e4m3fn, device='cuda')

fp4quant.scaled_fp4_quant(q, o, o_s, 1)

k_permute = [0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31]
o_permuted = torch.empty((b, h, n_padded, d // 2), dtype=torch.uint8, device='cuda')
o_s_permuted = torch.empty((b, h, n_padded, d // 16), dtype=torch.float8_e4m3fn, device='cuda')
fp4quant.scaled_fp4_quant_permute(q, o_permuted, o_s_permuted, 1)

# padding
if n % 128 != 0:
    o_permuted_gt = torch.cat([o, torch.zeros((b, h, n_padded - n, d // 2), dtype=torch.uint8, device='cuda')], dim=2)
    o_s_permuted_gt = torch.cat([o_s, torch.zeros((b, h, n_padded - n, d // 16), dtype=torch.float8_e4m3fn, device='cuda')], dim=2)
else:
    o_permuted_gt = o
    o_s_permuted_gt = o_s

# use scale_and_fp4_tensor + torch permutation to get the ground truth
o_permuted_gt = o_permuted_gt.reshape(b, h, n_padded // 32, 32, d // 2)[:, :, :, k_permute, :].reshape(b, h, n_padded, d // 2)
o_s_permuted_gt = o_s_permuted_gt.reshape(b, h, n_padded // 32, 32, d // 16)[:, :, :, k_permute, :].reshape(b, h, n_padded, d // 16)

assert((o_permuted - o_permuted_gt).abs().max() == 0)
assert((o_s_permuted.float() - o_s_permuted_gt.float()).abs().max() == 0)

print("All tests passed!")