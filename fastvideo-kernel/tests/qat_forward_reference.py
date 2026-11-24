"""Independent BF16 QAT forward and identity-STE reference for SM121 tests.

The quantizer uses torch operations, not Triton outputs. Autograd differentiates
only V through the frozen online quantized probability weights. This reference
is intentionally small: it materializes tile weights for short-sequence tests.
"""
import math

import torch
import torch.nn.functional as F


def fp4_fake_quantize(x, mask=None, global_scale=False, two_level=False):
    x = x.float()
    mask = torch.ones_like(x, dtype=torch.bool) if mask is None else mask
    rows, columns = x.shape[-2:]
    grouped_shape = (*x.shape[:-2], rows, columns // 16, 16)
    maxima = torch.where(mask, x.abs(), 0.0).reshape(grouped_shape).amax(-1)
    if two_level:
        row_max = torch.where(mask, x, 0.0).amax(-1, keepdim=True).clamp_min(1e-8)
        encode = 2688.0 / row_max
    elif global_scale:
        tile_max = maxima.amax((-2, -1), keepdim=True).clamp_min(1e-8)
        encode = 2688.0 / tile_max
    else:
        encode = torch.ones((*x.shape[:-2], 1, 1), device=x.device)
    decode = 1.0 / encode
    scale = (maxima / 6.0 * encode).to(torch.float8_e4m3fn).float()
    denominator = scale * decode
    inverse = torch.where(denominator > 0, 1.0 / denominator, 1.0)
    scaled = (x.reshape(grouped_shape) * inverse[..., None]).reshape(x.shape)
    scaled = torch.where(mask, scaled, 0.0)
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    magnitude = scaled.abs().clamp_max(6.0)
    indices = torch.bucketize(magnitude.contiguous(), bounds)
    ties = (indices < 7) & (magnitude == bounds[indices.clamp_max(6)]) & (indices % 2 == 1)
    indices = indices + ties.long()
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)[indices]
    values = values * torch.where(torch.signbit(scaled), -1.0, 1.0)
    product = (values.reshape(grouped_shape).to(torch.bfloat16) * scale[..., None].to(torch.bfloat16))
    result = (product.float() * decode[..., None]).reshape(x.shape).to(torch.bfloat16)
    return torch.where(mask, result, 0.0)


def quantize_qkv(x):
    batch, heads, length, dimension = x.shape
    padded_length = math.ceil(length / 32) * 32
    padded = F.pad(x, (0, 0, 0, padded_length - length))
    tiles = padded.reshape(batch * heads, padded_length // 32, 32, dimension)
    mask = (torch.arange(padded_length, device=x.device) < length).reshape(1, -1, 32, 1).expand_as(tiles)
    return fp4_fake_quantize(tiles, mask).reshape(batch, heads, padded_length, dimension)[:, :, :length]


def bf16_sum32(x):
    """Sum 32 BF16 lanes in the order the SM121 split-path MMA reduces them.

    The xor-stride order was found empirically: it is the only one that reproduces the
    kernel's BF16 row sums (the saved denominators) bit for bit.
    """
    value = x.to(torch.bfloat16)
    indices = torch.arange(32, device=x.device)
    for stride in (1, 4, 2, 16, 8):
        value = value + value.index_select(-1, indices ^ stride)
    return value[..., 0].float()


def forward_reference(q, k, v, grad_output):
    fake_q, fake_k, fake_v = [quantize_qkv(x) for x in (q, k, v)]
    batch, heads, nq, dimension = q.shape
    nk = k.shape[2]
    padded_q = math.ceil(nq / 32) * 32
    padded_k = math.ceil(nk / 32) * 32
    flat_heads = batch * heads
    q32 = F.pad(fake_q, (0, 0, 0, padded_q - nq)).reshape(flat_heads, padded_q, dimension).float()
    k32 = F.pad(fake_k, (0, 0, 0, padded_k - nk)).reshape(flat_heads, padded_k, dimension).float()
    frozen_v = F.pad(fake_v, (0, 0, 0, padded_k - nk)).reshape(flat_heads, padded_k, dimension).float()
    frozen_v = frozen_v.detach().requires_grad_(True)
    accumulator = torch.zeros_like(q32)
    ste_accumulator = torch.zeros_like(q32)
    running_max = torch.full((flat_heads, padded_q), -torch.inf, device=q.device)
    denominator = torch.ones_like(running_max)
    maxima = []
    scale = dimension**-0.5 * 1.44269504
    for start in range(0, padded_k, 32):
        logits = q32 @ k32[:, start:start + 32].transpose(-1, -2)
        kv_valid = torch.arange(start, start + 32, device=q.device) < nk
        logits = torch.where(kv_valid[None, None, :], logits, -1e6)
        next_max = torch.maximum(running_max, logits.amax(-1) * scale)
        probability = torch.exp2(logits * scale - next_max[..., None])
        high_probability = probability.to(torch.bfloat16).float()
        quant_probability = fp4_fake_quantize(probability.reshape(flat_heads, padded_q // 32, 32, 32))
        quant_probability = quant_probability.reshape_as(probability).float()
        alpha = torch.exp2(running_max - next_max)
        accumulator = accumulator * alpha[..., None] + quant_probability @ frozen_v[:, start:start + 32]
        ste_accumulator = ste_accumulator * alpha[..., None] + high_probability @ frozen_v.detach()[:, start:start + 32]
        denominator = denominator * alpha + bf16_sum32(high_probability)
        running_max = next_max
        maxima.append(next_max[:, :nq])
    continuous_output = (accumulator / denominator[..., None])[:, :nq].reshape(batch, heads, nq, dimension)
    dv = torch.autograd.grad(continuous_output, frozen_v, grad_output.float())[0]
    dv = dv[:, :nk].reshape(batch, heads, nk, dimension).to(torch.bfloat16)
    output = continuous_output.detach().to(torch.bfloat16)
    ste_output = (ste_accumulator / denominator[..., None])[:, :nq].reshape_as(q).to(torch.bfloat16)
    statistics = (running_max + torch.log2(denominator))[:, :nq].reshape(batch, heads, nq)
    maxima = torch.stack(maxima, dim=1).reshape(batch, heads, -1, nq)
    return output, dv, ste_output, statistics, maxima, denominator[:, :nq].reshape(batch, heads, nq)
