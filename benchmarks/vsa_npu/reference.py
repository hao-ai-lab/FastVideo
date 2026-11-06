# SPDX-License-Identifier: Apache-2.0
"""Frozen local complete-VSA and preselected sparse numerical references."""
import functools
import os
import time
import torch
import torch_npu
from typing import Any
from collections.abc import Callable

LAST_STAGES: dict[str, float] = {}


def error_metrics(actual, expected):
    x = actual.cpu().double().flatten()
    y = expected.cpu().double().flatten()
    cosine = (torch.dot(x, y) / (torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y))).item()
    error = (x - y).abs()
    return {
        'max_abs_error': (x - y).abs().max().item(),
        'max_rel_error': ((x - y).abs() / y.abs().clamp_min(1e-8)).max().item(),
        'max_error_over_allowed_atol_rtol': (error / (.02 + .02 * y.abs())).max().item(),
        'cosine_similarity_fp64': cosine,
        'cosine_in_range': -1 - 1e-12 <= cosine <= 1 + 1e-12,
        'allclose': torch.allclose(x, y, atol=.02, rtol=.02)
    }


def threshold_mask(scores: torch.Tensor, keep: int) -> torch.Tensor:
    values = scores.float()
    lo = values.amin(-1, keepdim=True)
    hi = values.amax(-1, keepdim=True)
    for _ in range(32):
        mid = (lo + hi) * .5
        ge = (values >= mid).sum(-1, keepdim=True, dtype=torch.int32) >= keep
        lo = torch.where(ge, mid, lo)
        hi = torch.where(ge, hi, mid)
    above = values > lo
    equal = values == lo
    return above | (
        equal & (equal.int().cumsum(-1, dtype=torch.int32) <= keep - above.sum(-1, keepdim=True, dtype=torch.int32)))


@functools.lru_cache(maxsize=16)
def metadata(shape, device):
    t, h, w = shape
    original = torch.arange(t * h * w).reshape(t, h, w)
    tiles = [
        original[a:a + 4, b:b + 4, c:c + 4].flatten() for a in range(0, t, 4) for b in range(0, h, 4)
        for c in range(0, w, 4)
    ]
    order = torch.cat(tiles)
    sizes = torch.tensor([x.numel() for x in tiles], device=device)
    slots = torch.cat([torch.arange(x.numel()) + 64 * i for i, x in enumerate(tiles)]).to(device)
    return order.to(device), sizes, slots, torch.argsort(order).to(device)


def full_vsa(q: torch.Tensor,
             k: torch.Tensor,
             v: torch.Tensor,
             g: torch.Tensor | None,
             shape: tuple[int, int, int],
             sparsity: float,
             cpu64: bool = False,
             candidate: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
    LAST_STAGES.clear()
    last = [time.perf_counter()]

    def trace(stage: str) -> None:
        if os.environ.get('VSA_PROFILE') == '1':
            if q.device.type == 'npu':
                torch.npu.synchronize()
            now = time.perf_counter()
            LAST_STAGES[stage] = (now - last[0]) * 1000
            last[0] = now
        if os.environ.get('VSA_TRACE') == '1':
            if q.device.type == 'npu':
                torch.npu.synchronize()
            print(stage, time.time(), flush=True)

    trace('metadata start')
    order, sizes, slots, reverse = metadata(shape, q.device)
    trace('metadata done')
    n = sizes.numel()
    d = q.shape[-1]

    def tile(x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros((x.shape[0], x.shape[2], n * 64, d), dtype=x.dtype, device=x.device)
        y[:, :, slots, :] = x[:, order].transpose(1, 2)
        return y.contiguous()

    q, k, v = map(tile, (q, k, v))
    gate = tile(g) if g is not None else None
    trace('tiling done')
    cast: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.to(torch.bfloat16).double() if cpu64 else x
    if candidate and hasattr(candidate, 'compression'):
        qc, kc, vc = [candidate.compression(x, sizes.to(torch.int32), 64) for x in (q, k, v)]
    else:
        means = [(x.reshape(x.shape[0], x.shape[1], n, 64, d).to(torch.float64 if cpu64 else torch.float32).sum(-2) /
                  sizes[None, None, :, None]).to(x.dtype) for x in (q, k, v)]
        qc, kc, vc = map(cast, means)
    scores = cast(cast(qc @ kc.transpose(-1, -2)) / (d**0.5))
    coarse = cast(cast(scores.softmax(-1)) @ vc)
    trace('coarse done')
    keep = max(1, min(__import__('math').ceil((1 - sparsity) * n), n))
    # <=4096 upstream tie rule: lower key index first at equal threshold.
    if candidate and hasattr(candidate, 'topk_mask'):
        blockmask = candidate.topk_mask(scores, keep)
    else:
        blockmask = threshold_mask(scores, keep)
    counts = blockmask.sum(-1, dtype=torch.int32)
    keep = int(counts.max().item())
    keyids = torch.arange(n, device=q.device, dtype=torch.float32)
    selected = torch.where(blockmask, keyids, float(n)).sort(-1).values[..., :keep].clamp_max(n - 1).to(torch.int32)
    trace('selection done')
    trace('sparse start')
    if candidate:
        sparse = candidate(q, k, v, selected, sizes.to(torch.int32), counts)
    else:
        valid = (torch.arange(n * 64, device=q.device) % 64) < sizes.repeat_interleave(64)

        def query_mask(start: int, end: int) -> torch.Tensor:
            return blockmask[:, :, (torch.arange(start, end, device=q.device) // 64), :].repeat_interleave(
                64, -1) & valid[None, None, None, :]

        if cpu64:
            mask = query_mask(0, n * 64)
            sparse = ((q @ k.transpose(-1, -2) / (d**0.5)).masked_fill(~mask, -float('inf')).softmax(-1)) @ v
        else:
            sparse = torch.empty_like(q)
            if os.environ.get('VSA_NATIVE_GATHER') == '1' and q.device.type == 'npu':
                import torch_npu
                # Batch independent query blocks; each keeps its entire selected domain.
                ordered = selected.float().sort(-1).values.to(torch.int32)
                offsets = torch.arange(64, device=q.device, dtype=torch.int32)
                batch_blocks = int(os.environ.get('VSA_GATHER_BATCH', '4'))
                for begin in range(0, n, batch_blocks):
                    count = min(batch_blocks, n - begin)
                    ids = ordered[:, :, begin:begin + count, :, None] * 64 + offsets
                    ids = ids.reshape(q.shape[0], q.shape[1], count, keep * 64).long()
                    kk = k[:, :, None].expand(-1, -1, count, -1, -1).gather(3, ids[..., None].expand(-1, -1, -1, -1, d))
                    vv = v[:, :, None].expand(-1, -1, count, -1, -1).gather(3, ids[..., None].expand(-1, -1, -1, -1, d))
                    qq = q[:, :, begin * 64:(begin + count) * 64].reshape(q.shape[0], q.shape[1], count, 64, d)
                    # B,H,block -> block-batch,H,sequence,D.
                    qq = qq.permute(0, 2, 1, 3, 4).reshape(-1, q.shape[1], 64, d).contiguous()
                    kk = kk.permute(0, 2, 1, 3, 4).reshape(-1, q.shape[1], keep * 64, d).contiguous()
                    vv = vv.permute(0, 2, 1, 3, 4).reshape_as(kk).contiguous()
                    validkeys = ((ids % 64) < sizes[(ids // 64)]) & (
                        (torch.arange(keep * 64, device=q.device) // 64)[None, None, None, :]
                        < counts[:, :, begin:begin + count, None])
                    mm = (~validkeys).permute(0, 2, 1, 3).reshape(-1, q.shape[1], 1, keep * 64).expand(-1, -1, 64,
                                                                                                       -1).contiguous()
                    oo = torch_npu.npu_fusion_attention(qq,
                                                        kk,
                                                        vv,
                                                        q.shape[1],
                                                        'BNSD',
                                                        atten_mask=mm,
                                                        scale=d**-.5,
                                                        keep_prob=1.,
                                                        sparse_mode=1)[0]
                    sparse[:, :,
                           begin * 64:(begin + count) * 64] = oo.reshape(q.shape[0], count, q.shape[1], 64, d).permute(
                               0, 2, 1, 3, 4).reshape(q.shape[0], q.shape[1], count * 64, d)
                coarse = coarse.repeat_interleave(64, -2)
                trace('sparse done')
                result = sparse + coarse if gate is None else torch.addcmul(sparse, coarse, gate)
                result = result[:, :, slots][:, :, reverse].transpose(1, 2)
                trace('gating restoration done')
                return result, blockmask
            chunk = int(os.environ.get('VSA_QUERY_CHUNK', '256'))
            for start in range(0, n * 64, chunk):
                end = min(start + chunk, n * 64)
                mask = query_mask(start, end)
                trace('mask done ' + str(start))
                if os.environ.get('VSA_NATIVE_FUSED') == '1' and q.device.type == 'npu':
                    import torch_npu
                    sparse[:, :, start:end] = torch_npu.npu_fusion_attention(q[:, :, start:end].contiguous(),
                                                                             k,
                                                                             v,
                                                                             q.shape[1],
                                                                             'BNSD',
                                                                             atten_mask=~mask,
                                                                             scale=d**-.5,
                                                                             keep_prob=1.,
                                                                             sparse_mode=1)[0]
                else:
                    sparse[:, :, start:end] = torch.nn.functional.scaled_dot_product_attention(q[:, :, start:end],
                                                                                               k,
                                                                                               v,
                                                                                               attn_mask=mask)
                trace('attention done ' + str(start))
    coarse = coarse.repeat_interleave(64, -2)
    trace('sparse done')
    result = sparse + coarse if gate is None else torch.addcmul(sparse, coarse, gate)
    result = result[:, :, slots][:, :, reverse].transpose(1, 2)
    trace('gating restoration done')
    return result, blockmask


def native(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, selected: torch.Tensor, sizes: torch.Tensor,
           counts: torch.Tensor) -> torch.Tensor:
    b, h, s, d = q.shape
    n = s // 64
    keep = selected.shape[-1]
    sparse = torch.empty_like(q)
    ordered = selected.float().sort(-1).values.to(torch.int32)
    offsets = torch.arange(64, device=q.device, dtype=torch.int32)
    batch_blocks = int(os.environ['VSA_GATHER_BATCH'])
    for begin in range(0, n, batch_blocks):
        count = min(batch_blocks, n - begin)
        ids = ordered[:, :, begin:begin + count, :, None] * 64 + offsets
        ids = ids.reshape(b, h, count, keep * 64).long()
        kk = k[:, :, None].expand(-1, -1, count, -1, -1).gather(3, ids[..., None].expand(-1, -1, -1, -1, d))
        vv = v[:, :, None].expand(-1, -1, count, -1, -1).gather(3, ids[..., None].expand(-1, -1, -1, -1, d))
        qq = q[:, :, begin * 64:(begin + count) * 64].reshape(b, h, count, 64, d)
        qq = qq.permute(0, 2, 1, 3, 4).reshape(-1, h, 64, d).contiguous()
        kk = kk.permute(0, 2, 1, 3, 4).reshape(-1, h, keep * 64, d).contiguous()
        vv = vv.permute(0, 2, 1, 3, 4).reshape_as(kk).contiguous()
        validkeys = ((ids % 64) < sizes[(ids // 64)]) & (
            (torch.arange(keep * 64, device=q.device) // 64)[None, None, None, :] < counts[:, :, begin:begin + count,
                                                                                           None])
        mm = (~validkeys).permute(0, 2, 1, 3).reshape(-1, h, 1, keep * 64).expand(-1, -1, 64, -1).contiguous()
        oo = torch_npu.npu_fusion_attention(qq,
                                            kk,
                                            vv,
                                            h,
                                            'BNSD',
                                            atten_mask=mm,
                                            scale=d**-.5,
                                            keep_prob=1.,
                                            sparse_mode=1)[0]
        sparse[:, :, begin * 64:(begin + count) * 64] = oo.reshape(b, count, h, 64,
                                                                   d).permute(0, 2, 1, 3,
                                                                              4).reshape(b, h, count * 64, d)
    return sparse
