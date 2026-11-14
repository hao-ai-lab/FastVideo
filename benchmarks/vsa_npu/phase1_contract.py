"""Replay all Phase 1 adversarial schedules against the integrated JIT code.

Only 64/64 is exposed by production dispatch. Other schedules verify the frozen
experimental contract without widening the supported backend domain.
"""
import argparse
import json
import os
from pathlib import Path
import traceback

import torch
from fastvideo.attention.backends.video_sparse_attn_npu_kernels import (
    _attn_fwd_sparse,
    _fused_block_mean_fwd,
    fused_topk_mask,
)
from reference import full_vsa

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
torch.npu.set_device(0)
torch.set_num_threads(4)
os.environ['VSA_NATIVE_GATHER'] = '1'
os.environ['VSA_GATHER_BATCH'] = '32'
gen = torch.Generator().manual_seed(1026)
records = []


def launch(q, k, v, selected, sizes, counts):
    b, h, s, d = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((b, h, s), device=q.device, dtype=torch.float32)
    _attn_fwd_sparse[(s // bm, b * h, 1)](
        q,
        k,
        v,
        d**-.5,
        selected,
        counts,
        selected.shape[-1],
        sizes,
        lse,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        b,
        h,
        s,
        k.shape[2],
        HEAD_DIM=d,
        BLOCK_M=bm,
        BLOCK_N=bn,
        STAGE=3,
        num_stages=1,
        num_warps=4,
    )
    return out


launch.compression = _fused_block_mean_fwd  # type: ignore[attr-defined]
launch.topk_mask = fused_topk_mask  # type: ignore[attr-defined]
with torch.inference_mode():
    for mode in ['irregular_stride', 'ties', 'near_ties', 'large_scores', 'reuse']:
        xs = [torch.randn(1, 105, 4, 128, generator=gen)[..., ::2].to('npu', dtype=torch.bfloat16) for _ in range(4)]
        if mode == 'irregular_stride':
            xs = [torch.stack([x, x], -1)[..., 0] for x in xs]
        if mode == 'ties':
            xs[0].zero_()
        if mode == 'near_ties':
            xs[0].mul_(1e-4)
            xs[1].fill_(1)
            xs[1][:, 0].add_(.0078125)
        if mode == 'large_scores':
            xs[0].mul_(100)
            xs[1].mul_(100)
            xs[2].mul_(.3)
        before = [x.cpu().clone() for x in xs]
        ref, rm = full_vsa(*xs, (3, 5, 7), .6)
        for bm, bn in [(32, 32), (32, 64), (64, 32), (64, 64)]:
            r = {'mode': mode, 'block_m': bm, 'block_n': bn, 'production_schedule': (bm, bn) == (64, 64)}
            try:
                out, mask = full_vsa(*xs, (3, 5, 7), .6, candidate=launch)
                again, am = full_vsa(*xs, (3, 5, 7), .6, candidate=launch)
                r.update(max_abs_error=(out - ref).abs().max().item(),
                         mask_agreement=torch.equal(mask, rm),
                         finite=torch.isfinite(out).all().item(),
                         repeat_equal=torch.equal(out, again) and torch.equal(mask, am),
                         input_preserved=all(torch.equal(x.cpu(), old) for x, old in zip(xs, before, strict=False)),
                         correctness=torch.allclose(out, ref, atol=.02, rtol=.02))
                r['status'] = 'PASS' if all(
                    r[k] for k in ['mask_agreement', 'finite', 'repeat_equal', 'input_preserved', 'correctness'
                                   ]) else 'FAILED'
            except Exception:
                r.update(status='FAILED', reason=traceback.format_exc())
            records.append(r)
        Path(a.output).write_text(json.dumps(records, indent=2) + '\n')
    x = torch.zeros(1, 4, 64, 64, device='npu', dtype=torch.bfloat16)
    zero = torch.zeros(1, device='npu', dtype=torch.int32)
    mean = _fused_block_mean_fwd(x, zero, 64)
    records.append({
        'mode': 'zero_valid_block',
        'expected': 'undefined nonfinite compression; invalid pipeline input',
        'status': 'PASS' if not torch.isfinite(mean).all().item() else 'FAILED'
    })
Path(a.output).write_text(json.dumps(records, indent=2) + '\n')
print({s: sum(r['status'] == s for r in records) for s in ['PASS', 'FAILED']})
