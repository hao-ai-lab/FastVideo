"""Untimed CPU FP64 small oracles and frozen primary-reference requalification."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path

import torch
from fastvideo.attention.layer import DistributedAttention_VSA
from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionMetadataBuilder
from fastvideo.attention.backends import video_sparse_attn_npu_kernels as kernels
from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, destroy_model_parallel, destroy_distributed_environment
from fastvideo.forward_context import set_forward_context
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.utils import set_mixed_precision_policy
from reference import full_vsa, error_metrics

p = argparse.ArgumentParser()
p.add_argument('--root', default='.')
p.add_argument('--reference-dir', type=Path)
p.add_argument('--output', required=True)
a = p.parse_args()
root = Path(a.root)
records = []
captured = {}
original = kernels.fused_topk_mask


def capture(scores, keep):
    mask = original(scores, keep)
    captured['mask'] = mask.cpu()
    return mask


kernels.fused_topk_mask = capture
torch.npu.set_device(0)
torch.set_num_threads(4)
set_mixed_precision_policy(torch.bfloat16, torch.float32)
init_distributed_environment(local_rank=0, timeout=timedelta(seconds=120))
initialize_model_parallel()


def run(xs: list[torch.Tensor | None], shape: tuple[int, int, int], sparsity: float) -> torch.Tensor:
    m = VideoSparseAttentionMetadataBuilder().build(0, shape, (1, 1, 1), sparsity, torch.device('npu:0'))
    layer = DistributedAttention_VSA(xs[0].shape[2],
                                     xs[0].shape[3],
                                     supported_attention_backends=(AttentionBackendEnum.VIDEO_SPARSE_ATTN, ))
    device = [None if x is None else x.to('npu') for x in xs]
    with set_forward_context(0, m):
        out = layer(*device[:3], original_seq_len=math.prod(shape), gate_compress=device[3], attn_metadata=m)[0]
    return out.cpu()


try:
    with torch.inference_mode():
        for shape in [(1, 7, 9), (1, 5, 13), (3, 5, 7)]:
            for gate in ['absent', 'zero', 'one', 'random']:
                gen = torch.Generator().manual_seed(1024)
                xs = [(torch.randn(1, math.prod(shape), 4, 64, generator=gen, dtype=torch.float64) * .3).to(
                    torch.bfloat16) for _ in range(4)]
                if gate == 'absent':
                    xs[3] = None
                if gate == 'zero':
                    xs[3].zero_()
                if gate == 'one':
                    xs[3].fill_(1)
                expected, mask = full_vsa(*[None if x is None else x.double() for x in xs], shape, .8, cpu64=True)
                out = run(xs, shape, .8)
                metrics = error_metrics(out, expected)
                agree = torch.equal(captured['mask'], mask)
                records.append({
                    'scope':
                    'CPU_FP64_small',
                    'shape':
                    shape,
                    'gate':
                    gate,
                    'metrics':
                    metrics,
                    'mask_agreement':
                    agree,
                    'status':
                    'PASS' if metrics['allclose'] and metrics['cosine_in_range'] and agree else 'FAILED'
                })
        for seed in [1024, 1025, 1026]:
            gen = torch.Generator().manual_seed(seed)
            xs = [(torch.randn(1, 32760, 12, 128, generator=gen) * .3).to(torch.bfloat16) for _ in range(4)]
            digest = hashlib.sha256(b''.join(x.view(torch.uint16).numpy().tobytes() for x in xs)).hexdigest()
            reference_dir = a.reference_dir or root / 'inputs/references/phase1-exact'
            path = reference_dir / f'f81-d128-s80-bfloat16-seed{seed}.pt'
            ref = torch.load(path, weights_only=True)
            assert digest == ref['input_hash']
            out = run(xs, (21, 30, 52), .8)
            metrics = error_metrics(out, ref['output'])
            agree = torch.equal(captured['mask'], ref['mask'])
            records.append({
                'scope':
                'frozen_primary',
                'seed':
                seed,
                'input_hash':
                digest,
                'reference_sha256':
                hashlib.sha256(path.read_bytes()).hexdigest(),
                'metrics':
                metrics,
                'mask_agreement':
                agree,
                'status':
                'PASS' if metrics['allclose'] and metrics['cosine_in_range'] and agree else 'FAILED'
            })
finally:
    destroy_model_parallel()
    destroy_distributed_environment()
Path(a.output).write_text(json.dumps(records, indent=2) + '\n')
print({status: sum(r['status'] == status for r in records) for status in ['PASS', 'FAILED']})
