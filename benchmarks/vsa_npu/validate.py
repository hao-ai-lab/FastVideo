"""Local SP1/SP4 numerical runner. Launch each rank with explicit visibility."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from fastvideo.attention.layer import DistributedAttention_VSA
from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionMetadataBuilder
from fastvideo.distributed.parallel_state import (init_distributed_environment, initialize_model_parallel,
                                                  destroy_model_parallel, destroy_distributed_environment)
from fastvideo.forward_context import set_forward_context
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.utils import set_mixed_precision_policy
from reference import full_vsa, error_metrics

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--oracle-dir', default='.', help='Legacy compatibility argument; references are generated locally')
p.add_argument('--suite', default='tiny')
a = p.parse_args()

rank = int(os.environ.get('RANK', '0'))
world = int(os.environ.get('WORLD_SIZE', '1'))
torch.npu.set_device(0)
torch.set_num_threads(4)
set_mixed_precision_policy(torch.bfloat16, torch.float32)
records = []
failure = None
head_divisibility_rejection_verified = world == 1
try:
    init_distributed_environment(world_size=world,
                                 rank=rank,
                                 local_rank=0,
                                 backend='hccl',
                                 timeout=timedelta(seconds=120))
    initialize_model_parallel(sequence_model_parallel_size=world, backend='hccl')
    if world == 4:
        try:
            DistributedAttention_VSA(10, 64, supported_attention_backends=(AttentionBackendEnum.VIDEO_SPARSE_ATTN, ))
        except NotImplementedError as exc:
            head_divisibility_rejection_verified = 'divisible' in str(exc)
        assert head_divisibility_rejection_verified, 'SP4 must reject nondivisible heads before attention collectives'
    if os.environ.get('VSA_FAIL_RANK') == str(rank):
        os._exit(42)
    from fastvideo.attention.backends import video_sparse_attn_npu_kernels as kernels
    original_topk = kernels.fused_topk_mask
    captured = {}

    def capture(scores, keep):
        mask = original_topk(scores, keep)
        captured['mask'] = mask.clone()
        return mask

    kernels.fused_topk_mask = capture
    cases = [((3, 5, 7), 4, 64, s, g, m) for s in [.5, .8, .9] for g in ['absent', 'zero', 'one', 'random']
             for m in [.3, 1., 3.]]
    if a.suite == 'smoke':
        cases = cases[:1]
    if a.suite == 'frames':
        cases = [((t, 30, 52), 12, 128, s, g, .3) for t in [5, 9, 21] for s in [.5, .8, .9]
                 for g in ['absent', 'zero', 'one', 'random']]
    if a.suite == 'magnitudes':
        cases = [((t, 30, 52), 12, d, .8, 'random', mag) for t in [5, 9, 21] for d in [64, 128] for mag in [.3, 1., 3.]]
    modes = ['normal'] * len(cases)
    if a.suite == 'contract':
        cases = [(shape, 4, d, .6, g, .3) for shape in [(1, 7, 9), (1, 8, 8), (1, 5, 13), (3, 5, 7)] for d in [64, 128]
                 for g in ['absent', 'zero', 'one', 'random']]
        modes = ['normal'] * len(cases)
        for mode in ['irregular_stride', 'ties', 'near_ties', 'large_scores', 'reuse']:
            cases.append(((3, 5, 7), 4, 64, .6, 'random', 1.))
            modes.append(mode)
    with torch.inference_mode():
        for (shape, heads, d, sparsity, gate, magnitude), mode in zip(cases, modes, strict=False):
            gen = torch.Generator().manual_seed(1024)
            xs = [(torch.randn(1, math.prod(shape), heads, d, generator=gen) * magnitude).to(torch.bfloat16)
                  for _ in range(4)]
            digest = hashlib.sha256(b''.join(x.view(torch.uint16).numpy().tobytes() for x in xs)).hexdigest()
            if gate == 'zero':
                xs[3].zero_()
            if gate == 'one':
                xs[3].fill_(1)
            if gate == 'absent':
                xs = xs[:3] + [None]
            device_xs = [None if x is None else x.to('npu') for x in xs]
            if mode == 'irregular_stride':
                device_xs = [None if x is None else torch.stack([x, x], -1)[..., 0] for x in device_xs]
            if mode == 'ties':
                device_xs[0].zero_()
            if mode == 'near_ties':
                assert device_xs[1] is not None
                device_xs[0].mul_(1e-4)
                device_xs[1].fill_(1)
                device_xs[1][:, 0].add_(.0078125)
            if mode == 'large_scores':
                device_xs[0].mul_(100)
                device_xs[1].mul_(100)
                device_xs[2].mul_(.3)
            original_xs = [None if x is None else x.clone() for x in device_xs]
            moments = {
                name: None if x is None else {
                    'mean': x.float().mean().item(),
                    'std': x.float().std().item(),
                    'abs_max': x.float().abs().max().item()
                }
                for name, x in zip(['q', 'k', 'v', 'gate'], device_xs, strict=False)
            }
            actual_digest = hashlib.sha256(b''.join(
                b'ABSENT' if x is None else x.cpu().contiguous().view(torch.uint16).numpy().tobytes()
                for x in device_xs)).hexdigest()
            metadata = VideoSparseAttentionMetadataBuilder().build(0, shape, (1, 1, 1), sparsity, torch.device('npu:0'))
            layer = DistributedAttention_VSA(heads,
                                             d,
                                             supported_attention_backends=(AttentionBackendEnum.VIDEO_SPARSE_ATTN, ))
            n = math.prod(shape)
            shard = math.ceil(n / world)
            padded = shard * world
            parts = [
                None if x is None else torch.nn.functional.pad(x,
                                                               (0, 0, 0, 0, 0, padded - n))[:, rank * shard:(rank + 1) *
                                                                                            shard].contiguous()
                for x in device_xs
            ]
            start = time.perf_counter()
            with set_forward_context(0, metadata):
                out, _ = layer(*parts[:3], original_seq_len=n, gate_compress=parts[3], attn_metadata=metadata)
            torch.npu.synchronize()
            first = (time.perf_counter() - start) * 1000
            gathered = [torch.empty_like(out) for _ in range(world)]
            torch.distributed.all_gather(gathered, out)
            actual = torch.cat(gathered, dim=1)[:, :n]
            os.environ['VSA_NATIVE_GATHER'] = '1'
            os.environ['VSA_GATHER_BATCH'] = '4'
            expected, mask = full_vsa(*device_xs, shape, sparsity)
            torch.npu.synchronize()
            diff = (actual.float() - expected.float()).abs()
            okay = torch.allclose(actual, expected, atol=.02, rtol=.02)
            numerical = error_metrics(actual, expected)
            expected_mask = mask[:, rank * (heads // world):(rank + 1) * (heads // world)]
            mask_equal = torch.equal(captured['mask'], expected_mask)
            saved = actual.clone()
            raw_saved = out.clone()
            with set_forward_context(0, metadata):
                repeat, _ = layer(*parts[:3], original_seq_len=n, gate_compress=parts[3], attn_metadata=metadata)
            torch.npu.synchronize()
            repeat_equal = torch.equal(out, repeat)
            workspace_distinct = out.untyped_storage().data_ptr() != repeat.untyped_storage().data_ptr()
            workspace_distinct = workspace_distinct and all(
                x is None or out.untyped_storage().data_ptr() != x.untyped_storage().data_ptr() for x in parts)
            input_preserved = all(x is None or torch.equal(x, old)
                                  for x, old in zip(device_xs, original_xs, strict=False))
            changed_input_reuse = True
            if mode == 'reuse':
                changed = [None if x is None else x.clone() for x in device_xs]
                changed[0].mul_(1.5)
                changed[1].neg_()
                changed_parts = [
                    None if x is None else torch.nn.functional.pad(x, (0, 0, 0, 0, 0, padded - n))[:, rank *
                                                                                                   shard:(rank + 1) *
                                                                                                   shard].contiguous()
                    for x in changed
                ]
                with set_forward_context(0, metadata):
                    new_out, _ = layer(*changed_parts[:3],
                                       original_seq_len=n,
                                       gate_compress=changed_parts[3],
                                       attn_metadata=metadata)
                new_parts = [torch.empty_like(new_out) for _ in range(world)]
                torch.distributed.all_gather(new_parts, new_out)
                fresh, freshmask = full_vsa(*changed, shape, sparsity)
                changed_input_reuse = torch.allclose(torch.cat(
                    new_parts, 1)[:, :n], fresh, atol=.02, rtol=.02) and torch.equal(
                        captured['mask'], freshmask[:, rank * (heads // world):(rank + 1) *
                                                    (heads // world)]) and torch.equal(actual, saved) and torch.equal(
                                                        out, raw_saved)
                workspace_distinct = workspace_distinct and out.untyped_storage().data_ptr() != new_out.untyped_storage(
                ).data_ptr()
            records.append({
                'mode':
                mode,
                'shape':
                shape,
                'heads':
                heads,
                'head_dim':
                d,
                'sparsity':
                sparsity,
                'gate':
                gate,
                'magnitude':
                magnitude,
                'input_hash_before_gate_override':
                digest,
                'max_abs_error':
                diff.max().item(),
                'allclose':
                okay,
                'actual_input_hash':
                actual_digest,
                'changed_input_reuse':
                changed_input_reuse,
                'metrics':
                numerical,
                'input_moments':
                moments,
                'repeat_equal':
                repeat_equal,
                'output_preserved':
                torch.equal(saved, actual) and torch.equal(out, raw_saved),
                'workspace_distinct':
                workspace_distinct,
                'mask_agreement':
                mask_equal,
                'input_preserved':
                input_preserved,
                'finite':
                torch.isfinite(actual).all().item(),
                'first_call_wall_ms':
                first,
                'peak_memory_bytes':
                torch.npu.max_memory_allocated(),
                'status':
                'PASS' if okay and repeat_equal and mask_equal and input_preserved and changed_input_reuse
                and workspace_distinct and torch.isfinite(actual).all().item() else 'FAILED'
            })
            print(rank, records[-1], flush=True)
except Exception:
    failure = traceback.format_exc()
    traceback.print_exc()
finally:
    try:
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        if failure is None:
            failure = traceback.format_exc()
    result = {
        'rank': rank,
        'world_size': world,
        'physical_device': os.environ['ASCEND_RT_VISIBLE_DEVICES'],
        'pid': os.getpid(),
        'records': records,
        'failure': failure,
        'head_divisibility_rejection_verified': head_divisibility_rejection_verified,
        'status': 'FAILED' if failure or any(r['status'] != 'PASS' for r in records) else 'PASS'
    }
    Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
sys.exit(result['status'] != 'PASS')
