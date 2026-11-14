"""One actual SP4 replica in a coordinated four-replica throughput run."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import statistics
import hashlib
import math

import torch
from fastvideo.attention.layer import DistributedAttention_VSA
from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionMetadataBuilder
from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, destroy_model_parallel, destroy_distributed_environment, get_world_group
from fastvideo.forward_context import set_forward_context
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.utils import set_mixed_precision_policy
from fastvideo.attention.backends.video_sparse_attn_npu import NPUVideoSparseAttentionImpl
from reference import native, threshold_mask


class NativeImpl(NPUVideoSparseAttentionImpl):

    def forward(self, q, k, v, g, m):
        q, k, v = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
        b, h, s, d = q.shape
        n = s // 64
        sizes = m.variable_block_sizes.to(torch.int32)
        qc, kc, vc = [(x.view(b, h, n, 64, d).float().sum(-2) / sizes[None, None, :, None]).to(x.dtype)
                      for x in (q, k, v)]
        scores = (qc @ kc.transpose(-1, -2)) / math.sqrt(d)
        coarse = (scores.softmax(-1) @ vc).repeat_interleave(64, -2)
        mask = threshold_mask(scores, max(1, min(math.ceil((1 - m.VSA_sparsity) * n), n)))
        counts = mask.sum(-1, dtype=torch.int32)
        maximum = int(counts.max().item())
        ids = torch.where(mask, torch.arange(n, device=q.device, dtype=torch.float32),
                          float(n)).sort(-1).values[..., :maximum].clamp_max(n - 1).to(torch.int32)
        sparse = native(q, k, v, ids, sizes, counts)
        out = coarse + sparse if g is None else torch.addcmul(sparse, coarse, g.transpose(1, 2))
        return out.transpose(1, 2)


p = argparse.ArgumentParser()
p.add_argument('--root', default='.')
p.add_argument('--reference-dir', type=Path)
p.add_argument('--output', required=True)
a = p.parse_args()
rank = int(os.environ['RANK'])
device = os.environ['ASCEND_RT_VISIBLE_DEVICES']
role = os.environ.get('VSA_REPLICA_ROLE', 'candidate')
assert role in ('candidate', 'reference')
os.environ['VSA_GATHER_BATCH'] = '32'
torch.npu.set_device(0)
torch.set_num_threads(4)
set_mixed_precision_policy(torch.bfloat16, torch.float32)
init_distributed_environment(world_size=4, rank=rank, local_rank=0, timeout=timedelta(seconds=180))
initialize_model_parallel(sequence_model_parallel_size=4, backend='hccl')
try:
    with torch.inference_mode():
        gen = torch.Generator().manual_seed(1024)
        cpu = [(torch.randn(1, 32760, 12, 128, generator=gen) * .3).to(torch.bfloat16) for _ in range(4)]
        digest = hashlib.sha256(b''.join(x.view(torch.uint16).numpy().tobytes() for x in cpu)).hexdigest()
        xs = [x.to('npu')[:, rank * 8190:(rank + 1) * 8190].contiguous() for x in cpu]
        del cpu
        m = VideoSparseAttentionMetadataBuilder().build(0, (21, 30, 52), (1, 1, 1), .8, torch.device('npu:0'))
        layer = DistributedAttention_VSA(12,
                                         128,
                                         supported_attention_backends=(AttentionBackendEnum.VIDEO_SPARSE_ATTN, ))
        if role == 'reference':
            layer.attn_impl = NativeImpl(12, 128, False, 128**-.5)

        def call() -> torch.Tensor:
            with set_forward_context(0, m):
                return layer(*xs[:3], original_seq_len=32760, gate_compress=xs[3], attn_metadata=m)[0]

        warms = []
        stable = False
        for _ in range(50):
            torch.npu.synchronize()
            t = time.perf_counter()
            out = call()
            torch.npu.synchronize()
            warms.append((time.perf_counter() - t) * 1000)
            if len(warms) >= 10 and len(warms) % 5 == 0:
                flag = torch.tensor(
                    int(abs(statistics.median(warms[-5:]) / statistics.median(warms[-10:-5]) - 1) <= .02),
                    dtype=torch.int32)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=get_world_group().cpu_group)
                if flag.item():
                    stable = True
                    break
        torch.npu.synchronize()
        reference_dir = a.reference_dir or Path(a.root) / 'inputs/references/phase1-exact'
        reference_path = reference_dir / 'f81-d128-s80-bfloat16-seed1024.pt'
        reference = torch.load(reference_path, weights_only=True)
        assert digest == reference['input_hash']
        reference_slice = reference['output'][:, rank * 8190:(rank + 1) * 8190].to('npu')
        correct = torch.allclose(out, reference_slice, atol=.02, rtol=.02)
        error = (out.float() - reference_slice.float()).abs().max().item()
        assert correct, 'SP4 replica differs from frozen local SP1 complete reference'
        barrier = Path(os.environ['VSA_REPLICA_BARRIER'])
        barrier.mkdir(parents=True, exist_ok=True)
        (barrier / (device + '.json')).write_text(json.dumps({'pid': os.getpid(), 'ready': time.time()}))
        deadline = time.monotonic() + 900
        while len(list(barrier.glob('*.json'))) < 16:
            if time.monotonic() > deadline:
                raise TimeoutError('global 16-rank prewarm barrier')
            time.sleep(.05)
        torch.distributed.barrier(group=get_world_group().cpu_group)
        start = time.time()
        wall = []
        for _ in range(30):
            torch.npu.synchronize()
            t = time.perf_counter()
            out = call()
            torch.npu.synchronize()
            wall.append((time.perf_counter() - t) * 1000)
        end = time.time()
        maximum = torch.tensor(wall, dtype=torch.float64)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=get_world_group().cpu_group)
        result = {
            'status': 'PASS' if stable else 'INCONCLUSIVE',
            'role': role,
            'input_hash': digest,
            'rank': rank,
            'physical_device': device,
            'pid': os.getpid(),
            'start': start,
            'end': end,
            'calls': 30,
            'wall_ms_samples': wall,
            'max_rank_wall_ms_samples': maximum.tolist(),
            'warmup_ms': warms,
            'stable': stable,
            'correctness': correct,
            'max_abs_error': error,
            'reference_input_hash': reference['input_hash'],
            'peak_memory_bytes': torch.npu.max_memory_allocated(),
            'finite': torch.isfinite(out).all().item()
        }
        Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
finally:
    destroy_model_parallel()
    destroy_distributed_environment()
