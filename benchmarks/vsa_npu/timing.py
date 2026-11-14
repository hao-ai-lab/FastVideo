"""Paired integrated SP1/SP4 timings with boundary synchronization and events."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from functools import partial
from collections.abc import Callable
from typing import Any

import torch
import torch_npu
from fastvideo.attention.layer import DistributedAttention, DistributedAttention_VSA
from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionMetadataBuilder
from fastvideo.attention.backends.video_sparse_attn_npu import NPUVideoSparseAttentionImpl, sparse_forward
from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, destroy_model_parallel, destroy_distributed_environment, get_world_group
from fastvideo.forward_context import set_forward_context
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.utils import set_mixed_precision_policy
from measure import benchmark, timed, jit_count
from reference import native, full_vsa, threshold_mask, metadata as oracle_metadata

p = argparse.ArgumentParser()
p.add_argument('--root', default='.', help='Legacy provenance root; no external timing or reference files required')
p.add_argument('--input-manifest', type=Path, default=Path(__file__).with_name('primary_inputs.json'))
p.add_argument('--output', required=True)
p.add_argument('--mode', choices=['primary', 'curve', 'sp4', 'sp4curve'], default='primary')
a = p.parse_args()
root = Path(a.root).resolve()

rank = int(os.environ.get('RANK', '0'))
world = int(os.environ.get('WORLD_SIZE', '1'))
torch.npu.set_device(0)
torch.set_num_threads(4)
set_mixed_precision_policy(torch.bfloat16, torch.float32)
os.environ['VSA_NATIVE_GATHER'] = '1'
os.environ['VSA_GATHER_BATCH'] = '32'
init_distributed_environment(world_size=world, rank=rank, local_rank=0, timeout=timedelta(seconds=180))
initialize_model_parallel(sequence_model_parallel_size=world, backend='hccl')


def measure(fn: Callable[[], torch.Tensor]) -> dict[str, Any]:
    if world == 1:
        return benchmark(fn)
    group = get_world_group().cpu_group
    warms = []
    stable = False
    started = time.perf_counter()
    before = jit_count()
    for i in range(50):
        _, wall, _ = timed(fn)
        warms.append(wall)
        if len(warms) >= 10 and len(warms) % 5 == 0:
            good = abs(statistics.median(warms[-5:]) / statistics.median(warms[-10:-5]) - 1) <= .02
            flag = torch.tensor(int(good), dtype=torch.int32)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group)
            if flag.item():
                stable = True
                break
    walls = []
    events = []
    maxima = []
    sample_started = time.perf_counter()
    for _ in range(30):
        _, wall, event = timed(fn, True)
        walls.append(wall)
        events.append(event)
        maximum = torch.tensor(wall, dtype=torch.float64)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=group)
        maxima.append(maximum.item())
    return {
        'warmup_ms': warms,
        'stable': stable,
        'warm_wall_ms_samples': walls,
        'warm_device_ms_samples': events,
        'max_rank_wall_ms_samples': maxima,
        'jit_count_before_after': [before, jit_count()],
        'sample_interval_coordinator_wall_ms_including_cpu_reporting': (time.perf_counter() - sample_started) * 1000,
        'coordinator_interval_ms_including_warmup_and_cpu_reporting': (time.perf_counter() - started) * 1000
    }


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


class DenseImpl:
    """Different-semantics dense comparator in the existing SP collectives."""

    def preprocess_qkv(self, x, metadata):
        return x

    def postprocess_output(self, x, metadata):
        return x

    def forward(self, q, k, v, metadata):
        h = q.shape[2]
        d = q.shape[3]
        return torch_npu.npu_fusion_attention(*[x.transpose(1, 2).contiguous() for x in (q, k, v)],
                                              h,
                                              'BNSD',
                                              scale=d**-.5,
                                              keep_prob=1.,
                                              sparse_mode=0)[0].transpose(1, 2).contiguous()


def distributed_call(layer: DistributedAttention_VSA, xs: list[torch.Tensor], m: Any) -> torch.Tensor:
    n = xs[0].shape[1]
    shard = math.ceil(n / world)
    padded = shard * world
    parts = [
        torch.nn.functional.pad(x, (0, 0, 0, 0, 0, padded - n))[:, rank * shard:(rank + 1) * shard].contiguous()
        for x in xs
    ]
    # Input shards are supplied by the caller in real SP; this harness also pays
    # pad/shard preparation inside each timed invocation, identically both ways.
    with set_forward_context(0, m):
        return layer(*parts[:3], original_seq_len=n, gate_compress=parts[3], attn_metadata=m)[0]


def dense_forward(xs: list[torch.Tensor], d: int) -> torch.Tensor:
    return torch_npu.npu_fusion_attention(*[x.transpose(1, 2).contiguous() for x in xs[:3]],
                                          12,
                                          'BNSD',
                                          scale=d**-.5,
                                          keep_prob=1.,
                                          sparse_mode=0)[0].transpose(1, 2).contiguous()


def native_complete(xs: list[torch.Tensor], shape: tuple[int, int, int], sparsity: float) -> torch.Tensor:
    return full_vsa(*xs, shape, sparsity)[0]


def distributed_dense_call(layer: DistributedAttention, xs: list[torch.Tensor], m: Any) -> torch.Tensor:
    n = xs[0].shape[1]
    shard = math.ceil(n / world)
    padded = shard * world
    parts = [
        torch.nn.functional.pad(x, (0, 0, 0, 0, 0, padded - n))[:, rank * shard:(rank + 1) * shard].contiguous()
        for x in xs[:3]
    ]
    with set_forward_context(0, m):
        return layer(*parts, original_seq_len=n, attn_metadata=m)[0]


def pack_tensor(x: torch.Tensor, blocks: int, order: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    packed = torch.zeros((x.shape[0], x.shape[2], blocks * 64, x.shape[3]), device=x.device, dtype=x.dtype)
    packed[:, :, slots, :] = x[:, order].transpose(1, 2)
    return packed.contiguous()


records = []
cases = [(81, 128, .8, seed) for seed in [1024, 1025, 1026]]
if a.mode in ('curve', 'sp4curve'):
    cases = [(f, d, s, 1024) for f in [17, 33, 81] for d in [64, 128] for s in [.5, .6, .7, .8, .9]]
try:
    with torch.inference_mode():
        if a.mode == 'primary' and os.environ.get('VSA_PHASE2_BARRIER'):
            for preseed in [1024, 1025, 1026]:
                gen = torch.Generator().manual_seed(preseed)
                prexs = [(torch.randn(1, 32760, 12, 128, generator=gen) * .3).to(torch.bfloat16).to('npu')
                         for _ in range(4)]
                prem = VideoSparseAttentionMetadataBuilder().build(0, (21, 30, 52), (1, 1, 1), .8,
                                                                   torch.device('npu:0'))
                prelayer = DistributedAttention_VSA(
                    12, 128, supported_attention_backends=(AttentionBackendEnum.VIDEO_SPARSE_ATTN, ))
                distributed_call(prelayer, prexs, prem)
                full_vsa(*prexs, (21, 30, 52), .8)
                torch.npu.synchronize()
                del prexs, prem, prelayer
        for frames, d, sparsity, seed in cases:
            shape = ((frames - 1) // 4 + 1, 30, 52)
            n = math.prod(shape)
            gen = torch.Generator().manual_seed(seed)
            cpu = [(torch.randn(1, n, 12, d, generator=gen) * .3).to(torch.bfloat16) for _ in range(4)]
            digest = hashlib.sha256(b''.join(x.view(torch.uint16).numpy().tobytes() for x in cpu)).hexdigest()
            xs = [x.to('npu') for x in cpu]
            del cpu
            if a.mode == 'primary':
                expected_inputs = json.loads(a.input_manifest.read_text())
                assert expected_inputs['input_sha256'][str(seed)] == digest
            m = VideoSparseAttentionMetadataBuilder().build(0, shape, (1, 1, 1), sparsity, torch.device('npu:0'))
            layer = DistributedAttention_VSA(12,
                                             d,
                                             supported_attention_backends=(AttentionBackendEnum.VIDEO_SPARSE_ATTN, ))
            ref_layer = DistributedAttention_VSA(
                12, d, supported_attention_backends=(AttentionBackendEnum.VIDEO_SPARSE_ATTN, ))
            ref_layer.attn_impl = NativeImpl(12, d, False, d**-.5)
            funcs: dict[str, Callable[[], torch.Tensor]] = {
                'candidate':
                partial(distributed_call, layer, xs, m),
                'reference':
                partial(native_complete, xs, shape, sparsity) if world == 1 else partial(
                    distributed_call, ref_layer, xs, m),
                'dense':
                partial(dense_forward, xs, d),
            }
            if a.mode == 'sp4curve':
                dense_layer = DistributedAttention(
                    12, d, supported_attention_backends=(AttentionBackendEnum.VIDEO_SPARSE_ATTN, ))
                dense_layer.attn_impl = DenseImpl()
                funcs['dense'] = partial(distributed_dense_call, dense_layer, xs, m)
            before = jit_count()
            out, cold, event = timed(funcs['candidate'], True)
            after = jit_count()
            ref = funcs['reference']()
            torch.npu.synchronize()
            assert torch.allclose(out, ref, atol=.02, rtol=.02)
            operator_funcs: dict[str, Callable[[], torch.Tensor]] | None = None
            if a.mode == 'primary':
                _, mask = full_vsa(*xs, shape, sparsity)
                order, sizes, slots, inverse = oracle_metadata(shape, xs[0].device)

                packed = [pack_tensor(x, sizes.numel(), order, slots) for x in xs[:3]]
                counts = mask.sum(-1, dtype=torch.int32)
                maximum = int(counts.max().item())
                blocks = sizes.numel()
                ids = torch.where(mask, torch.arange(blocks, device='npu', dtype=torch.float32),
                                  float(blocks)).sort(-1).values[..., :maximum].clamp_max(blocks - 1).to(torch.int32)
                op_args = (*packed, ids, sizes.to(torch.int32), counts)
                operator_funcs = {
                    'reference': partial(native, *op_args),
                    'candidate': partial(sparse_forward, *op_args)
                }
                assert torch.allclose(operator_funcs['candidate'](), operator_funcs['reference'](), atol=.02, rtol=.02)
                free, _ = torch.npu.mem_get_info()
                assert 32 * 12 * blocks * 64 * d * 8 + sum(x.numel() * x.element_size() for x in packed) * 8 < free * .8
            if seed == 1024 and os.environ.get('VSA_PHASE2_BARRIER'):
                barrier = Path(os.environ['VSA_PHASE2_BARRIER'])
                barrier.mkdir(parents=True, exist_ok=True)
                (barrier / (os.environ['ASCEND_RT_VISIBLE_DEVICES'] + '.json')).write_text(
                    json.dumps({
                        'pid': os.getpid(),
                        'time': time.time(),
                        'jit_count': jit_count()
                    }))
                deadline = time.monotonic() + 900
                while len(list(barrier.glob('*.json'))) < int(os.environ['VSA_PHASE2_BARRIER_COUNT']):
                    if time.monotonic() > deadline:
                        raise TimeoutError('precompile barrier')
                    time.sleep(.5)
            base = {
                'frames': frames,
                'head_dim': d,
                'sparsity': sparsity,
                'seed': seed,
                'input_hash': digest,
                'rank': rank,
                'world_size': world,
                'physical_device': os.environ['ASCEND_RT_VISIBLE_DEVICES'],
                'first_call_wall_ms': cold,
                'first_call_event_ms': event,
                'jit_cold_counts': [before, after]
            }
            repeats = 4 if a.mode == 'sp4' else 1
            for repeat in range(repeats):
                orders = [['reference', 'candidate'], ['reference', 'candidate'], ['candidate', 'reference'],
                          ['candidate', 'reference']]
                random.Random(seed).shuffle(orders)
                index = repeat if world > 1 else (
                    [0, 4, 5, 6].index(int(os.environ['ASCEND_RT_VISIBLE_DEVICES'])) if a.mode == 'primary' else 0)
                order = orders[index]
                r = dict(base, repetition=repeat, order=order, paired={}, correctness=True)
                for role in order:
                    torch.npu.reset_peak_memory_stats()
                    r['paired'][role] = measure(funcs[role])
                    r[role + '_peak_bytes'] = torch.npu.max_memory_allocated()
                if world == 1 or a.mode == 'sp4curve':
                    r['dense'] = measure(funcs['dense'])
                if operator_funcs is not None:
                    r['operator_paired'] = {}
                    for role in order:
                        torch.npu.reset_peak_memory_stats()
                        r['operator_paired'][role] = measure(operator_funcs[role])
                        r['operator_' + role + '_peak_bytes'] = torch.npu.max_memory_allocated()
                r['status'] = 'PASS' if all(
                    v['stable'] and v['jit_count_before_after'][0] == v['jit_count_before_after'][1]
                    for v in r['paired'].values()) else 'INCONCLUSIVE'
                if operator_funcs is not None and not all(
                        v['stable'] and v['jit_count_before_after'][0] == v['jit_count_before_after'][1]
                        for v in r['operator_paired'].values()):
                    r['status'] = 'INCONCLUSIVE'
                records.append(r)
                Path(a.output).write_text(json.dumps(records, indent=2) + '\n')
                print(rank, frames, d, sparsity, seed, repeat, r['status'], flush=True)
finally:
    destroy_model_parallel()
    destroy_distributed_environment()
