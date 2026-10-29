"""Cold compiler API or existing-disk-cache startup through the real SP1 layer."""
import time

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import triton.compiler as compiler
from fastvideo.attention.layer import DistributedAttention_VSA
from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionMetadataBuilder
from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, destroy_model_parallel, destroy_distributed_environment
from fastvideo.forward_context import set_forward_context
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.utils import set_mixed_precision_policy
from measure import benchmark, jit_count, timed

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--root', default='.')
p.add_argument('--reference-dir', type=Path)
a = p.parse_args()
root = Path(a.root).resolve()

torch.npu.set_device(0)
torch.set_num_threads(4)
set_mixed_precision_policy(torch.bfloat16, torch.float32)
init_distributed_environment(local_rank=0, timeout=timedelta(seconds=120))
initialize_model_parallel()
calls: list[dict[str, Any]] = []
original = compiler.compile


def compile_record(*args, **kwargs):
    start = time.perf_counter()
    try:
        return original(*args, **kwargs)
    finally:
        calls.append({
            'wall_ms': (time.perf_counter() - start) * 1000,
            'function':
            str(getattr(getattr(args[0], 'fn', None), '__name__', 'unknown')) if args else 'unknown'
        })


compiler.compile = compile_record
try:
    with torch.inference_mode():
        gen = torch.Generator().manual_seed(1024)
        cpu = [(torch.randn(1, 32760, 12, 128, generator=gen) * .3).to(torch.bfloat16) for _ in range(4)]
        digest = hashlib.sha256(b''.join(x.view(torch.uint16).numpy().tobytes() for x in cpu)).hexdigest()
        xs = [x.to('npu') for x in cpu]
        meta_start = time.perf_counter()
        m = VideoSparseAttentionMetadataBuilder().build(0, (21, 30, 52), (1, 1, 1), .8, torch.device('npu:0'))
        torch.npu.synchronize()
        metadata_wall = (time.perf_counter() - meta_start) * 1000
        layer_start = time.perf_counter()
        layer = DistributedAttention_VSA(12,
                                         128,
                                         supported_attention_backends=(AttentionBackendEnum.VIDEO_SPARSE_ATTN, ))
        layer_wall = (time.perf_counter() - layer_start) * 1000

        def call():
            with set_forward_context(0, m):
                return layer(*xs[:3], original_seq_len=32760, gate_compress=xs[3], attn_metadata=m)[0]

        before = jit_count()
        out, wall, event = timed(call, True)
        after = jit_count()
        reference_dir = a.reference_dir or root / 'inputs/references/phase1-exact'
        reference = torch.load(reference_dir / 'f81-d128-s80-bfloat16-seed1024.pt', weights_only=True)
        assert digest == reference['input_hash']
        correct = torch.allclose(out, reference['output'].to('npu'), atol=.02, rtol=.02)
        result = {
            'physical_device': os.environ['ASCEND_RT_VISIBLE_DEVICES'],
            'pid': os.getpid(),
            'input_hash': digest,
            'correctness': correct,
            'shape_metadata_build_wall_ms': metadata_wall,
            'backend_layer_init_wall_ms': layer_wall,
            'cache_scope': os.environ['TRITON_CACHE_DIR'],
            'jit_before_after': [before, after],
            'first_complete_wall_ms': wall,
            'first_complete_event_ms': event,
            'compiler_api_calls': list(calls),
            'compiler_api_wall_sum_ms': sum(r['wall_ms'] for r in calls),
            'warm': benchmark(call),
            'finite': torch.isfinite(out).all().item(),
            'peak_memory_bytes': torch.npu.max_memory_allocated()
        }
        result['status'] = 'PASS' if correct and result['finite'] and result['warm']['stable'] else 'INCONCLUSIVE'
        Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
        print(result['status'], wall)
finally:
    destroy_model_parallel()
    destroy_distributed_environment()
