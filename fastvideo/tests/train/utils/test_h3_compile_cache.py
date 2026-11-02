"""Real CPU Dynamo + AOTAutograd cache exhaustion with nonreentrant checkpoints."""
from copy import deepcopy

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from fastvideo.train.utils.compile_cache import CheckpointCompileCache, canonicalize_singleton_batch


REPLAY_TAG = 'aa'


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.5, 1.5]))

    def forward(self, x, tag):
        # A different equal-length tag forces a new replay guard without changing
        # the mathematical graph or its saved tensors.
        return (x * self.weight).sin() * len(REPLAY_TAG) * len(tag)


@pytest.mark.parametrize('exhaust', ['per_call', 'accumulated'])
@pytest.mark.parametrize('compiler_backend', ['aot_eager', 'inductor'])
def test_exhaustion_with_outstanding_checkpoints_and_new_replay_guard(exhaust, compiler_backend):
    global REPLAY_TAG
    torch._dynamo.reset()
    REPLAY_TAG = 'aa'
    original_limits = torch._dynamo.config.recompile_limit, torch._dynamo.config.accumulated_recompile_limit
    cache = CheckpointCompileCache()
    block = Block()
    reference = deepcopy(block)
    parameters = list(block.parameters())
    wrapped = checkpoint_wrapper(block, preserve_rng_state=False)
    original_checkpoint = wrapped.checkpoint_fn
    cache.install_checkpoints(wrapped)
    backend = CompileCounterWithBackend(compiler_backend)
    kwargs = dict(fullgraph=True, dynamic=True, backend=backend,
                  recompile_limit=1 if exhaust == 'per_call' else 128)
    block.forward = torch.compile(block.forward, **kwargs)
    compiled = block.forward
    assert cache.install_dense(wrapped, kwargs, 'critic') == 1
    try:
        with torch._dynamo.config.patch(accumulated_recompile_limit=1 if exhaust == 'accumulated' else 256):
            pending = []
            for tag in ['x', 'yy', 'zzz', 'x']:
                x = torch.randn(1, 3, 2, requires_grad=True)
                ref_x = x.detach().clone().requires_grad_()
                pending.append((wrapped(x, tag), reference(ref_x, tag), x, ref_x))
            assert cache.calls['eager_fallbacks'] == 2
            REPLAY_TAG = 'bb'  # First seen in recomputation, with the cache already full.
            for actual, expected, x, ref_x in reversed(pending):
                actual.sum().backward(retain_graph=True)
                expected.sum().backward(retain_graph=True)
                # A checkpoint's context must be reusable, not a spent generator.
                actual.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(x.grad, ref_x.grad, atol=0, rtol=0)
            torch.testing.assert_close(block.weight.grad, reference.weight.grad, atol=0, rtol=0)
            assert cache.calls['eager_replays'] == 4 and cache.calls['compiled_replays'] == 4
            assert backend.frame_count == 2  # One forward graph + one emergency replay graph.
            assert cache.active.get() is None
            REPLAY_TAG = 'aa'
            # Limit exhaustion never flushes compiled graphs that still match.
            before = cache.calls['eager_fallbacks']
            wrapped(torch.ones(1, 3, 2, requires_grad=True), 'x').sum().backward()
            assert cache.calls['eager_fallbacks'] == before and backend.frame_count == 2
            # Dense no-grad execution must also survive exhaustion.
            with torch.no_grad():
                x = torch.randn(1, 3, 2)
                torch.testing.assert_close(wrapped(x, 'teacher'), reference(x, 'teacher'), atol=0, rtol=0)
        assert list(block.parameters()) == parameters
    finally:
        cache.close()
        REPLAY_TAG = 'aa'
        torch._dynamo.reset()
    assert wrapped.checkpoint_fn is original_checkpoint and block.forward is compiled
    assert original_limits == (torch._dynamo.config.recompile_limit, torch._dynamo.config.accumulated_recompile_limit)


def test_unrelated_compiler_error_propagates(monkeypatch):
    def bad(*args, **kwargs):
        raise RuntimeError('compiler bug, not a cache limit')
    monkeypatch.setattr(torch, 'compile', lambda *a, **kw: bad)
    cache = CheckpointCompileCache()
    eager = lambda *a: pytest.fail('unexpected eager retry')
    guarded = cache.wrap(eager, bad, {}, 'student')
    with pytest.raises(RuntimeError, match='compiler bug'):
        guarded(torch.ones(1, 2))
    assert cache.calls['eager_fallbacks'] == 0


@pytest.mark.parametrize('step', [1, 2])
def test_singleton_batch_stride_is_normalized_without_copy_or_gradient_change(step):
    base = torch.randn(1, 16, 8, requires_grad=True)
    view = base[:, :7:step]
    canonical = canonicalize_singleton_batch(view)
    assert canonical.stride(0) == canonical.shape[1] * canonical.stride(1)
    assert canonical.stride()[1:] == view.stride()[1:]
    assert canonical.data_ptr() == view.data_ptr()
    actual = torch.autograd.grad(canonical.sin().sum(), base, retain_graph=True)[0]
    expected = torch.autograd.grad(view.sin().sum(), base)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
