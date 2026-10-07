# SPDX-License-Identifier: Apache-2.0
"""CUDA host-registration lifetime and offload regressions (one GPU required)."""

import gc
import weakref

import pytest
import torch

import fastvideo.envs as envs
from torch import nn

from fastvideo.hooks.hooks import ModuleHookManager
from fastvideo.hooks.layerwise_offload import LayerwiseOffloadHook, LayerwiseOffloadState
from fastvideo.hooks.pinned_memory import PinnedTensorArena

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host registration requires one GPU")


def test_arena_mixed_dtype_exact_copy_and_lifetime():
    sources = {
        "weight": torch.randn(17, 33, device="cuda", dtype=torch.bfloat16),
        "scale": torch.randn(17, 1, device="cuda", dtype=torch.float32),
        "packed": torch.arange(513, device="cuda").to(torch.uint8),
        "scalar": torch.tensor(3.0, device="cuda"),
        "empty": torch.empty(0, 4, device="cuda"),
    }
    arena = PinnedTensorArena(sources.items())
    assert arena.buffer is not None, "This GPU must exercise registration, not fallback"
    assert arena.buffer.numel() < sum(t.numel() * t.element_size() for t in sources.values()) + 4096 + 256 * len(sources)
    hosts = {}
    for name, source in sources.items():
        host = arena.empty_like(name, source)
        host.copy_(source)
        if host.numel():
            assert host.is_pinned()
        assert (host.data_ptr() - arena.buffer.data_ptr()) % 256 == 0 or host.numel() == 0
        torch.testing.assert_close(host.to("cuda", non_blocking=True), source, rtol=0, atol=0)
        hosts[name] = host
    owner = weakref.ref(arena)
    del arena
    gc.collect()
    assert owner() is not None, "Live typed views must retain the registration"
    del hosts, host
    gc.collect()
    assert owner() is None


def test_registration_failure_uses_pinned_allocator(monkeypatch):
    class RefusingRuntime:

        def cudaHostRegister(self, *_args):
            return 1

    monkeypatch.setattr(torch.cuda, "cudart", lambda: RefusingRuntime())
    source = torch.arange(27, dtype=torch.float32)
    arena = PinnedTensorArena([("weight", source)])
    assert arena.buffer is None
    host = arena.empty_like("weight", source)
    host.copy_(source)
    assert host.is_pinned()
    torch.testing.assert_close(host, source, rtol=0, atol=0)
    arena.close()
    arena.close()


def test_offload_mutation_and_prefetched_detach(env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_LAYERWISE_OFFLOAD_BUFFERS.override(True))
    module = nn.Linear(16, 16, device="cuda", dtype=torch.bfloat16)
    module.register_buffer("packed", torch.arange(1 << 20, device="cuda").to(torch.uint8))
    expected = {name: tensor.clone() for name, tensor in list(module.named_parameters()) + list(module.named_buffers())}
    state = LayerwiseOffloadState(torch.cuda.Stream(), torch.device("cuda"))
    hook = LayerwiseOffloadHook(state)
    manager = ModuleHookManager.get_from_or_default(module)
    manager.append_forward_hook(hook)
    old_buffer = state.cpu_arena.buffer
    assert old_buffer.is_pinned()
    with hook.mutate_params_scope(), torch.no_grad():
        module.weight.add_(1)
        module.packed.add_(1)
    assert not old_buffer.is_pinned(), "Reinitialization must unregister old storage"
    expected["weight"].add_(1)
    expected["packed"].add_(1)
    state.prefetch_params()
    new_buffer = state.cpu_arena.buffer
    manager.remove_forward_hook(hook.name())
    assert not new_buffer.is_pinned(), "Detachment must unregister storage"
    assert state.cpu_arena is None
    assert not state.cpu_named_parameters and not state.gpu_named_parameters
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


def test_h3_swap_reuses_host_storage_and_updates_buffers():
    from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import _pinned_swap

    module = nn.Linear(16, 16, device="cuda", dtype=torch.bfloat16)
    module.register_buffer("cache", torch.ones(11, device="cuda"))
    expected_weight = module.weight.detach().clone()
    _pinned_swap(module, torch.device("cpu"))
    hosts = module._pinned_host_tensors
    pointers = {name: tensor.data_ptr() for name, tensor in hosts.items()}
    assert all(tensor.is_pinned() for tensor in hosts.values())
    _pinned_swap(module, torch.device("cuda", torch.cuda.current_device()))
    module.cache.add_(3)
    _pinned_swap(module, torch.device("cpu"))
    assert pointers == {name: tensor.data_ptr() for name, tensor in hosts.items()}
    torch.testing.assert_close(module.cache, torch.full((11,), 4.0), rtol=0, atol=0)
    torch.testing.assert_close(module.weight, expected_weight.cpu(), rtol=0, atol=0)
