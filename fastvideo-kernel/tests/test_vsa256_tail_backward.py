"""Opt-in tail packing preserves gradients across device-plan/Graph transitions."""
import pytest
import torch

from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter
from .test_vsa256_backward import _check, _GRAD_TOL


@pytest.mark.parametrize('dim', [64, 128])
def test_tail_backward_graph(monkeypatch, dim):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip('SM100 GPU required')
    import inspect
    pytest.importorskip("flash_attn.cute.interface")
    if "_workspace" not in inspect.signature(adapter._load_fa4_cute()[3]).parameters:
        pytest.skip("FA4 backward workspace support required")
    monkeypatch.setenv('FASTVIDEO_VSA_PACK_TAILS', '1')
    torch.manual_seed(451 + dim)
    q = torch.randn(2, 1024, 3, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k, v = [torch.randn(2, 1536, 3, dim, device='cuda', dtype=torch.bfloat16,
                       requires_grad=True) for _ in range(2)]
    dout = torch.randn_like(q)
    dlse = torch.randn(2, 3, 1024, device='cuda') * .1
    cases = ([256, 131, 128, 0, 132, 256], [127, 1, 128, 0, 256, 0],
             [67, 67, 128, 256, 0, 128], [256]*6, [0]*6, [256, 131, 67, 0, 255, 32])
    sizes = torch.tensor(cases[0], device='cuda', dtype=torch.int32)
    routes = torch.rand(2, 3, 4, 6, device='cuda') > .4
    routes[:, :, 0] = False

    def call(optimized, lse_only=False):
        fn = adapter._cute_attention if optimized else adapter._CuteAttentionQ256Training.apply
        out, lse = fn(q, k, v, routes, sizes)
        grad_lse = dlse.masked_fill(~torch.isfinite(lse), 0)
        grads = (torch.autograd.grad(lse, (q, k, v), grad_lse) if lse_only else
                 torch.autograd.grad((out, lse), (q, k, v), (dout, grad_lse)))
        return grads

    for lse_only in (False, True):
        for ref, got in zip(call(False, lse_only), call(True, lse_only)):
            _check('gradient', ref, got, _GRAD_TOL)
    # Fresh leaves avoid default-stream AccumulateGrad nodes during capture.
    q, k, v = (t.detach().requires_grad_(True) for t in (q, k, v))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call(True)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = call(True)
    for iteration in range(12):
        with torch.no_grad():
            sizes.copy_(torch.tensor(cases[iteration % len(cases)], device='cuda', dtype=torch.int32))
            for tensor in (q, k, v, dout, dlse):
                tensor.normal_()
            routes.copy_(torch.rand_like(routes, dtype=torch.float32) > .4)
            routes[:, :, 0] = False
            for tensor in actual:
                tensor.fill_(float('nan'))
        graph.replay()
        for ref, got in zip(call(False), actual):
            _check('replayed gradient', ref, got, _GRAD_TOL)
            if not any(cases[iteration % len(cases)]):
                assert torch.count_nonzero(got) == 0
