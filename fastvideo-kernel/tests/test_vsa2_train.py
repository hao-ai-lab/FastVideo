"""The VSA2 training kernel (fastvideo-kernel/vsa2_train) through its autograd wrapper, against the fp32 reference.

Needs an sm_100a or sm_103a GPU (B200 / GB200, B300); the first test builds the extension (about 20 s). The kernel's
own checks are vsa2_train/tools/bench.py (the five workloads) and vsa2_train/tools/edge.py (any other input); these
tests cover the wrapper: the outputs and gradients it returns, its layout, and that it calls the same kernels as `run`.
"""

import math
import os
import sys

import pytest
import torch

PACKAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vsa2_train")
sys.path.insert(0, PACKAGE)
sys.path.insert(0, os.path.join(PACKAGE, "tools"))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the VSA2 training kernel is built for sm_100a or sm_103a (B200 / GB200, B300)",
)


def _step(inputs):
    """The wrapper's forward and backward on run()'s inputs: out, selection, dq, dk, dv, dgate."""
    from vsa2_train import vsa2_attention

    q, k, v, gate, dout, t, h, w, text, topk = inputs
    leaves = [x.detach().clone().requires_grad_() for x in (q, k, v, gate)]
    out, selection = vsa2_attention(*leaves, t, h, w, text, topk)
    out.backward(dout)
    return (out.detach(), selection, *(x.grad for x in leaves))


def _passes(x, ref, atol, rtol):
    """Every judged (non-NaN) element within atol + rtol |ref|, and every element finite."""
    judged = ~torch.isnan(ref)
    x, ref = x.float(), ref.float()
    return bool(torch.isfinite(x).all()) and bool(((x - ref).abs() <= atol + rtol * ref.abs())[judged].all())


def test_s8_against_reference():
    import reference
    import spec

    inputs = spec.make_inputs("s8", 0, "cuda")
    out, selection, dq, dk, dv, dgate = _step(inputs)
    ref = reference.run(*inputs, outputs=(out, selection))
    ref_out, ref_sel, _, ref_dq, ref_dk, ref_dv, ref_dgate = ref

    tol = spec.WORKLOADS["s8"]
    atol, rtol = tol["atol"], tol["rtol"]
    judged = ~torch.isnan(ref_sel)
    assert bool((selection[judged] == ref_sel[judged]).all()), "selection differs away from the threshold"
    assert _passes(out, ref_out, atol, rtol), "out"
    assert _passes(dgate, ref_dgate, atol, rtol), "dgate"
    for name, x, r in (("dq", dq, ref_dq), ("dk", dk, ref_dk), ("dv", dv, ref_dv)):
        assert _passes(x, r, tol["outputs"][name]["atol"], rtol), name


@pytest.mark.parametrize("thw,text,topk,heads,batch", [((9, 10, 21), 130, 3, 2, 2), ((5, 8, 16), 0, 0, 3, 1)])
def test_wrapper_runs_the_kernels_of_run(thw, text, topk, heads, batch):
    """On a layout off the workloads (an odd latent, a partial text block or none, topk 0), the wrapper gives what
    `run` gives, writes zeros on the padding rows and nothing that is not finite."""
    import edge
    import reference
    from vsa2_train import _module

    lay = reference.Layout(text, thw)
    inputs = edge.make_inputs(lay, heads, batch, topk, 1.0, 1, "cuda")
    out, selection, dq, dk, dv, dgate = _step(inputs)

    q = inputs[0]
    s, nv = q.shape[2], lay.video_blocks
    expect = [torch.empty_like(q), torch.empty(batch, heads, nv, nv, device=q.device),
              torch.empty(batch, heads, s, device=q.device)] + [torch.empty_like(q) for _ in range(4)]
    _module().run(*inputs, *expect)
    run_out, run_sel, _, run_dq, run_dk, run_dv, run_dgate = expect

    assert torch.equal(selection, run_sel)
    assert torch.equal(out, run_out)
    # dQ is reduced across key blocks in no fixed order: equal up to the last bf16 bit.
    for x, r in ((dq, run_dq), (dk, run_dk), (dv, run_dv), (dgate, run_dgate)):
        assert bool(torch.isfinite(x).all())
        assert float((x.float() - r.float()).abs().max()) <= 1e-2

    padding = ~reference.row_valid(lay, q.device).reshape(-1)
    for x in (out, dq, dk, dv, dgate):
        assert float(x[:, :, padding].abs().max()) == 0.0
    assert selection.shape == (batch, heads, nv, nv) and math.isclose(float(selection.max()), 1.0)
