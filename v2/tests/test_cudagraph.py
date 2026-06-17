"""Piecewise CUDA-graph capture/replay at the step boundary (Path A; design_v3 §6.2).

These tests exercise the capture LIFECYCLE that is correctness-critical regardless of device: the
capture key (capture-once / replay-many for compatible steps), eager-break for data-dependent steps
(SDE rollout, interceptor override), invalidation on shape / resident-weight changes, the
capture==eager contract, and the workspace-collision safety net. Replay re-runs the current step
thunk (CPU models the lifecycle, not the speedup) — see runtime/cudagraph.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from v2._enums import ExecutionProfile
from v2.cache import CacheManager
from v2.card import load_card
from v2.loop.driver import LoopRunner
from v2.models.backend import ToyDiT
from v2.models.common import text_encode_node_fn
from v2.models.wan21 import build_wan21_card
from v2.models.wan_causal import build_wan_causal_card
from v2.parity import bit_identical
from v2.platform import Platform
from v2.request import DiffusionParams, TaskType, make_request
from v2.request.streams import Stream
from v2.runtime import Engine
from v2.runtime.context import RuntimeLoopContext
from v2.runtime.cudagraph import GraphCapturer

_HUB = Engine()


def _wan(platform=None):
    card = build_wan21_card()
    return load_card(card, cache_manager=CacheManager.from_card(card), platform=platform)


def _drive(inst, *, seed=1, steps=8, sde=False, h=480, w=832, loop_id="diffusion_denoise"):
    req = make_request(TaskType.T2V, inst.card.model_id, "a fox",
                       diffusion=DiffusionParams(num_steps=steps, seed=seed, sde_rollout=sde,
                                                 height=h, width=w))
    slots: dict = {}
    text_encode_node_fn(inst, slots, req, None)
    profile = ExecutionProfile.ROLLOUT if sde else ExecutionProfile.SERVE
    ctx = RuntimeLoopContext(inst, observers=_HUB.observers, interceptors=_HUB.interceptors,
                             slots=slots, stream=Stream(req.request_id), cancel_scope=None,
                             profile=profile, metrics={}, request_id=req.request_id)
    runner = LoopRunner(inst.loop(loop_id), ctx, req, inst)
    while not runner.done:
        runner.step()
    return np.asarray(runner.result.outputs["latents"])


# --------------------------------------------------------------------------- #
# Capture lifecycle through a real driven loop                                #
# --------------------------------------------------------------------------- #
def test_capture_once_then_replay_for_constant_shape():
    inst = _wan()
    _drive(inst, steps=8)
    s = inst.graphs.stats
    assert s["captures"] == 1 and s["replays"] == 7 and s["eager_breaks"] == 0   # 8 steps, 1 shape


def test_sde_rollout_steps_all_eager_break():
    inst = _wan()
    _drive(inst, steps=6, sde=True)                  # stochastic step ⇒ capturable=False
    s = inst.graphs.stats
    assert s["eager_breaks"] == 6 and s["captures"] == 0 and s["replays"] == 0


def test_capture_replay_equals_pure_eager_bit_identical():
    on = _wan()
    a = _drive(on, seed=9, steps=8)
    off = _wan()
    off.graphs = GraphCapturer(enabled=False)         # the no-cudagraph baseline
    b = _drive(off, seed=9, steps=8)
    assert bit_identical(a, b)                        # capture+replay path ≡ eager path
    assert on.graphs.stats["replays"] == 7 and off.graphs.stats["replays"] == 0


def test_shape_change_recaptures_on_same_instance():
    inst = _wan()
    _drive(inst, seed=1, steps=4, h=480, w=832)
    _drive(inst, seed=1, steps=4, h=720, w=1280)      # different latent shape ⇒ different key
    assert inst.graphs.stats["captures"] == 2          # one graph per distinct shape (cache shared)


def test_weight_version_bump_recaptures_and_evicts_stale():
    inst = _wan()
    _drive(inst, seed=1, steps=4)
    assert inst.graphs.stats["captures"] == 1 and inst.graphs.captured == 1
    inst.set_weights_version("v1", ["transformer"])    # RL weight sync: bumps version + evicts graph
    assert inst.graphs.captured == 0                    # stale graph dropped (no leak)
    _drive(inst, seed=1, steps=4)
    assert inst.graphs.stats["captures"] == 2           # recaptured under the new version...
    assert inst.graphs.captured == 1                    # ...and only the live graph remains


def test_eager_loop_never_engages_capture():
    inst = load_card(build_wan_causal_card(),
                     cache_manager=CacheManager.from_card(build_wan_causal_card()))
    loop_id = next(iter(inst.card.loops))
    if inst.card.loops[loop_id].graph_capture == "breakable_cudagraph":
        pytest.skip("wan_causal opted into capture; this test asserts the eager-gated path")
    _drive(inst, seed=1, loop_id=loop_id)
    assert inst.graphs is None                          # gating: capturer never created for eager loops


def test_accel_backend_also_captures():
    inst = _wan(Platform.accel("sm90"))
    _drive(inst, steps=5)
    assert inst.graphs.stats["captures"] == 1 and inst.graphs.stats["replays"] == 4


# --------------------------------------------------------------------------- #
# Capturer unit tests (key, eager-break, workspace-collision safety net)      #
# --------------------------------------------------------------------------- #
class _Sig:
    def __init__(self, bk): self._bk = bk
    @property
    def batch_key(self): return self._bk


class _Res:
    def __init__(self, ws): self.peak_activation_bytes = ws


class _Plan:
    def __init__(self, ws=64, gk=(), cap=True, bk=("step",), gin=None, gfn=None):
        self.loop_id = "L"
        self.shape_sig = _Sig(bk)
        self.resources = _Res(ws)
        self.graph_key = gk
        self.capturable = cap
        self.run = None
        self.graph_inputs = gin if gin is not None else {"x": np.zeros((2, 2), dtype="float32")}
        self.graph_fn = gfn if gfn is not None else (lambda model, w: {"ok": True})


class _Plat:
    device, arch = "accel", "sm90"


class _Card:
    loops: dict = {}


class _Inst:
    def __init__(self): self.platform = _Plat(); self.card = _Card()
    def version_of(self, _c): return "v0"


def test_key_distinguishes_op_structure():
    inst = _Inst()
    k1 = GraphCapturer.key(_Plan(gk=("cond",)), inst)
    k2 = GraphCapturer.key(_Plan(gk=("cond", "uncond")), inst)
    assert k1 != k2                                     # a different branch set ⇒ a different graph


def test_override_and_noncapturable_eager_break():
    cap, inst, seen = GraphCapturer(), _Inst(), {}
    eager = lambda ov: (seen.update(ov=ov), {"r": 1})[1]
    cap.dispatch(_Plan(), inst, {"noise_pred": 1}, eager)      # override ⇒ eager-break
    assert cap.stats["eager_breaks"] == 1 and seen["ov"] == {"noise_pred": 1}
    cap.dispatch(_Plan(cap=False), inst, None, eager)          # non-capturable ⇒ eager-break
    assert cap.stats["eager_breaks"] == 2 and cap.stats["captures"] == 0


def test_no_static_graph_form_eager_breaks():
    cap, inst = GraphCapturer(), _Inst()
    p = _Plan(); p.graph_fn = None                             # a plan with no static capture form
    cap.dispatch(p, inst, None, lambda ov: {"r": 1})
    assert cap.stats["eager_breaks"] == 1 and cap.stats["captures"] == 0


def test_invalidate_drops_only_matching_component_graphs():
    from v2.runtime.cudagraph import CapturedGraph
    cap = GraphCapturer()
    # two graphs whose keys carry different resident components (key[4] = resident ((cid, ver), ...))
    k_dit = ("accel", "sm90", "L", ("a",), (("transformer", "v0"),), ())
    k_txt = ("accel", "sm90", "L", ("a",), (("text_encoder", "v0"),), ())
    cap._graphs[k_dit] = CapturedGraph(k_dit, None, 1)
    cap._graphs[k_txt] = CapturedGraph(k_txt, None, 1)
    dropped = cap.invalidate({"transformer"})
    assert dropped == 1 and cap.captured == 1 and k_txt in cap._graphs   # only the dit graph evicted


def test_workspace_bytes_collision_raises():
    """Same key, different declared static-workspace size ⇒ the key is insufficient. Refuse."""
    cap, inst = GraphCapturer(), _Inst()
    eager = lambda ov: {"ok": True}
    cap.dispatch(_Plan(ws=100), inst, None, eager)             # capture: key→100B workspace
    with pytest.raises(RuntimeError, match="key collision"):
        cap.dispatch(_Plan(ws=200), inst, None, eager)         # same key, 200B ⇒ refuse


def test_static_buffer_shape_mismatch_raises():
    """The real backstop: rebinding an input whose shape doesn't fit the captured static buffer
    raises (np.copyto) — catching a key that let incompatible shapes share a graph."""
    cap, inst = GraphCapturer(), _Inst()
    eager = lambda ov: {}
    cap.dispatch(_Plan(gin={"x": np.zeros((2, 2), dtype="float32")}), inst, None, eager)  # 2x2 buffer
    with pytest.raises(ValueError):                            # 4x4 won't fit the captured 2x2 buffer
        cap.dispatch(_Plan(gin={"x": np.zeros((4, 4), dtype="float32")}), inst, None, eager)
