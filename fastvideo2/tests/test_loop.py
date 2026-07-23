"""T0: the driven-loop contract, with a fake loop (no torch, no weights)."""
from fastvideo2.loop import Done, LoopRunner, LoopState, WorkPlan


class FakeLoop:
    """Minimal conforming loop: n steps, each step doubles a counter."""
    semantics = "fake.loop/v1"

    def __init__(self, *, loop_id: str, n: int = 3):
        self.loop_id = loop_id
        self.n = n

    def init(self, request, instance, inputs):
        st = LoopState(loop_id=self.loop_id, request_id=getattr(request, "request_id", "r"))
        st.latents = 1
        st.scratch["executed"] = 0
        st.scratch["inputs"] = dict(inputs)
        return st

    def next(self, st):
        if st.step_idx >= self.n:
            return Done()

        def run():
            st.scratch["executed"] += 1
            return {"latents": st.latents * 2}

        return WorkPlan(label=f"{self.loop_id}.{st.step_idx}", step=st.step_idx, run=run)

    def advance(self, st, result):
        st.latents = result["latents"]
        st.step_idx += 1
        return st

    def finalize(self, st):
        return {"latents": st.latents, "executed": st.scratch["executed"],
                "inputs": st.scratch["inputs"]}


class _Req:
    request_id = "req-loop"


def test_next_is_kernel_free():
    loop = FakeLoop(loop_id="fake")
    st = loop.init(_Req(), None, {})
    loop.next(st)
    loop.next(st)  # planning twice runs nothing
    assert st.scratch["executed"] == 0


def test_runner_drives_to_completion_and_observes():
    seen = []
    loop = FakeLoop(loop_id="fake", n=4)
    runner = LoopRunner(loop, _Req(), None, {"x": 9},
                        observe=lambda label, sec, meta: seen.append(label))
    out = runner.run()
    assert out["latents"] == 16 and out["executed"] == 4
    assert out["inputs"] == {"x": 9}
    assert seen == [f"fake.{i}" for i in range(4)]
    assert runner.done


def test_build_loop_from_spec():
    from fastvideo2.card import LoopSpec
    from fastvideo2.loop import build_loop
    loop = build_loop(LoopSpec("fake", loop="fastvideo2.tests.test_loop:FakeLoop",
                               params={"n": 2}))
    assert isinstance(loop, FakeLoop) and loop.n == 2 and loop.loop_id == "fake"


def test_dmd_sigma_table_lookup_matches_timesteps():
    # fastvideo's scheduler indexes warped sigmas by their own x1000 values,
    # so looking up t=757 returns sigma ~= 0.757 (nearest table entry),
    # nearly independent of shift.
    from fastvideo2.wan21.loop import dmd_sigma_for, dmd_sigma_table
    for shift in (3.0, 8.0):
        table = dmd_sigma_table(shift)
        assert len(table) == 1000
        for t in (1000.0, 757.0, 522.0):
            sigma = dmd_sigma_for(t, table)
            assert abs(sigma - t / 1000.0) < 2e-3, (shift, t, sigma)
    # endpoints: sigma_max == 1.0 at t=1000
    assert dmd_sigma_for(1000.0, dmd_sigma_table(8.0)) == 1.0


def test_dmd_loop_semantics_id_is_distinct():
    from fastvideo2.wan21.loop import WanDenoiseLoop, WanDMDLoop
    assert WanDMDLoop.semantics == "wan.dmd.x0renoise/v1"
    assert WanDMDLoop.semantics != WanDenoiseLoop.semantics  # the card teeth
