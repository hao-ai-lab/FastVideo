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
