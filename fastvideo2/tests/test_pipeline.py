"""T0: pipeline edges are enforced, not decorative (no torch, no weights)."""
import pytest

from fastvideo2.pipeline import (ComponentStage, LoopStage, Pipeline, PipelineError,
                                 run_component_stage)


def _noop(instance, inputs, request):
    return {}


def test_read_before_produce_rejected():
    p = Pipeline("p", inputs=("prompt",), stages=(
        ComponentStage("a", fn=_noop, reads=("not_yet",), writes=("x",)),),
        outputs={})
    with pytest.raises(PipelineError, match="reads 'not_yet'"):
        p.validate()


def test_slot_rewrite_rejected():
    p = Pipeline("p", inputs=("prompt",), stages=(
        ComponentStage("a", fn=_noop, reads=(), writes=("x",)),
        ComponentStage("b", fn=_noop, reads=(), writes=("x",)),),
        outputs={})
    with pytest.raises(PipelineError, match="rewrites slot 'x'"):
        p.validate()


def test_output_must_be_produced():
    p = Pipeline("p", inputs=(), stages=(), outputs={"video": "video"})
    with pytest.raises(PipelineError, match="output 'video'"):
        p.validate()


def test_loop_stage_single_write():
    p = Pipeline("p", inputs=(), stages=(
        LoopStage("l", loop_id="main", reads=(), writes=("a", "b")),),
        outputs={})
    with pytest.raises(PipelineError, match="exactly one write"):
        p.validate()


def test_stage_sees_only_declared_reads():
    def peeker(instance, inputs, request):
        with pytest.raises(KeyError):
            _ = inputs["secret"]
        return {"out": inputs["allowed"]}

    slots = {"allowed": 1, "secret": 2}
    st = ComponentStage("s", fn=peeker, reads=("allowed",), writes=("out",))
    run_component_stage(st, None, slots, None)
    assert slots["out"] == 1


def test_undeclared_write_rejected():
    st = ComponentStage("s", fn=lambda i, s, r: {"out": 1, "sneaky": 2},
                        reads=(), writes=("out",))
    with pytest.raises(PipelineError, match="declared writes"):
        run_component_stage(st, None, {}, None)


def test_missing_write_rejected():
    st = ComponentStage("s", fn=lambda i, s, r: {}, reads=(), writes=("out",))
    with pytest.raises(PipelineError, match="declared writes"):
        run_component_stage(st, None, {}, None)


def test_wan_pipeline_validates():
    from fastvideo2.wan21 import build_wan_t2v_pipeline
    p = build_wan_t2v_pipeline()
    assert [s.stage_id for s in p.stages] == ["text_encode", "denoise", "vae_decode"]
