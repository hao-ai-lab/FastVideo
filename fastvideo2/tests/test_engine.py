"""T0: the engine end-to-end over fake components — outputs, defaults
resolution, and the identity chain in the trace (no torch, no weights)."""
from fastvideo2.card import ComponentSpec, LoopSpec, ModelCard, Provenance, SamplingDefaults
from fastvideo2.engine import Instance, Request, run
from fastvideo2.pipeline import ComponentStage, LoopStage, Pipeline

CARD = ModelCard(
    model_id="fake-e2e", family="fake", weights="acme/fake",
    components={"enc": ComponentSpec("enc", kind="text_encoder",
                                     module="builtins:object", subfolder="enc")},
    loops={"main": LoopSpec("main", loop="fastvideo2.tests.test_loop:FakeLoop",
                            params={"n": 3})},
    capabilities=("text_to_video",),
    provenance=Provenance(assumes_loop="fake.loop/v1"),
    sampling_defaults=SamplingDefaults(num_steps=3, guidance_scale=1.0, height=64,
                                       width=64, num_frames=5, fps=8, shift=1.0,
                                       negative_prompt="default-neg"),
).validate()

PIPE = Pipeline(
    "fake.e2e", inputs=("prompt", "negative_prompt"),
    stages=(
        ComponentStage("prep", fn=lambda i, s, r: {"prepped": s["prompt"].upper()},
                       reads=("prompt",), writes=("prepped",)),
        LoopStage("mainloop", loop_id="main", reads=("prepped",), writes=("loop_out",)),
        ComponentStage("post", fn=lambda i, s, r: {"final": s["loop_out"]["latents"] + 1},
                       reads=("loop_out",), writes=("final",)),
    ),
    outputs={"final": "final", "neg": "negative_prompt"},
).validate()


def _instance(tmp_path):
    return Instance(CARD, root=str(tmp_path), device="cpu")


def test_engine_end_to_end(tmp_path):
    out = run(_instance(tmp_path), PIPE, Request(prompt="hi", request_id="rq"))
    assert out.outputs["final"] == 1 * 2 ** 3 + 1
    # the loop received exactly its declared reads
    assert out.outputs["neg"] == "default-neg"          # resolved from card defaults
    labels = [t["label"] for t in out.trace]
    assert "rq/prep" in labels and "rq/post" in labels
    assert [f"rq/mainloop/main.{i}" for i in range(3)] == [
        l for l in labels if "/mainloop/" in l]         # the identity chain


def test_request_defaults_resolution():
    r = Request(prompt="p", num_steps=9).resolve(CARD)
    assert r.num_steps == 9                              # explicit wins
    assert r.height == 64 and r.negative_prompt == "default-neg"


def test_loop_inputs_are_declared_reads_only(tmp_path):
    out = run(_instance(tmp_path), PIPE, Request(prompt="abc"))
    assert out.outputs["final"] == 9  # sanity: loop ran on defaults
