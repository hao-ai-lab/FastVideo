"""T0: the SDK facade over a fake card (no torch, no weights)."""
import numpy as np
import pytest

from fastvideo2.engine import Instance, Request
from fastvideo2.sdk import Model, Result
from fastvideo2.tests.test_engine import CARD, PIPE


def _model(tmp_path) -> Model:
    return Model(CARD, PIPE, Instance(CARD, root=str(tmp_path), device="cpu"))


def test_generate_resolves_defaults_and_returns_typed_result(tmp_path):
    m = _model(tmp_path)
    r = m.generate("hi", seed=3)
    assert isinstance(r, Result)
    assert r.outputs["final"] == 9                       # fake loop ran
    assert r.request.negative_prompt == "default-neg"    # card defaults resolved
    assert r.model_id == "fake-e2e" and r.card_digest == CARD.digest()
    assert r.video is None and r.latents is None         # fake card has no media


def test_prompt_xor_request_enforced(tmp_path):
    m = _model(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        m.generate()
    with pytest.raises(ValueError, match="exactly one"):
        m.generate("p", request=Request(prompt="p"))


def test_request_passthrough_and_capabilities(tmp_path):
    m = _model(tmp_path)
    r = m.generate(request=Request(prompt="abc", request_id="sdk1"))
    assert any(t["label"].startswith("sdk1/") for t in r.trace)
    assert m.capabilities == ("text_to_video",)
    assert m.describe()["model_id"] == "fake-e2e"


def test_result_save_dispatches_video(tmp_path):
    r = Result(outputs={"video": np.zeros((3, 8, 8, 3), dtype=np.uint8)}, trace=[],
               request=Request(prompt="p"), model_id="m", card_digest="d", fps=8)
    pytest.importorskip("imageio")
    out = r.save(str(tmp_path / "v.mp4"))
    assert out.endswith("v.mp4")
    with pytest.raises(ValueError, match="nothing saveable"):
        Result(outputs={}, trace=[], request=Request(prompt="p"),
               model_id="m", card_digest="d", fps=8).save("x.mp4")
