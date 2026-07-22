"""T0: golden probe determinism, IO round-trip, and comparison math (numpy only)."""
import numpy as np

from fastvideo2.wan21 import goldens as G
from fastvideo2.wan21.anchor import report_markdown


def test_probe_inputs_are_deterministic():
    a, b = G.dit_probe_latent(), G.dit_probe_latent()
    assert a.shape == G.DIT_LATENT_SHAPE and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert np.array_equal(G.vae_probe_latent(), G.vae_probe_latent())


def test_pad_context():
    emb = np.ones((3, 8), dtype=np.float32)
    padded = G.pad_context(emb, text_len=6)
    assert padded.shape == (6, 8)
    assert np.array_equal(padded[:3], emb) and not padded[3:].any()


def test_golden_io_round_trip(tmp_path):
    d = str(tmp_path)
    arr = {"x": np.arange(6, dtype=np.float32).reshape(2, 3), "n": np.int64(2)}
    G.save_golden(d, "probe", arr, {"note": "t"})
    G.save_golden(d, "other", {"y": np.zeros(1)}, {"note": "u"})
    back = G.load_golden(d, "probe")
    assert np.array_equal(back["x"], arr["x"]) and int(back["n"]) == 2
    m = G.load_manifest(d)
    assert m["probe"]["note"] == "t" and m["other"]["note"] == "u"  # merge, not overwrite


def test_compare_pass_fail_and_shape():
    g = np.ones((4, 4))
    assert G.compare("a", g + 1e-6, g, tol_rel=1e-3)["status"] == "pass"
    assert G.compare("a", g + 1.0, g, tol_rel=1e-3)["status"] == "fail"
    assert G.compare("a", np.ones((2, 2)), g, tol_rel=1e-3)["status"] == "fail"
    assert G.rel_l2(g, g) == 0.0


def test_report_markdown_marks_failures():
    recs = {"fastvideo2": [{"name": "dit.t750", "status": "pass", "rel_l2": 1e-3}],
            "fastvideo_main": [{"name": "dit.t750", "status": "fail", "rel_l2": 0.5}]}
    md = report_markdown(recs, {"capture": {"repo": "r", "commit": "c"}})
    assert "| dit.t750 |" in md and "**FAIL**" in md and "fastvideo_main" in md
