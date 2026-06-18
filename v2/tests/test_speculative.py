"""Speculative (draft-verify) decoding — exact + lower-latency AR (design_v3 §2.2; §9.16).

A cheap draft model proposes K tokens, the target verifies them, and the loop accepts the matching
prefix + one target correction — a variable accepted-length per round (a ragged AR loop). The two
load-bearing claims:

  * **Exactness:** the emitted sequence equals the target's OWN greedy decode, for ANY draft quality —
    the speedup is free (same output, fewer sequential target steps).
  * **Speedup scales with the accept rate:** a better draft ⇒ more tokens accepted per verify round ⇒
    fewer rounds (the expensive model's latency steps) for the same output.
"""
from __future__ import annotations

from v2.cache import CacheManager
from v2.card import load_card
from v2.platform.backends.toy import ToyTokenizer, _spec_target_next
from v2.recipes.speculative import build_speculative_card, build_speculative_program
from v2.parity import assert_interleave_parity
from v2.request import SamplingParams, TaskType, make_request
from v2.runtime import Engine

PROMPT = "hello world"
MAXTOK = 12


def _engine(*, agree=0.7, spec_len=4):
    card = build_speculative_card(spec_len=spec_len, draft_agree=agree)
    inst = load_card(card, cache_manager=CacheManager.from_card(card))
    eng = Engine()
    eng.register(card.model_id, inst, build_speculative_program())
    return eng


def _spec_tokens(eng, prompt=PROMPT, seed=1):
    out = eng.run(make_request(TaskType.REASON, "spec-decode", prompt,
                               sampling=SamplingParams(max_tokens=MAXTOK, seed=seed)))
    return list(out.artifacts["tokens"].tensor), out.metrics


def _target_greedy(prompt=PROMPT, n=MAXTOK):
    ctx, out = ToyTokenizer().encode(prompt), []
    for _ in range(n):
        nx = _spec_target_next(ctx)
        out.append(nx)
        ctx = ctx + [nx]
    return out


def test_speculative_output_equals_target_greedy_for_any_draft():
    """Exactness: regardless of draft quality, the spec sequence == the target's standalone greedy."""
    greedy = _target_greedy()
    for agree in (0.3, 0.7, 1.0):
        toks, _ = _spec_tokens(_engine(agree=agree))
        assert toks == greedy, f"draft agree={agree} broke exactness"


def test_speedup_scales_with_accept_rate():
    """A better draft accepts more per round ⇒ fewer verify rounds (latency steps) for the SAME output."""
    _, m_lo = _spec_tokens(_engine(agree=0.3))
    _, m_hi = _spec_tokens(_engine(agree=1.0))
    assert m_hi["tokens_per_round"] > m_lo["tokens_per_round"]
    assert m_hi["verify_rounds"] < m_lo["verify_rounds"]
    assert m_hi["tokens"] == m_lo["tokens"]              # same number of tokens emitted (same output)


def test_perfect_draft_reaches_k_tokens_per_round():
    _, m = _spec_tokens(_engine(agree=1.0, spec_len=4))
    assert m["tokens_per_round"] >= 4.0                 # all K accepted + the free token each round


def test_two_models_co_scheduled_one_instance():
    eng = _engine()
    inst = eng._registry["spec-decode"][0]
    assert inst.component("draft") is not inst.component("target")    # draft + target, one resident card
    _, m = _spec_tokens(eng)
    assert m["draft_forwards"] > 0 and m["target_forwards"] > 0


def test_speculative_interleave_parity():
    eng = _engine()
    reqs = [make_request(TaskType.REASON, "spec-decode", p, sampling=SamplingParams(max_tokens=MAXTOK, seed=s))
            for p, s in [("alpha", 1), ("beta", 2), ("alpha", 1)]]
    assert not assert_interleave_parity(eng, reqs)
