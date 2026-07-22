"""T0: cards are pure data with real teeth (no torch, no weights)."""
import json

import pytest

from fastvideo2.card import (CardError, ComponentSpec, LoopSpec, ModelCard, Provenance,
                             SamplingDefaults, derive)
from fastvideo2.wan21.card import WAN21_T2V_1_3B


def _fake_card(**over):
    base = dict(
        model_id="fake-model", family="fake", weights="acme/fake",
        components={"enc": ComponentSpec("enc", kind="text_encoder",
                                         module="builtins:object", subfolder="enc")},
        loops={"main": LoopSpec("main", loop="fastvideo2.tests.test_loop:FakeLoop",
                                params={"n": 2})},
        capabilities=("text_to_video",),
        provenance=Provenance(assumes_loop="fake.loop/v1"),
        sampling_defaults=SamplingDefaults(num_steps=2, guidance_scale=1.0, height=64,
                                           width=64, num_frames=5, fps=8, shift=1.0),
    )
    base.update(over)
    return ModelCard(**base)


def test_wan_card_json_round_trip_preserves_digest():
    card = WAN21_T2V_1_3B
    rt = ModelCard.from_dict(json.loads(card.to_json()))
    assert rt == card
    assert rt.digest() == card.digest()


def test_derive_merges_loop_params_and_leaves_base_untouched():
    base = _fake_card().validate()
    variant = derive(base, model_id="fake-variant",
                     loops={"main": {"params": {"n": 7}}})
    assert variant.loops["main"].params == {"n": 7}
    assert variant.loops["main"].loop == base.loops["main"].loop  # merged, not replaced
    assert base.loops["main"].params == {"n": 2}                  # base untouched
    assert variant.digest() != base.digest()


def test_derive_rejects_unknown_fields():
    with pytest.raises(CardError, match="unknown card fields"):
        derive(_fake_card(), flux_capacitor=1)


def test_assumes_loop_semantics_teeth():
    # weights that assume a sampler the card does not declare must not validate
    with pytest.raises(CardError, match="assumes_loop"):
        derive(_fake_card().validate(), provenance={"assumes_loop": "bogus.sampler/v0"})


def test_unresolvable_loop_ref_fails_validation():
    bad = _fake_card(loops={"main": LoopSpec("main", loop="no.such.module:Loop")})
    with pytest.raises(CardError, match="cannot resolve"):
        bad.validate()


def test_non_json_loop_params_fail_validation():
    bad = _fake_card(loops={"main": LoopSpec("main",
                                             loop="fastvideo2.tests.test_loop:FakeLoop",
                                             params={"fn": object()})})
    with pytest.raises(CardError, match="plain JSON"):
        bad.validate()


def test_wan_card_declares_the_semantics_its_loop_carries():
    from fastvideo2.card import resolve_ref
    spec = WAN21_T2V_1_3B.loops["denoise"]
    cls = resolve_ref(spec.loop)
    assert cls.semantics == WAN21_T2V_1_3B.provenance.assumes_loop
