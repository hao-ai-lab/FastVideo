"""T0: the identity policy — model_id primary, HF repos as unambiguous aliases."""
from types import SimpleNamespace

import pytest

from fastvideo2.registry import CARDS, _build_aliases, resolve
from fastvideo2.wan21 import WAN21_T2V_1_3B


def test_resolve_by_model_id():
    card, build = resolve("wan2.1-t2v-1.3b")
    assert card is WAN21_T2V_1_3B and callable(build)


def test_resolve_by_weights_repo_alias():
    card, _ = resolve("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    assert card.model_id == "wan2.1-t2v-1.3b"


def test_resolve_by_component_source_alias():
    # the official-layout DiT repo is an ingredient of the same card
    card, _ = resolve("Wan-AI/Wan2.1-T2V-1.3B")
    assert card.model_id == "wan2.1-t2v-1.3b"


def test_unknown_name_lists_ids_and_aliases():
    with pytest.raises(KeyError, match="known ids"):
        resolve("no/such-model")


def test_ambiguous_repo_alias_fails_closed():
    def fake(mid):
        spec = SimpleNamespace(source="")
        return SimpleNamespace(model_id=mid, weights="acme/shared-repo",
                               components={"c": spec})
    aliases = _build_aliases({"a": fake("card-a"), "b": fake("card-b")})
    assert aliases["acme/shared-repo"] == ["card-a", "card-b"]  # resolve() would error


def test_every_card_id_has_a_pipeline():
    from fastvideo2.registry import PIPELINES
    assert set(CARDS) == set(PIPELINES)
