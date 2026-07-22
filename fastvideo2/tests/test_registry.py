"""T0: the identity policy — model_id is the only load key."""
import pytest

from fastvideo2.registry import CARDS, PIPELINES, resolve
from fastvideo2.wan21 import WAN21_T2V_1_3B


def test_resolve_by_model_id():
    card, build = resolve("wan2.1-t2v-1.3b")
    assert card is WAN21_T2V_1_3B and callable(build)


def test_hf_repo_strings_do_not_resolve():
    for repo in ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "Wan-AI/Wan2.1-T2V-1.3B"):
        with pytest.raises(KeyError, match="not load keys"):
            resolve(repo)


def test_unknown_id_lists_catalog():
    with pytest.raises(KeyError, match="known ids"):
        resolve("no-such-model")


def test_every_card_id_has_a_pipeline():
    assert set(CARDS) == set(PIPELINES)
