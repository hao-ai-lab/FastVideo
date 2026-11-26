# SPDX-License-Identifier: Apache-2.0
"""The MotionKernel ``mk`` tier is opt-in and never changes the default fusion set."""
import pytest

from fastvideo.models.dits.minimax_h3 import (_MINIMAX_H3_FUSION_NAMES, _MINIMAX_H3_MK_MODE,
                                              _enabled_minimax_h3_fusions)


def test_default_is_unchanged():
    for value in ("", "0", "none", None):
        assert _enabled_minimax_h3_fusions(value) == frozenset()


def test_all_is_unchanged():
    assert _enabled_minimax_h3_fusions("all") == _MINIMAX_H3_FUSION_NAMES
    assert _enabled_minimax_h3_fusions("1") == _MINIMAX_H3_FUSION_NAMES


def test_subset_is_unchanged():
    assert _enabled_minimax_h3_fusions("modulate,swiglu") == frozenset({"modulate", "swiglu"})


def test_mk_enables_every_fusion_plus_the_tier_marker():
    assert _enabled_minimax_h3_fusions("mk") == _MINIMAX_H3_FUSION_NAMES | {_MINIMAX_H3_MK_MODE}
    assert _enabled_minimax_h3_fusions(" MK ") == _MINIMAX_H3_FUSION_NAMES | {_MINIMAX_H3_MK_MODE}


def test_mk_cannot_be_combined_with_individual_fusions():
    with pytest.raises(ValueError, match="whole-tier mode"):
        _enabled_minimax_h3_fusions("mk,swiglu")


def test_unknown_fusion_still_raises():
    with pytest.raises(ValueError, match="Unknown MiniMax H3 fusion"):
        _enabled_minimax_h3_fusions("modulate,nope")
