# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``LTX2VideoArchConfig.param_names_mapping``.

The ``to_gate_compress`` -> ``to_gate_logits`` rename is the LTX-2.3 gated
attention loader rule.  It must only fire when ``apply_gated_attention=True``;
otherwise it would silently retarget:

- LTX-2.0 ``VIDEO_SPARSE_ATTN`` checkpoints, whose attention modules
  legitimately carry a ``to_gate_compress`` VSA-QAT gate (a sibling of
  ``attn_masked`` in ``fastvideo/models/dits/ltx2.py``).
- LoRAs trained with the default ``lora_target_modules`` list (which
  includes ``to_gate_compress``; see ``fastvideo/train/utils/lora.py:36``
  and ``fastvideo/pipelines/lora_pipeline.py:171``).
"""

from __future__ import annotations

import pytest

from fastvideo.configs.models.dits.ltx2 import LTX2VideoArchConfig
from fastvideo.models.loader.utils import get_param_names_mapping


def _map_with_metadata(
    name: str,
    *,
    apply_gated_attention: bool = False,
    pack_attention_projections: bool = False,
) -> tuple[str, int | None, int | None]:
    """Run ``name`` through a fresh config's mapping function."""
    cfg = LTX2VideoArchConfig(
        apply_gated_attention=apply_gated_attention,
        pack_attention_projections=pack_attention_projections,
    )
    mapper = get_param_names_mapping(cfg.param_names_mapping)
    return mapper(name)


def _map(name: str, *, apply_gated_attention: bool) -> str:
    return _map_with_metadata(
        name,
        apply_gated_attention=apply_gated_attention,
    )[0]


class TestLTX20ParamMappingDefault:
    """``apply_gated_attention=False`` (LTX-2.0 default): ``to_gate_compress``
    must pass through unmodified except for prefix normalization."""

    @pytest.mark.parametrize("source", [
        "transformer_blocks.0.attn1.to_gate_compress.weight",
        "model.transformer_blocks.5.attn1.to_gate_compress.bias",
        "diffusion_model.transformer_blocks.10.attn1.to_gate_compress.weight",
        "model.diffusion_model.transformer_blocks.2.attn1.to_gate_compress.weight",
    ])
    def test_vsa_gate_weight_not_renamed(self, source):
        target = _map(source, apply_gated_attention=False)
        assert "to_gate_logits" not in target, (
            f"LTX-2.0 VSA gate {source!r} was rewritten to {target!r}; "
            "expected pass-through with only prefix normalization.")
        assert ".to_gate_compress." in target, (
            f"LTX-2.0 VSA gate {source!r} lost its identity in {target!r}.")

    @pytest.mark.parametrize("source", [
        "transformer_blocks.0.attn1.to_gate_compress.lora_A.weight",
        "transformer_blocks.0.attn1.to_gate_compress.lora_B.weight",
        "model.transformer_blocks.3.attn1.to_gate_compress.lora_A.weight",
    ])
    def test_default_lora_target_not_renamed(self, source):
        """``to_gate_compress`` is in ``DEFAULT_LORA_TARGET_MODULES``, so LoRAs
        trained with default targets ship these keys."""
        target = _map(source, apply_gated_attention=False)
        assert "to_gate_logits" not in target, (
            f"Default-target LoRA key {source!r} was rewritten to {target!r}; "
            "this would break every LTX-2.0 LoRA in the wild.")

    def test_unrelated_param_still_prefix_stripped(self):
        """Generic prefix-strip behavior is unchanged in LTX-2.0 mode."""
        assert (_map("diffusion_model.transformer_blocks.0.attn1.to_q.weight",
                     apply_gated_attention=False) ==
                "model.transformer_blocks.0.attn1.to_q.weight")


class TestLTX23ParamMappingGated:
    """``apply_gated_attention=True`` (LTX-2.3): the gated-attention
    ``to_gate_compress`` upstream key is renamed to ``to_gate_logits``."""

    @pytest.mark.parametrize("source,expected", [
        # No upstream prefix
        ("transformer_blocks.0.attn1.to_gate_compress.weight",
         "model.transformer_blocks.0.attn1.to_gate_logits.weight"),
        # ``model.`` prefix
        ("model.transformer_blocks.0.attn1.to_gate_compress.weight",
         "model.transformer_blocks.0.attn1.to_gate_logits.weight"),
        # ``diffusion_model.`` prefix
        ("diffusion_model.transformer_blocks.0.attn1.to_gate_compress.weight",
         "model.transformer_blocks.0.attn1.to_gate_logits.weight"),
        # ``model.diffusion_model.`` prefix
        ("model.diffusion_model.transformer_blocks.0.attn1.to_gate_compress.weight",
         "model.transformer_blocks.0.attn1.to_gate_logits.weight"),
    ])
    def test_gate_rename_fires(self, source, expected):
        assert _map(source, apply_gated_attention=True) == expected

    def test_non_gate_param_still_prefix_stripped(self):
        """Non-gate params still get the prefix-strip behavior."""
        assert (_map("diffusion_model.transformer_blocks.0.attn1.to_q.weight",
                     apply_gated_attention=True) ==
                "model.transformer_blocks.0.attn1.to_q.weight")


class TestPackedAttentionProjectionMapping:

    @pytest.mark.parametrize("prefix", [
        "",
        "model.",
        "diffusion_model.",
        "model.diffusion_model.",
    ])
    @pytest.mark.parametrize("suffix", ["weight", "bias"])
    @pytest.mark.parametrize(
        "attention,projection,packed,index,total",
        [
            ("attn1", "to_q", "to_qkv", 0, 3),
            ("attn1", "to_k", "to_qkv", 1, 3),
            ("attn1", "to_v", "to_qkv", 2, 3),
            ("attn2", "to_k", "to_kv", 0, 2),
            ("attn2", "to_v", "to_kv", 1, 2),
        ],
    )
    def test_opt_in_maps_split_checkpoint_projections_to_packed_parameters(
        self,
        prefix,
        suffix,
        attention,
        projection,
        packed,
        index,
        total,
    ):
        source = f"{prefix}transformer_blocks.7.{attention}.{projection}.{suffix}"

        assert _map_with_metadata(
            source,
            pack_attention_projections=True,
        ) == (
            f"model.transformer_blocks.7.{attention}.{packed}.{suffix}",
            index,
            total,
        )

    @pytest.mark.parametrize("suffix", ["weight", "bias"])
    @pytest.mark.parametrize(
        "attention,projection",
        [
            ("attn1", "to_q"),
            ("attn1", "to_k"),
            ("attn1", "to_v"),
            ("attn2", "to_k"),
            ("attn2", "to_v"),
        ],
    )
    def test_default_keeps_checkpoint_projections_split(
        self,
        suffix,
        attention,
        projection,
    ):
        source = f"transformer_blocks.7.{attention}.{projection}.{suffix}"

        assert _map_with_metadata(source) == (
            f"model.{source}",
            None,
            None,
        )

    @pytest.mark.parametrize("prefix", [
        "",
        "model.",
        "diffusion_model.",
        "model.diffusion_model.",
    ])
    def test_cross_attention_query_stays_separate(self, prefix):
        assert _map_with_metadata(
            f"{prefix}transformer_blocks.3.attn2.to_q.weight",
            pack_attention_projections=True,
        ) == (
            "model.transformer_blocks.3.attn2.to_q.weight",
            None,
            None,
        )

    def test_post_init_is_idempotent_and_removes_disabled_pack_rules(self):
        cfg = LTX2VideoArchConfig(pack_attention_projections=True)
        initial_mapping = tuple(cfg.param_names_mapping.items())
        packed_patterns = {
            pattern
            for pattern, replacement in initial_mapping
            if isinstance(replacement, tuple)
        }

        cfg.__post_init__()
        assert tuple(cfg.param_names_mapping.items()) == initial_mapping

        cfg.pack_attention_projections = False
        cfg.__post_init__()
        assert packed_patterns.isdisjoint(cfg.param_names_mapping)
        assert get_param_names_mapping(cfg.param_names_mapping)(
            "transformer_blocks.0.attn1.to_q.weight"
        ) == (
            "model.transformer_blocks.0.attn1.to_q.weight",
            None,
            None,
        )


class TestParamMappingRuleOrdering:
    """The gate rules must be inserted *before* the generic prefix-strip rules
    so first-match-wins matching fires the rename first."""

    def test_gate_rule_wins_over_prefix_strip_when_enabled(self):
        cfg = LTX2VideoArchConfig(apply_gated_attention=True)
        # The first key in the dict iteration order should be a gate rule
        # (not a generic prefix-strip rule), so that ``transformer_blocks
        # .0.attn1.to_gate_compress.weight`` is renamed rather than just
        # prefix-normalized to ``model.transformer_blocks.0.attn1
        # .to_gate_compress.weight``.
        first_pattern = next(iter(cfg.param_names_mapping))
        assert "to_gate_compress" in first_pattern, (
            "gate rename rule must be inserted at the front of "
            "param_names_mapping; got first pattern: "
            f"{first_pattern!r}")
