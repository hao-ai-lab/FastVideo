# SPDX-License-Identifier: Apache-2.0
"""FP8 suffix-matching contract for feed-forward stacks.

``FP8Config.get_quant_method`` decides what gets quantized by matching a
layer's state-dict name against ``_FP8_SUFFIXES`` and ``_FP8_EXACT_SUFFIXES``.
The first list carried ``ffn.fc_in``/``ffn.fc_out``, which fits WanVideo and
LTX-2, but MiniMax-H3 mounts its feed-forward on ``.ff``.
``"ffn.fc_in" in "transformer_blocks.0.ff.fc_in"`` is False, so every H3 FFN
kept its bf16 weights while quantization reported success: 8.01B of 22.09B
parameters were converted and nothing said so.

These are CPU-only contracts. Attaching ``quant_method`` needs neither a GPU
nor real weights.
"""
from __future__ import annotations

import logging

import torch
from torch import nn

from fastvideo.layers.linear import (
    LinearBase,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from fastvideo.layers.quantization.fp8_config import (
    FP8Config,
    FP8QuantizeMethod,
    convert_model_to_fp8,
)
from fastvideo.models.dits.minimax_h3 import MiniMaxH3FeedForward

# The two prefixes H3 actually builds, from ``transformer_blocks`` in
# ``MiniMaxH3Transformer3DModel`` and ``refiner_blocks`` in the token refiner.
# A made-up prefix would keep passing if the block constructors stopped
# propagating ``quant_config`` or renamed ``.ff``.
H3_MAIN_PREFIX = "minimax_h3.transformer_blocks.0.ff"
H3_REFINER_PREFIX = "minimax_h3.token_refiner.refiner_blocks.0.ff"
# What WanVideo and LTX-2 hand theirs. Must keep matching.
WAN_FFN_PREFIX = "wanvideo.blocks.0.ffn"


def _ff(prefix: str) -> MiniMaxH3FeedForward:
    return MiniMaxH3FeedForward(hidden_size=16, ffn_dim=32, quant_config=FP8Config(), prefix=prefix)


def test_minimax_h3_feed_forward_linears_get_fp8_method() -> None:
    """The regression: both H3 block families must be tagged for FP8."""
    for block_prefix in (H3_MAIN_PREFIX, H3_REFINER_PREFIX):
        ff = _ff(block_prefix)
        for attr in ("fc_in", "fc_out"):
            linear = getattr(ff, attr)
            assert isinstance(linear, ReplicatedLinear)
            assert isinstance(linear.quant_method, FP8QuantizeMethod), (
                f"{block_prefix}.{attr} fell back to {type(linear.quant_method).__name__}; "
                "H3's feed-forward would stay in bf16 under FP8")


def test_the_real_h3_block_propagates_quant_config_to_its_feed_forward() -> None:
    """Guard the wiring, not just the tuple.

    ``MiniMaxH3FeedForward`` is reached through a block that must pass both
    ``quant_config`` and a ``.ff`` prefix down. Dropping either would leave the
    tuple correct and the model unquantized, so build the block itself.
    """
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3TokenRefinerBlock
    from fastvideo.platforms import AttentionBackendEnum

    block = MiniMaxH3TokenRefinerBlock(
        16,
        2,
        8,
        32,
        1e-6,
        1e-6,
        (AttentionBackendEnum.TORCH_SDPA, ),
        FP8Config(),
        prefix="minimax_h3.token_refiner.refiner_blocks.0",
    )
    for attr in ("fc_in", "fc_out"):
        linear = getattr(block.ff, attr)
        assert isinstance(linear.quant_method, FP8QuantizeMethod), (
            f"block.ff.{attr} is {type(linear.quant_method).__name__}; the block "
            "stopped propagating quant_config or renamed .ff")


def test_ffn_named_stack_still_matches() -> None:
    """Adding the H3 names must not stop matching ``ffn.fc_in``/``ffn.fc_out``."""
    config = FP8Config()
    for name in ("fc_in", "fc_out"):
        linear = ReplicatedLinear(8, 8, bias=False, quant_config=config, prefix=f"{WAN_FFN_PREFIX}.{name}")
        assert isinstance(linear.quant_method, FP8QuantizeMethod)


def test_unrelated_linear_keeps_the_unquantized_fallback() -> None:
    """Projections and embedders stay in their loaded dtype, as intended."""
    linear = ReplicatedLinear(8, 8, bias=False, quant_config=FP8Config(), prefix="minimax_h3.proj_out")
    assert isinstance(linear.quant_method, UnquantizedLinearMethod)


def test_a_root_prefix_cannot_drag_every_descendant_into_fp8() -> None:
    """``.ff.fc_in`` is a true suffix, not a substring.

    ``DiTConfig.prefix`` is settable from the CLI. Under a substring rule a root
    prefix containing ``.ff.fc_in`` would tag every descendant, including
    ``adaln_proj.linear`` whose forward reads ``self.linear.weight.dtype`` after
    the converter has removed that weight.
    """
    linear = ReplicatedLinear(8, 8, bias=False, quant_config=FP8Config(), prefix="tenant.ff.fc_in.adaln_proj.linear")
    assert isinstance(linear.quant_method, UnquantizedLinearMethod)


class _FeedForward(nn.Module):

    def __init__(self, config: FP8Config) -> None:
        super().__init__()
        # 64 x 16 = 1024 weights.
        self.fc_in = ReplicatedLinear(16, 64, bias=False, quant_config=config, prefix=f"{H3_MAIN_PREFIX}.fc_in")


class _MixedModel(nn.Module):
    """One matched linear, one unmatched, one never handed an FP8Config.

    Nested so ``named_modules`` yields realistic dotted paths, which is what the
    conversion log reports.
    """

    def __init__(self) -> None:
        super().__init__()
        config = FP8Config()
        self.ff = _FeedForward(config)
        # 8 x 16 = 128 weights.
        self.proj_out = ReplicatedLinear(16, 8, bias=False, quant_config=config, prefix="minimax_h3.proj_out")
        # No quant_config at all, the way WanVideo builds its image embedder.
        self.untagged = ReplicatedLinear(16, 4, bias=False)


def test_conversion_reports_both_quantized_and_unmatched_parameters(caplog) -> None:
    """A whole stack going unquantized must be visible, named, and counted."""
    model = _MixedModel()
    for linear in model.modules():
        if isinstance(linear, LinearBase):
            linear.weight.data = torch.randn_like(linear.weight.data)

    with caplog.at_level(logging.INFO, logger="fastvideo.layers.quantization.fp8_config"):
        convert_model_to_fp8(model)

    records = [r for r in caplog.records if r.getMessage().startswith("FP8: quantized")]
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    # The parameter totals reach the logger already divided by 1e9 for display.
    quantized_layers, quantized_billions, unmatched_layers, unmatched_billions, names, untagged = records[0].args

    # Assert the arguments, not the rendered string: both totals format to
    # "0.00B" at this size, so a hardcoded zero would pass a string check.
    assert quantized_layers == 1
    assert quantized_billions == 16 * 64 / 1e9
    assert unmatched_layers == 1
    assert unmatched_billions == 16 * 8 / 1e9
    assert untagged == 1
    # Naming the unmatched layer is what makes a missing stack actionable
    # rather than merely visible.
    assert names == "proj_out"

    assert isinstance(model.ff.fc_in.quant_method, FP8QuantizeMethod)
    assert model.ff.fc_in._fp8_weight.dtype is torch.float8_e4m3fn
    assert "weight" not in model.ff.fc_in._parameters
    assert isinstance(model.proj_out.quant_method, UnquantizedLinearMethod)
    assert model.proj_out.weight.dtype is not torch.float8_e4m3fn
