# SPDX-License-Identifier: Apache-2.0
"""LTX-2 NVFP4 wiring contract tests.

These tests lock in the public-facing FP4 contract so the wire-up
between :class:`NVFP4Config`, the LTX-2 DiT (linear-class swap +
``prefix=`` plumbing), and the loader (``convert_model_to_nvfp4``
trigger via registered ``quant_method``) doesn't silently regress.

We don't need flashinfer or a GPU to assert the wiring: the layer's
``quant_method`` attachment, the prefix used by
``FP4Config.get_quant_method``, and the ``UnquantizedLinearMethod``
fallback for non-tagged layers are all CPU-only contracts.
"""
from __future__ import annotations

from fastvideo.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from fastvideo.layers.quantization.nvfp4_config import (
    NVFP4Config,
    NVFP4QuantizeMethod,
)
from fastvideo.models.dits.ltx2 import (
    BasicAVTransformerBlock,
    FeedForward,
    LTXRopeType,
    LTXSelfAttention,
    TransformerConfig,
)
from fastvideo.platforms import AttentionBackendEnum


def _matched_attn() -> LTXSelfAttention:
    """Self-attention with a prefix in the NVFP4 layer set."""
    return LTXSelfAttention(
        query_dim=64,
        context_dim=None,
        heads=2,
        dim_head=32,
        norm_eps=1e-6,
        rope_type=LTXRopeType.INTERLEAVED,
        supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ),
        quant_config=NVFP4Config(),
        prefix="ltx2.blocks.0.attn1",
    )


def test_self_attention_to_q_k_v_out_get_nvfp4_method() -> None:
    attn = _matched_attn()
    for attr in ("to_q", "to_k", "to_v"):
        linear = getattr(attn, attr)
        assert isinstance(linear, ReplicatedLinear), (
            f"{attr} must be ReplicatedLinear, got {type(linear).__name__}")
        assert isinstance(linear.quant_method, NVFP4QuantizeMethod), (
            f"{attr}.quant_method must be NVFP4QuantizeMethod, got "
            f"{type(linear.quant_method).__name__}")
    out_linear = attn.to_out[0]
    assert isinstance(out_linear, ReplicatedLinear)
    assert isinstance(out_linear.quant_method, NVFP4QuantizeMethod)


def test_self_attention_layer_prefixes_match_fp4_target_paths() -> None:
    attn = _matched_attn()
    assert attn.to_q.quant_method.layer_prefix == "ltx2.blocks.0.attn1.to_q"
    assert attn.to_k.quant_method.layer_prefix == "ltx2.blocks.0.attn1.to_k"
    assert attn.to_v.quant_method.layer_prefix == "ltx2.blocks.0.attn1.to_v"
    assert attn.to_out[0].quant_method.layer_prefix == "ltx2.blocks.0.attn1.to_out"


def test_unmatched_prefix_falls_back_to_unquantized() -> None:
    """Layers whose prefix isn't in ``NVFP4Config.fp4_layers`` must
    receive ``UnquantizedLinearMethod`` so the asserts in
    ``LinearBase`` subclasses don't fire."""
    attn = LTXSelfAttention(
        query_dim=64,
        context_dim=128,
        heads=2,
        dim_head=32,
        norm_eps=1e-6,
        rope_type=LTXRopeType.INTERLEAVED,
        supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ),
        quant_config=NVFP4Config(),
        prefix="some.unrelated.prefix",
    )
    assert isinstance(attn.to_q.quant_method, UnquantizedLinearMethod)


def test_no_quant_config_keeps_unquantized_methods() -> None:
    attn = LTXSelfAttention(
        query_dim=64,
        context_dim=None,
        heads=2,
        dim_head=32,
        norm_eps=1e-6,
        rope_type=LTXRopeType.INTERLEAVED,
        supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ),
        quant_config=None,
        prefix="ltx2.blocks.0.attn1",
    )
    assert isinstance(attn.to_q.quant_method, UnquantizedLinearMethod)
    assert isinstance(attn.to_out[0].quant_method, UnquantizedLinearMethod)


def test_feedforward_fc_in_fc_out_get_nvfp4_method() -> None:
    ff = FeedForward(
        dim=64,
        dim_out=64,
        mult=2,
        quant_config=NVFP4Config(),
        prefix="ltx2.blocks.0",
    )
    fc_in = ff.net[0].proj
    fc_out = ff.net[2]
    assert isinstance(fc_in, ReplicatedLinear)
    assert isinstance(fc_out, ReplicatedLinear)
    assert isinstance(fc_in.quant_method, NVFP4QuantizeMethod)
    assert isinstance(fc_out.quant_method, NVFP4QuantizeMethod)
    assert fc_in.quant_method.layer_prefix == "ltx2.blocks.0.ffn.fc_in"
    assert fc_out.quant_method.layer_prefix == "ltx2.blocks.0.ffn.fc_out"


def test_basic_av_block_propagates_quant_config_to_all_children() -> None:
    """``BasicAVTransformerBlock`` is the seam where ``quant_config``
    branches into the four attention modules and FFN. Verify each
    child sees the config and ends up with the right prefix.
    """
    video_cfg = TransformerConfig(dim=64, heads=2, d_head=32, context_dim=128)
    audio_cfg = TransformerConfig(dim=64, heads=2, d_head=32, context_dim=128)
    block = BasicAVTransformerBlock(
        idx=3,
        video=video_cfg,
        audio=audio_cfg,
        rope_type=LTXRopeType.INTERLEAVED,
        norm_eps=1e-6,
        use_distributed_attention=False,
        quant_config=NVFP4Config(),
        prefix="ltx2",
    )
    # NVFP4 layer set is deliberately asymmetric — only the modules
    # whose runtime cost dominates the DiT step are quantized:
    #   * attn1 (video self-attn): to_q/to_k/to_v/to_out
    #   * attn2 (text cross-attn for video): to_q + to_out only
    #     (text context isn't quantized)
    #   * audio_to_video_attn: to_q + to_out only
    #   * video_to_audio_attn: to_k + to_v only
    #   * ff (video FFN): fc_in + fc_out
    # Audio self/cross attention and audio FFN are NOT quantized in
    # the LTX-2 NVFP4 set (the audio path is much cheaper than video).
    nvfp4_expectations = {
        block.attn1.to_q: "ltx2.blocks.3.attn1.to_q",
        block.attn1.to_out[0]: "ltx2.blocks.3.attn1.to_out",
        block.attn2.to_q: "ltx2.blocks.3.attn2.to_q",
        block.attn2.to_out[0]: "ltx2.blocks.3.attn2.to_out",
        block.audio_to_video_attn.to_q:
        "ltx2.blocks.3.audio_to_video_attn.to_q",
        block.video_to_audio_attn.to_v:
        "ltx2.blocks.3.video_to_audio_attn.to_v",
        block.ff.net[0].proj: "ltx2.blocks.3.ffn.fc_in",
        block.ff.net[2]: "ltx2.blocks.3.ffn.fc_out",
    }
    for linear, expected_prefix in nvfp4_expectations.items():
        assert isinstance(linear, ReplicatedLinear)
        assert isinstance(linear.quant_method, NVFP4QuantizeMethod), (
            f"{expected_prefix} expected NVFP4QuantizeMethod, got "
            f"{type(linear.quant_method).__name__}")
        assert linear.quant_method.layer_prefix == expected_prefix

    # Confirm the non-quantized projections still get the
    # UnquantizedLinearMethod fallback (they're ReplicatedLinear, just
    # not in the NVFP4 set).
    unquantized_projections = (
        block.attn2.to_k,
        block.attn2.to_v,
        block.audio_attn1.to_q,
        block.audio_attn1.to_k,
        block.audio_attn1.to_v,
        block.audio_attn1.to_out[0],
        block.audio_attn2.to_q,
        block.audio_attn2.to_k,
        block.audio_attn2.to_v,
        block.audio_attn2.to_out[0],
        block.audio_ff.net[0].proj,
        block.audio_ff.net[2],
    )
    for linear in unquantized_projections:
        assert isinstance(linear, ReplicatedLinear)
        assert isinstance(linear.quant_method, UnquantizedLinearMethod)
