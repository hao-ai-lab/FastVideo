# SPDX-License-Identifier: Apache-2.0
"""One activation quantization per compatible H3 attention projection group."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from fastvideo.models.dits.minimax_h3_vsa_fp4 import _shared_input_projections


class _FakeLinear(nn.Module):
    """Minimal linear stand-in that records whether its own forward ran."""

    def __init__(self, method: object) -> None:
        super().__init__()
        self.quant_method = method
        self.bias = None
        self.own_forwards = 0

    def forward(self, x: torch.Tensor):
        self.own_forwards += 1
        return (x, None)


class _FakeNVFP4Method:
    """Version of ``NVFP4QuantizeMethod`` with a countable activation quantization."""

    def __init__(self, unit_scale: bool = True) -> None:
        self.unit_scale = unit_scale
        self.quantize_calls = 0
        self.apply_calls = 0

    def wants_prequantized_input(self) -> bool:
        return True

    def uses_unit_activation_scale(self, layer: object) -> bool:
        del layer
        return self.unit_scale

    def quantize_input(self, x: torch.Tensor):
        self.quantize_calls += 1
        return (x + 1, "scale", "global")

    def apply(self, layer: object, x: torch.Tensor, bias: torch.Tensor | None, pre_quantized):
        del x, bias
        assert pre_quantized is not None
        self.apply_calls += 1
        return pre_quantized[0]


def test_shared_input_projections_quantize_unit_scale_nvfp4_group_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastvideo.layers.quantization.nvfp4_config as nvfp4

    monkeypatch.setattr(nvfp4, "NVFP4QuantizeMethod", _FakeNVFP4Method)
    methods = [_FakeNVFP4Method() for _ in range(4)]
    linears = tuple(_FakeLinear(method) for method in methods)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    outputs = _shared_input_projections(linears, x)

    assert [method.quantize_calls for method in methods] == [1, 0, 0, 0]
    assert [method.apply_calls for method in methods] == [1, 1, 1, 1]
    assert [linear.own_forwards for linear in linears] == [0, 0, 0, 0]
    for output in outputs:
        torch.testing.assert_close(output, x + 1)


def test_shared_input_projections_leave_calibrated_nvfp4_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastvideo.layers.quantization.nvfp4_config as nvfp4

    monkeypatch.setattr(nvfp4, "NVFP4QuantizeMethod", _FakeNVFP4Method)
    methods = [_FakeNVFP4Method(), _FakeNVFP4Method(), _FakeNVFP4Method(), _FakeNVFP4Method(unit_scale=False)]
    linears = tuple(_FakeLinear(method) for method in methods)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    _shared_input_projections(linears, x)

    assert [method.quantize_calls for method in methods] == [1, 0, 0, 0]
    assert [method.apply_calls for method in methods] == [1, 1, 1, 0]
    assert [linear.own_forwards for linear in linears] == [0, 0, 0, 1]


def _install_fp8_buffers(layer, granularity: str, keep_weight: bool = False) -> None:
    weight = layer.weight.data.float()
    if granularity == "channel":
        scale = (weight.abs().amax(dim=1, keepdim=True) / 448).clamp_min(1e-6)
    else:
        scale = (weight.abs().amax().reshape(1) / 448).clamp_min(1e-6)
    layer.register_buffer("_fp8_weight", (weight / scale).to(torch.float8_e4m3fn))
    layer.register_buffer("_fp8_weight_scale", scale)
    if not keep_weight:
        layer.register_parameter("weight", None)


@pytest.mark.parametrize("granularity", ["tensor", "channel"])
def test_shared_fp8_projections_cover_qkv_and_gate(granularity: str) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("sm89+ CUDA is required for FP8 GEMM")
    from fastvideo.layers.linear import ReplicatedLinear
    from fastvideo.layers.quantization.fp8_config import FP8Config, FP8QuantizeMethod

    torch.manual_seed(23)
    linears = tuple(
        ReplicatedLinear(128, 256, bias=True, quant_config=FP8Config(granularity), prefix=f"block.attn.{name}")
        for name in ("to_q", "to_k", "to_v", "to_gate_compress"))
    linears[3].quant_method = FP8QuantizeMethod(granularity)
    for layer in linears:
        layer.to(device="cuda", dtype=torch.bfloat16)
        layer.weight.data.normal_(std=0.1)
        layer.bias.data.normal_(std=0.1)
        _install_fp8_buffers(layer, granularity)
    x = torch.randn(1, 272, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        reference = [layer(x)[0] for layer in linears]
        with patch.object(FP8QuantizeMethod, "quantize_input", autospec=True,
                          side_effect=FP8QuantizeMethod.quantize_input) as quant:
            actual = _shared_input_projections(linears, x)
        assert quant.call_count == 1
        for expected, output in zip(reference, actual, strict=True):
            torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_shared_fp8_projections_keep_qkv_sharing_with_dense_gate() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("sm89+ CUDA is required for FP8 GEMM")
    from fastvideo.layers.linear import ReplicatedLinear
    from fastvideo.layers.quantization.fp8_config import FP8Config, FP8QuantizeMethod

    torch.manual_seed(29)
    linears = tuple(
        ReplicatedLinear(128, 256, bias=True, quant_config=FP8Config("tensor"), prefix=f"block.attn.{name}")
        for name in ("to_q", "to_k", "to_v"))
    linears += (ReplicatedLinear(128, 256, bias=True, prefix="block.attn.to_gate_compress"), )
    for layer in linears[:3]:
        layer.to(device="cuda", dtype=torch.bfloat16)
        layer.weight.data.normal_(std=0.1)
        layer.bias.data.normal_(std=0.1)
        _install_fp8_buffers(layer, "tensor")
    linears[3].to(device="cuda", dtype=torch.bfloat16)
    linears[3].weight.data.normal_(std=0.1)
    linears[3].bias.data.normal_(std=0.1)
    x = torch.randn(1, 272, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        reference = [layer(x)[0] for layer in linears]
        with patch.object(FP8QuantizeMethod, "quantize_input", autospec=True,
                          side_effect=FP8QuantizeMethod.quantize_input) as quant:
            actual = _shared_input_projections(linears, x)
        assert quant.call_count == 1
        for expected, output in zip(reference, actual, strict=True):
            torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_generic_h3_attention_quantizes_qkv_and_gate_once(env_overrides, distributed_setup) -> None:
    """The gate must reuse the q/k/v activation quantization, not prepare it again."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("sm89+ CUDA is required for FP8 GEMM")
    import fastvideo.layers.quantization.fp8_config as fp8_config
    import fastvideo.envs as envs
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAMetadataBuilder
    from fastvideo.forward_context import set_forward_context
    from fastvideo.layers.quantization.fp8_config import FP8Config
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3Attention
    from fastvideo.platforms import AttentionBackendEnum

    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("VIDEO_SPARSE_ATTN_H3"))
    env_overrides.enter_context(envs.override_external("FASTVIDEO_VSA_TRITON", "1"))
    env_overrides.enter_context(envs.FASTVIDEO_VSA_PTX_BLACKWELL.override(False))
    env_overrides.enter_context(envs.FASTVIDEO_H3_VSA_FP4.override(False))
    env_overrides.enter_context(envs.FASTVIDEO_H3_VSA_TILE_FIRST.override(False))
    from fastvideo.layers.quantization.fp8_config import FP8QuantizeMethod

    class _GateFP8Config(FP8Config):
        """FP8 config that also tags the VSA compression gate (the NVFP4 profile's set)."""

        def get_quant_method(self, layer, prefix):
            if prefix.endswith("to_gate_compress"):
                from fastvideo.layers.linear import LinearBase
                if isinstance(layer, LinearBase):
                    return FP8QuantizeMethod(self.granularity)
            return super().get_quant_method(layer, prefix)

    torch.manual_seed(31)
    attn = MiniMaxH3Attention(256, 2, 128, 1e-5, (AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3,),
                              _GateFP8Config("channel"), "transformer_blocks.0.attn")
    attn.to(device="cuda", dtype=torch.bfloat16)
    for parameter in attn.parameters():
        parameter.data.normal_(std=0.1)
    for layer in (attn.to_q, attn.to_k, attn.to_v, attn.to_out):
        _install_fp8_buffers(layer, "channel")
    _install_fp8_buffers(attn.to_gate_compress, "channel", keep_weight=True)
    meta = MiniMaxH3VSAMetadataBuilder().build(current_timestep=999, patch_size=(1, 1, 1),
                                               VSA_sparsity=0.8, packed_segments=(65, 97, (4, 6, 10)),
                                               device=torch.device("cuda"), tile_size=64)
    length = meta.total_seq_length
    x = torch.randn(1, length, 256, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(length, 96, device="cuda")
    rope = angles.cos(), angles.sin()
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=meta):
        with patch.object(fp8_config, "_quantize_rowwise", side_effect=fp8_config._quantize_rowwise) as quant:
            attn(x, rope, length)
        assert quant.call_count == 2


@pytest.mark.parametrize("tile_first", [False, True])
def test_h3_attention_nvfp4_quantizes_qkv_and_gate_once(env_overrides, distributed_setup, tile_first: bool) -> None:
    """The NVFP4 ``h3_dit_vsa`` profile prepares q/k/v/gate activations once."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MiniMax-H3 NVFP4 inference requires an NVIDIA Blackwell GPU")
    pytest.importorskip("flashinfer")
    import fastvideo.envs as envs
    import fastvideo.layers.quantization.nvfp4_config as nvfp4
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAMetadataBuilder
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3Attention
    from fastvideo.platforms import AttentionBackendEnum

    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("VIDEO_SPARSE_ATTN_H3"))
    env_overrides.enter_context(envs.override_external("FASTVIDEO_VSA_TRITON", "1"))
    env_overrides.enter_context(envs.FASTVIDEO_VSA_PTX_BLACKWELL.override(False))
    env_overrides.enter_context(envs.FASTVIDEO_H3_VSA_FP4.override(False))
    env_overrides.enter_context(envs.FASTVIDEO_H3_VSA_TILE_FIRST.override(tile_first))
    torch.manual_seed(37)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        attn = MiniMaxH3Attention(256, 2, 128, 1e-5, (AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3,),
                                  nvfp4.NVFP4Config(layer_profile="h3_dit_vsa"), "transformer_blocks.0.attn")
    finally:
        torch.set_default_dtype(previous_dtype)
    attn = attn.cuda().eval()
    for parameter in attn.parameters():
        parameter.data.normal_(std=0.1)
    nvfp4.convert_model_to_nvfp4(attn)
    assert attn.to_q.weight is None
    assert attn.to_gate_compress.weight is None
    assert attn._gate_active()
    meta = MiniMaxH3VSAMetadataBuilder().build(current_timestep=999, patch_size=(1, 1, 1),
                                               VSA_sparsity=0.8, packed_segments=(65, 97, (4, 6, 10)),
                                               device=torch.device("cuda"), tile_size=64)
    length = meta.total_seq_length
    x = torch.randn(1, length, 256, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(length, 96, device="cuda")
    rope = angles.cos(), angles.sin()
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=meta):
        with patch.object(nvfp4, "_nvfp4_quantize", side_effect=nvfp4._nvfp4_quantize) as quant:
            attn(x, rope, length)
        assert quant.call_count == 2
