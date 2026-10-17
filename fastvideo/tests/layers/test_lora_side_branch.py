# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the LoRA side branch used on packed (NVFP4/FP8-like) linears.

A packed linear cannot absorb a low-rank delta without being re-quantized, so the adapter runs beside it as
``y = base(x) + strength * (alpha / rank) * B(A x)``. A stand-in packed method (int8 weight times a scale) keeps
the tests on CPU while exercising the same wrapper state the real NVFP4/FP8 methods leave behind: no ``weight``
parameter and a populated ``_nvfp4_weight`` buffer.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import fastvideo.envs as envs
from fastvideo.layers.lora import linear as lora_linear
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.lora.linear import BaseLayerWithLoRA, get_lora_layer, requires_side_branch

IN, OUT, RANK = 16, 12, 4


@pytest.fixture(autouse=True)
def _eager_forward(monkeypatch: pytest.MonkeyPatch):
    """``BaseLayerWithLoRA.forward`` is ``torch.compile``d; run it eagerly, and merge on CPU even on MPS hosts."""
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    monkeypatch.setattr(lora_linear, "get_local_torch_device", lambda: torch.device("cpu"))


class _PackedMethod:
    """Stand-in for a packed-weight quant method: dequantizes ``_nvfp4_weight * scale`` on every call."""

    def apply(self, layer, x, bias=None):
        weight = layer._nvfp4_weight.to(x.dtype) * layer._packed_scale.to(x.dtype)
        return F.linear(x, weight, bias)


def _dense_linear(seed: int = 0) -> ReplicatedLinear:
    torch.manual_seed(seed)
    layer = ReplicatedLinear(IN, OUT, bias=False, params_dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(OUT, IN))
    return layer


def _packed_linear(seed: int = 0) -> ReplicatedLinear:
    layer = _dense_linear(seed)
    quantized = torch.round(layer.weight.detach().float() * 8).to(torch.int8)
    layer.register_parameter("weight", None)
    layer.register_buffer("_nvfp4_weight", quantized, persistent=False)
    layer.register_buffer("_packed_scale", torch.tensor(1 / 8), persistent=False)
    layer.quant_method = _PackedMethod()
    return layer


def _factors(seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(RANK, IN, generator=generator), torch.randn(OUT, RANK, generator=generator)


def _x(seed: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(3, 5, IN, generator=generator).to(torch.bfloat16)


def test_packed_linear_selects_side_branch_and_dense_linear_merges() -> None:
    assert requires_side_branch(_packed_linear())
    assert not requires_side_branch(_dense_linear())
    assert get_lora_layer(_packed_linear()).side_branch
    assert not get_lora_layer(_dense_linear()).side_branch


def test_env_forces_side_branch_on_dense_linear() -> None:
    with envs.FASTVIDEO_LORA_SIDE_BRANCH.override(True):
        assert get_lora_layer(_dense_linear()).side_branch


def test_side_branch_matches_base_plus_scaled_low_rank_product() -> None:
    base = _packed_linear()
    layer = get_lora_layer(base)
    a, b = _factors()
    layer.set_lora_weights(a, b, lora_alpha=8, strength=0.75)
    x = _x()

    out, _ = layer(x)

    expected = base(x)[0].float() + 0.75 * (8 / RANK) * (x.float() @ a.bfloat16().float().T @ b.bfloat16().float().T)
    torch.testing.assert_close(out.float(), expected, atol=0.15, rtol=0.05)
    assert not layer.merged


def test_side_branch_never_modifies_the_packed_base_weight() -> None:
    base = _packed_linear()
    packed_before = base._nvfp4_weight.clone()
    layer = get_lora_layer(base)
    a, b = _factors()

    layer.set_lora_weights(a, b, strength=1.0)
    layer.merge_lora_weights()
    layer.unmerge_lora_weights()

    assert torch.equal(base._nvfp4_weight, packed_before)
    assert base.weight is None
    assert layer.cpu_weight is None


def test_strength_zero_is_bitwise_identical_to_the_unadapted_layer() -> None:
    base = _packed_linear()
    x = _x()
    reference, _ = base(x)
    layer = get_lora_layer(base)
    a, b = _factors()

    layer.set_lora_weights(a, b, lora_alpha=16, strength=0.0)
    out, _ = layer(x)

    assert torch.equal(out, reference)


def test_strength_can_change_at_runtime_without_reloading() -> None:
    layer = get_lora_layer(_packed_linear())
    a, b = _factors()
    x = _x()
    layer.set_lora_weights(a, b, strength=1.0)
    full, _ = layer(x)
    layer.set_lora_weights(a, b, strength=0.5)
    half, _ = layer(x)
    layer.set_lora_weights(a, b, strength=0.0)
    off, _ = layer(x)

    base_out, _ = layer.base_layer(x)
    torch.testing.assert_close((half - base_out).float(), 0.5 * (full - base_out).float(), atol=0.1, rtol=0.05)
    assert torch.equal(off, base_out)


def test_layer_without_an_adapter_behaves_like_the_base_layer() -> None:
    base = _packed_linear()
    layer = get_lora_layer(base)
    x = _x()
    assert torch.equal(layer(x)[0], base(x)[0])


def test_side_branch_rejects_accumulating_a_second_adapter() -> None:
    layer = get_lora_layer(_packed_linear())
    a, b = _factors()
    layer.set_lora_weights(a, b)
    with pytest.raises(ValueError, match="Accumulating"):
        layer.set_lora_weights(a, b, accumulate=True)


def test_dense_linear_still_merges_and_unmerges_exactly() -> None:
    base = _dense_linear()
    original = base.weight.detach().clone()
    layer = get_lora_layer(base)
    a, b = _factors()

    layer.set_lora_weights(a, b, lora_alpha=RANK, strength=1.0)
    assert layer.merged
    assert not torch.equal(base.weight.detach(), original)
    layer.unmerge_lora_weights()
    assert torch.equal(base.weight.detach(), original)


def test_parallel_wrappers_keep_merging() -> None:
    assert BaseLayerWithLoRA.supports_side_branch
    from fastvideo.layers.lora.linear import ColumnParallelLinearWithLoRA, RowParallelLinearWithLoRA
    assert not ColumnParallelLinearWithLoRA.supports_side_branch
    assert not RowParallelLinearWithLoRA.supports_side_branch


def _pipeline_over(layer: BaseLayerWithLoRA) -> "SimpleNamespace":
    from types import SimpleNamespace
    transformer = torch.nn.Module()
    transformer.layer = layer
    layers = SimpleNamespace(lora_layers_by_block=lambda: iter([(None, {"layer": layer})]))
    return SimpleNamespace(lora_layers={"transformer": layers}, trainable_transformer_modules={"transformer": transformer})


def test_packed_nvfp4_layers_behind_a_side_branch_do_not_block_adapter_changes() -> None:
    from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod
    from fastvideo.pipelines.lora_pipeline import _all_side_branch, _has_nvfp4_weights_without_bf16

    base = _packed_linear()
    base.quant_method = object.__new__(NVFP4QuantizeMethod)
    pipeline = _pipeline_over(get_lora_layer(base))

    assert not _has_nvfp4_weights_without_bf16(pipeline, pipeline.trainable_transformer_modules)
    assert _all_side_branch([pipeline.lora_layers["transformer"].lora_layers_by_block().__next__()[1]["layer"]])


def test_forced_side_branch_over_unpacked_quantized_base_still_packs(monkeypatch: pytest.MonkeyPatch) -> None:
    """FASTVIDEO_LORA_SIDE_BRANCH=1 over a dense checkpoint must still run the deferred NVFP4/MXFP8 packing."""
    from fastvideo.layers.quantization.mxfp8_config import MXFP8QuantizeMethod
    from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod
    from fastvideo.pipelines import lora_pipeline

    monkeypatch.setenv("FASTVIDEO_LORA_SIDE_BRANCH", "1")
    for method_cls, buffer in ((NVFP4QuantizeMethod, "_nvfp4_weight"), (MXFP8QuantizeMethod, "_mxfp8_weight")):
        base = _dense_linear()
        base.quant_method = object.__new__(method_cls)
        base.register_buffer(buffer, None, persistent=False)
        layer = get_lora_layer(base)
        assert layer.side_branch
        assert not lora_pipeline._all_side_branch([layer]), f"{buffer}: unpacked base must not skip packing"

        base.register_buffer(buffer, torch.zeros(1), persistent=False)
        assert lora_pipeline._all_side_branch([layer]), f"{buffer}: once packed, repacking is skipped"
