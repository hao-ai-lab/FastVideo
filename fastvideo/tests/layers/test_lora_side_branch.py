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


@pytest.mark.parametrize("packed", [False, True])
def test_env_forces_side_branch_on_retained_dense_weight(packed: bool) -> None:
    base = _dense_linear()
    if packed:
        base.register_buffer("_nvfp4_weight", torch.zeros(1), persistent=False)
    assert not get_lora_layer(base).side_branch
    with envs.FASTVIDEO_LORA_SIDE_BRANCH.override(True):
        assert get_lora_layer(base).side_branch


def test_side_branch_chunked_update_matches_the_full_product(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lora_linear, "_SIDE_BRANCH_CHUNK_BYTES", 64)
    base = _packed_linear()
    layer = get_lora_layer(base)
    a, b = _factors()
    layer.set_lora_weights(a, b, lora_alpha=8, strength=0.75)
    x = _x()

    out, _ = layer(x)

    expected = base(x)[0].float() + 0.75 * (8 / RANK) * (x.float() @ a.bfloat16().float().T @ b.bfloat16().float().T)
    torch.testing.assert_close(out.float(), expected, atol=0.15, rtol=0.05)


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_side_branch_chunked_update_compiles_fullgraph(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    if backend == "inductor" and not torch.cuda.is_available():
        pytest.skip("Inductor execution requires a CUDA GPU for this regression")
    device = torch.device("cuda" if backend == "inductor" else "cpu")
    monkeypatch.setattr(torch._dynamo.config, "disable", False)
    monkeypatch.setattr(lora_linear, "_SIDE_BRANCH_CHUNK_BYTES", 64)
    torch._dynamo.reset()
    base = _packed_linear().to(device)
    packed_before = base._nvfp4_weight.clone()
    layer = get_lora_layer(base)
    a, b = (factor.to(device) for factor in _factors())
    layer.set_lora_weights(a, b, lora_alpha=8, strength=0.75)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = layer

        def forward(self, x):
            return self.proj(x)[0] + x[..., :OUT]

    block = Block()
    compiled = torch.compile(block, backend=backend, fullgraph=True)
    try:
        with torch.inference_mode():
            for tokens in (5, 7):
                x = torch.randn(3, tokens, IN, device=device, dtype=torch.bfloat16)
                base_out = base(x)[0].float()
                delta = x.float() @ a.bfloat16().float().T @ b.bfloat16().float().T
                expected = base_out + 0.75 * (8 / RANK) * delta + x[..., :OUT].float()
                out = compiled(x)
                torch.testing.assert_close(out.float(), expected, atol=0.2, rtol=0.05)
            layer.lora_strength = 0.0
            torch.testing.assert_close(compiled(x), base(x)[0] + x[..., :OUT], rtol=0, atol=0)
            layer.lora_strength = 0.5
            expected = base_out + 0.5 * (8 / RANK) * delta + x[..., :OUT].float()
            torch.testing.assert_close(compiled(x).float(), expected, atol=0.2, rtol=0.05)
        assert torch.equal(base._nvfp4_weight, packed_before)
    finally:
        torch._dynamo.reset()


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


def test_runtime_adapter_on_packed_nvfp4_without_constructor_lora_is_not_blocked() -> None:
    """Wrapping runs before the NVFP4 check, so packed linears that become side branches do not block it."""
    from types import SimpleNamespace

    from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    base = _packed_linear()
    base.quant_method = object.__new__(NVFP4QuantizeMethod)
    transformer = torch.nn.Module()
    transformer.layer = base
    pipeline = SimpleNamespace(_setting_constructor_adapter=False,
                               _constructor_dense_lora_path=None,
                               cur_adapter_name="",
                               cur_adapter_path="",
                               cur_adapter_strength=1.0,
                               lora_adapter_paths={},
                               lora_adapters={},
                               lora_initialized=False,
                               lora_layers={},
                               trainable_transformer_modules={"transformer": transformer})

    def convert_to_lora_layers() -> None:
        transformer.layer = get_lora_layer(base)
        layers = SimpleNamespace(lora_layers_by_block=lambda: iter([(None, {"layer": transformer.layer})]))
        pipeline.lora_layers["transformer"] = layers
        pipeline.lora_initialized = True

    pipeline.convert_to_lora_layers = convert_to_lora_layers
    # The adapter is unknown and has no path, so the call stops at the lookup that follows the NVFP4 check.
    with pytest.raises(ValueError, match="not found in the pipeline"):
        LoRAPipeline.set_lora_adapter(pipeline, "runtime-adapter")
    assert transformer.layer.side_branch


def test_forced_side_branch_over_unpacked_quantized_base_still_packs(env_overrides) -> None:
    """FASTVIDEO_LORA_SIDE_BRANCH=1 over a dense checkpoint must still run the deferred NVFP4/MXFP8 packing."""
    from fastvideo.layers.quantization.mxfp8_config import MXFP8QuantizeMethod
    from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod
    from fastvideo.pipelines import lora_pipeline

    env_overrides.enter_context(envs.FASTVIDEO_LORA_SIDE_BRANCH.override(True))
    for method_cls, buffer in ((NVFP4QuantizeMethod, "_nvfp4_weight"), (MXFP8QuantizeMethod, "_mxfp8_weight")):
        base = _dense_linear()
        base.quant_method = object.__new__(method_cls)
        base.register_buffer(buffer, None, persistent=False)
        layer = get_lora_layer(base)
        assert layer.side_branch
        assert not lora_pipeline._all_side_branch([layer]), f"{buffer}: unpacked base must not skip packing"

        base.register_buffer(buffer, torch.zeros(1), persistent=False)
        assert lora_pipeline._all_side_branch([layer]), f"{buffer}: once packed, repacking is skipped"


def _attention_only_block(env_overrides):
    """A block whose attention is LoRA-wrapped while its FFN linear still awaits the NVFP4 packing."""
    from types import SimpleNamespace

    from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod

    env_overrides.enter_context(envs.FASTVIDEO_LORA_SIDE_BRANCH.override(True))
    attention = get_lora_layer(_dense_linear())
    ffn = _dense_linear(seed=2)
    ffn.quant_method = object.__new__(NVFP4QuantizeMethod)
    ffn.register_buffer("_nvfp4_weight", None, persistent=False)
    block = torch.nn.Module()
    block.attn = attention
    block.ff = ffn
    layers = SimpleNamespace(lora_layers_by_block=lambda: iter([(block, {"attn": attention})]))
    pipeline = SimpleNamespace(lora_layers={"transformer": layers}, trainable_transformer_modules={})
    return pipeline, block, ffn


@pytest.mark.parametrize("method_name", ["merge_lora_weights", "unmerge_lora_weights"])
def test_unwrapped_linear_awaiting_packing_blocks_the_repack_skip(monkeypatch: pytest.MonkeyPatch,
                                                                  env_overrides,
                                                                  method_name: str) -> None:
    from fastvideo.pipelines import lora_pipeline

    pipeline, block, ffn = _attention_only_block(env_overrides)
    skips: list[bool] = []
    monkeypatch.setattr(lora_pipeline, "_convert_quantized_weights_after_lora_change",
                        lambda modules, skip=False: skips.append(skip))

    getattr(lora_pipeline.LoRAPipeline, method_name)(pipeline)
    assert skips == [False]

    ffn.register_buffer("_nvfp4_weight", torch.zeros(1), persistent=False)
    getattr(lora_pipeline.LoRAPipeline, method_name)(pipeline)
    assert skips == [False, True]


def test_attention_only_adapter_skips_repack_only_once_the_whole_block_is_packed(
        env_overrides) -> None:
    from fastvideo.pipelines.lora_pipeline import _all_side_branch

    pipeline, block, ffn = _attention_only_block(env_overrides)
    layers = list(pipeline.lora_layers["transformer"].lora_layers_by_block())[0][1].values()

    assert not _all_side_branch(layers, [block])
    ffn.register_buffer("_nvfp4_weight", torch.zeros(1), persistent=False)
    assert _all_side_branch(layers, [block])
