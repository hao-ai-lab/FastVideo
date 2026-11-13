# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for replaying active LoRA contributions after an NVFP4 purge."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.lora.linear import get_lora_layer
from fastvideo.layers.quantization.mxfp8_config import MXFP8QuantizeMethod
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.tests.ops.quantization.test_nvfp4_minimax_h3_wiring import (
    _assert_cpu_packed_weight,
    _CpuPackedNVFP4Method,
    _cpu_pack_nvfp4,
    _nvfp4_lifecycle_pipeline,
    cpu_nvfp4_lifecycle,
)


def _reference_update(weight, factors, *, alpha=None, strength=1.0):
    a, b = factors
    rank = a.shape[0]
    scale = ((alpha / rank) if alpha is not None else 1.0) * strength
    dtype = torch.promote_types(weight.dtype, a.dtype)
    delta = b.to(dtype) @ a.to(dtype)
    return (weight.to(dtype) + delta * scale).to(weight.dtype)


def _install(pipeline, name, factors, *, alpha=None, strength=1.0, accumulate=True):
    a, b = factors
    pipeline.lora_adapters[name] = {"linear.lora_A": a, "linear.lora_B": b}
    if alpha is not None:
        pipeline.lora_adapters[name]["linear.lora_alpha"] = alpha
    LoRAPipeline.set_lora_adapter(pipeline, name, strength=strength, accumulate=accumulate)


def test_accumulation_replays_multiple_adapters_with_original_dtype_and_scale(cpu_nvfp4_lifecycle):
    pipeline, wrapped, original, first, second = _nvfp4_lifecycle_pipeline(False)
    third = (torch.randn(2, 32) / 8, torch.randn(32, 2) / 4)
    expected = _reference_update(original, first)

    for name, factors, alpha, strength in [
        ("second", second, 4.0, 0.5),
        ("third", third, 3.5, -0.75),
        ("third", third, 3.5, -0.75),
    ]:
        expected = _reference_update(expected, factors, alpha=alpha, strength=strength)
        _install(pipeline, name, factors, alpha=alpha, strength=strength)
        _assert_cpu_packed_weight(wrapped, expected, False)

    records = wrapped._merged_loras
    assert len(records) == 4
    assert records[0].lora_A.dtype == torch.bfloat16
    assert records[2].lora_A.dtype == records[2].lora_B.dtype == torch.float32
    assert records[2].scale == (3.5 / 2) * -0.75
    assert all(record.lora_A.device.type == record.lora_B.device.type == "cpu" for record in records)
    with pytest.raises(FrozenInstanceError):
        records[0].scale = 0.0
    torch.testing.assert_close(wrapped.cpu_weight, original, rtol=0, atol=0)

    _install(pipeline, "replacement", third, alpha=3.5, strength=-0.75, accumulate=False)
    _assert_cpu_packed_weight(wrapped, _reference_update(original, third, alpha=3.5, strength=-0.75), False)
    assert len(wrapped._merged_loras) == 1


def test_direct_accumulating_merge_after_each_purge_keeps_previous_updates(cpu_nvfp4_lifecycle):
    _, wrapped, original, first, _ = _nvfp4_lifecycle_pipeline(False)
    expected = _reference_update(original, first)

    for _ in range(2):
        wrapped.merge_lora_weights(accumulate=True)
        expected = _reference_update(expected, first)
        torch.testing.assert_close(wrapped.base_layer.weight, expected, rtol=0, atol=0)
        _cpu_pack_nvfp4(wrapped)
        _assert_cpu_packed_weight(wrapped, expected, False)

    assert len(wrapped._merged_loras) == 3


def test_merge_after_clear_preserves_purged_state(cpu_nvfp4_lifecycle):
    pipeline, wrapped, original, _, _ = _nvfp4_lifecycle_pipeline(False)
    LoRAPipeline.clear_lora_adapter(pipeline)
    assert wrapped.base_layer.weight is None
    previous_flags = (wrapped.merged, wrapped.disable_lora, wrapped.side_branch)
    previous_records = wrapped._merged_loras
    buffers = {name: (value, value.clone()) for name, value in wrapped.base_layer.named_buffers()}

    with pytest.raises(AssertionError, match="LoRA weights not set"):
        LoRAPipeline.merge_lora_weights(pipeline)

    assert wrapped.base_layer.weight is None
    assert (wrapped.merged, wrapped.disable_lora, wrapped.side_branch) == previous_flags
    assert wrapped.lora_A is None and wrapped.lora_B is None
    assert wrapped._merged_loras is previous_records
    torch.testing.assert_close(wrapped.cpu_weight, original, rtol=0, atol=0)
    for name, (previous_buffer, expected) in buffers.items():
        assert getattr(wrapped.base_layer, name) is previous_buffer
        torch.testing.assert_close(getattr(wrapped.base_layer, name), expected, rtol=0, atol=0)


def test_accumulated_round_trip_repacking_matches_dense_lifecycle(cpu_nvfp4_lifecycle):
    pipeline, wrapped, original, first, second = _nvfp4_lifecycle_pipeline(False)
    base = ReplicatedLinear(32, 32, bias=False, params_dtype=torch.bfloat16)
    base.weight.data.copy_(original)
    dense = get_lora_layer(base)
    dense.set_lora_weights(*(factor.clone() for factor in first), lora_alpha=4)
    dense.set_lora_weights(*(factor.clone() for factor in second), lora_alpha=4, strength=0.5, accumulate=True)
    _install(pipeline, "second", second, alpha=4, strength=0.5)
    _assert_cpu_packed_weight(wrapped, dense.base_layer.weight, False)

    for _ in range(2):
        dense.unmerge_lora_weights()
        LoRAPipeline.unmerge_lora_weights(pipeline)
        assert wrapped.merged == dense.merged
        _assert_cpu_packed_weight(wrapped, dense.base_layer.weight, False)

        dense.merge_lora_weights()
        LoRAPipeline.merge_lora_weights(pipeline)
        assert wrapped.merged == dense.merged
        _assert_cpu_packed_weight(wrapped, dense.base_layer.weight, False)


def test_replay_rounds_to_bf16_after_each_contribution(cpu_nvfp4_lifecycle):
    base = ReplicatedLinear(32, 32, bias=False, params_dtype=torch.bfloat16)
    base.weight.data.fill_(256)
    base.quant_method = _CpuPackedNVFP4Method(False)
    wrapped = get_lora_layer(base)
    factors = (torch.ones(1, 32, dtype=torch.bfloat16), torch.ones(32, 1, dtype=torch.bfloat16))

    wrapped.set_lora_weights(*factors, lora_alpha=1)
    _cpu_pack_nvfp4(wrapped)
    wrapped.set_lora_weights(*factors, lora_alpha=1, accumulate=True)
    _cpu_pack_nvfp4(wrapped)
    wrapped.set_lora_weights(*factors, lora_alpha=1, strength=0.0, accumulate=True)

    # 256 + 1 rounds back to 256 at each BF16 merge; combining the two deltas gives 258.
    expected = torch.full((32, 32), 256, dtype=torch.bfloat16)
    torch.testing.assert_close(wrapped.base_layer.weight, expected, rtol=0, atol=0)
    assert not torch.equal(wrapped.base_layer.weight, expected + 2)


def test_cpu_contributions_do_not_alias_adapter_cache_or_follow_scalar_changes(cpu_nvfp4_lifecycle):
    pipeline, wrapped, original, first, second = _nvfp4_lifecycle_pipeline(False)
    applied_first = tuple(t.clone() for t in first)
    records = wrapped._merged_loras
    saved_a, saved_b, saved_scale = records[0].lora_A, records[0].lora_B, records[0].scale
    assert saved_a.data_ptr() != first[0].data_ptr()
    assert saved_b.data_ptr() != first[1].data_ptr()

    first[0].fill_(9)
    first[1].fill_(-7)
    wrapped.lora_strength = 11.0
    wrapped.lora_alpha = 13.0
    _install(pipeline, "second", second, alpha=4, strength=0.5)

    expected = _reference_update(_reference_update(original, applied_first), second, alpha=4, strength=0.5)
    _assert_cpu_packed_weight(wrapped, expected, False)
    torch.testing.assert_close(saved_a, applied_first[0], rtol=0, atol=0)
    torch.testing.assert_close(saved_b, applied_first[1], rtol=0, atol=0)
    assert saved_scale == 1.0


@pytest.mark.parametrize("retain", [False, True])
def test_tensor_restore_preserves_state_until_explicit_reset(cpu_nvfp4_lifecycle, retain: bool):
    _, wrapped, original, _, _ = _nvfp4_lifecycle_pipeline(retain)
    previous_records = wrapped._merged_loras
    previous_a, previous_b = wrapped.lora_A, wrapped.lora_B

    wrapped._restore_base_weight()

    torch.testing.assert_close(wrapped.base_layer.weight, original, rtol=0, atol=0)
    assert wrapped.merged
    assert wrapped._merged_loras is previous_records
    assert wrapped.lora_A is previous_a and wrapped.lora_B is previous_b

    wrapped._reset_to_base_weight()

    assert not wrapped.merged
    if retain:
        assert wrapped._merged_loras is None
    else:
        assert wrapped._merged_loras == [] and wrapped._merged_loras is not previous_records
    torch.testing.assert_close(wrapped.base_layer.weight, original, rtol=0, atol=0)
    torch.testing.assert_close(wrapped.cpu_weight, original, rtol=0, atol=0)
    assert wrapped.lora_A is previous_a and wrapped.lora_B is previous_b


@pytest.mark.parametrize("invalid", ["rank", "width", "alpha", "strength"])
def test_invalid_accumulation_preserves_previous_factors_history_and_packed_weights(cpu_nvfp4_lifecycle, invalid):
    pipeline, wrapped, _, _, second = _nvfp4_lifecycle_pipeline(False)
    _install(pipeline, "second", second, strength=0.5)
    previous_a, previous_b = wrapped.lora_A, wrapped.lora_B
    previous_records = wrapped._merged_loras
    packed = {name: value.clone() for name, value in wrapped.base_layer.named_buffers()}
    a, b = torch.randn(3, 32), torch.randn(32, 3)
    alpha, strength = 3.0, 1.0
    if invalid == "rank":
        b = torch.randn(32, 2)
    elif invalid == "width":
        a = torch.randn(3, 31)
    elif invalid == "alpha":
        alpha = float("nan")
    else:
        strength = float("inf")

    with pytest.raises(ValueError):
        wrapped.set_lora_weights(a, b, lora_alpha=alpha, strength=strength, accumulate=True)

    assert wrapped.base_layer.weight is None and wrapped.merged
    assert wrapped.lora_A is previous_a and wrapped.lora_B is previous_b
    assert wrapped._merged_loras is previous_records
    assert len(previous_records) == 2
    for name, value in packed.items():
        torch.testing.assert_close(getattr(wrapped.base_layer, name), value, rtol=0, atol=0)


def test_failed_replay_restores_previous_state_and_can_be_retried(cpu_nvfp4_lifecycle,
                                                               monkeypatch: pytest.MonkeyPatch):
    pipeline, wrapped, original, first, second = _nvfp4_lifecycle_pipeline(False)
    _install(pipeline, "second", second, strength=0.5)
    third = (torch.randn(2, 32) / 8, torch.randn(32, 2) / 4)
    previous_a, previous_b = wrapped.lora_A, wrapped.lora_B
    previous_records = wrapped._merged_loras
    packed = {name: value.clone() for name, value in wrapped.base_layer.named_buffers()}
    real_merge = wrapped._merged_weight
    replayed = 0

    def fail_on_second_contribution(weight, contribution=None):
        nonlocal replayed
        if contribution is not None:
            replayed += 1
            if replayed == 2:
                raise RuntimeError("injected replay failure")
        return real_merge(weight, contribution)

    with monkeypatch.context() as patch:
        patch.setattr(wrapped, "_merged_weight", fail_on_second_contribution)
        with pytest.raises(RuntimeError, match="injected replay failure"):
            wrapped.set_lora_weights(*third, lora_alpha=2, strength=-0.25, accumulate=True)

    assert replayed == 2
    assert wrapped.base_layer.weight is None and wrapped.merged
    assert wrapped.lora_A is previous_a and wrapped.lora_B is previous_b
    assert wrapped._merged_loras is previous_records
    assert len(previous_records) == 2
    for name, value in packed.items():
        torch.testing.assert_close(getattr(wrapped.base_layer, name), value, rtol=0, atol=0)

    wrapped.set_lora_weights(*third, lora_alpha=2, strength=-0.25, accumulate=True)
    expected = _reference_update(_reference_update(original, first), second, strength=0.5)
    expected = _reference_update(expected, third, alpha=2, strength=-0.25)
    _cpu_pack_nvfp4(wrapped)
    _assert_cpu_packed_weight(wrapped, expected, False)
    assert len(wrapped._merged_loras) == 3


@pytest.mark.parametrize("mode", ["bf16", "mxfp8", "nvfp4_retained", "nvfp4_refine", "training", "side_branch"])
def test_other_paths_do_not_keep_cpu_contribution_snapshots(cpu_nvfp4_lifecycle, mode):
    base = ReplicatedLinear(32, 32, bias=False, params_dtype=torch.bfloat16)
    base.weight.data.fill_(0.125)
    if mode == "mxfp8":
        base.quant_method = object.__new__(MXFP8QuantizeMethod)
    elif mode in ("nvfp4_retained", "nvfp4_refine", "training", "side_branch"):
        base.quant_method = _CpuPackedNVFP4Method(mode == "nvfp4_retained")
        if mode == "nvfp4_refine":
            base.quant_method._is_refine_only_layer = True
        if mode == "side_branch":
            _cpu_pack_nvfp4(base)
    training = mode == "training"
    wrapped = get_lora_layer(base, lora_rank=2, lora_alpha=2, training_mode=training)
    factors = (torch.ones(2, 32, dtype=torch.bfloat16) / 8,
               torch.ones(32, 2, dtype=torch.bfloat16) / 8)
    wrapped.set_lora_weights(*factors, lora_alpha=2, training_mode=training)
    assert wrapped._merged_loras is None
