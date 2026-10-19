# SPDX-License-Identifier: Apache-2.0
"""CPU ownership checks for FSDP precision groups; no distributed numerical claim."""
import pytest
import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy

from fastvideo.models.loader import fsdp_load
from fastvideo.models.loader.fsdp_load import _mixed_precision_module_groups


class _MixedModel(nn.Module):
    _keep_in_fp32_modules = frozenset({"boundary"})

    def __init__(self):
        super().__init__()
        self.boundary = nn.Linear(2, 2)
        self.block = nn.Linear(2, 2)
        self.uncovered = nn.Parameter(torch.zeros(1))

    def _get_parameter_dtype(self, name, default):
        if name.startswith("boundary."):
            return torch.float32
        if name == "uncovered":
            return torch.float16
        return default


def test_declared_fp32_child_covers_only_its_parameters():
    model = _MixedModel()
    groups, uncovered = _mixed_precision_module_groups(model, torch.bfloat16)
    assert groups == [("boundary", model.boundary)]
    assert uncovered == {model.uncovered}
    assert not set(model.block.parameters()).intersection(uncovered)


def test_checkpoint_wrapping_preserves_canonical_precision_names():
    model = _MixedModel()
    original = model.boundary
    wrapper = nn.Module()
    wrapper.add_module("_checkpoint_wrapped_module", original)
    model.boundary = wrapper
    groups, uncovered = _mixed_precision_module_groups(model, torch.bfloat16)
    assert groups == [("boundary", wrapper)]
    assert uncovered == {model.uncovered}


def test_shard_calls_keep_fp32_child_separate_from_default_parent(monkeypatch, env_overrides):
    from fastvideo import envs
    env_overrides.enter_context(envs.FASTVIDEO_FSDP2_AUTOWRAP.override(False))
    model = _MixedModel()
    calls = []
    monkeypatch.setattr(fsdp_load, "fully_shard", lambda module, **kwargs: calls.append((module, kwargs)))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                  cast_forward_inputs=False)
    fsdp_load.shard_model(model, cpu_offload=False, mp_policy=policy,
                         fsdp_shard_conditions=[lambda name, _: name == "block"])
    assert [module for module, _ in calls] == [model.block, model.boundary, model]
    assert calls[0][1]["mp_policy"] is policy
    assert calls[1][1]["mp_policy"].param_dtype == torch.float32
    assert calls[1][1]["mp_policy"].reduce_dtype == torch.float32
    assert calls[1][1]["mp_policy"].cast_forward_inputs is False
    assert calls[2][1]["ignored_params"] == {model.uncovered}
    assert not set(model.boundary.parameters()).intersection(calls[2][1]["ignored_params"])


def test_parent_shard_boundary_cannot_capture_declared_child(monkeypatch, env_overrides):
    from fastvideo import envs
    env_overrides.enter_context(envs.FASTVIDEO_FSDP2_AUTOWRAP.override(False))
    model = _MixedModel()
    calls = []
    monkeypatch.setattr(fsdp_load, "fully_shard", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(ValueError, match="contains a declared FP32 compute group"):
        fsdp_load.shard_model(model, cpu_offload=False,
                             mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
                             fsdp_shard_conditions=[lambda name, _: name == ""])
    assert not calls


def test_real_h3_declared_groups_are_disjoint_from_block_shards():
    from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3Transformer3DModel
    arch = MiniMaxH3ArchConfig(num_layers=1, num_refiner_layers=1, num_attention_heads=2,
                              attention_head_dim=16, hidden_size=32, ffn_dim=64, freq_dim=16,
                              time_embed_hidden_dim=32, time_embed_dim=16, rope_freq_dim=2)
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel(config=MiniMaxH3Config(arch_config=arch), hf_config={})
    groups, uncovered = _mixed_precision_module_groups(model, torch.bfloat16)
    assert {name for name, _ in groups} == {"proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out"}
    assert not uncovered
    grouped = {parameter for _, module in groups for parameter in module.parameters()}
    shards = [(name, module) for name, module in model.named_modules()
              if any(condition(name, module) for condition in model._fsdp_shard_conditions)]
    assert {name for name, _ in shards} == {"transformer_blocks.0", "token_refiner.refiner_blocks.0"}
    assert all(not grouped.intersection(module.parameters()) for _, module in shards)
    # The remaining root-owned tensors use the default compute dtype.
    for name, parameter in model.named_parameters():
        if parameter not in grouped:
            assert model._get_parameter_dtype(name, torch.bfloat16) == torch.bfloat16
