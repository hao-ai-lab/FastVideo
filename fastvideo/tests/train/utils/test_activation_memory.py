# SPDX-License-Identifier: Apache-2.0
"""CPU solver and training-state checks, independent of CUDA model imports."""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils import activation_memory as memory
from fastvideo.train.utils import builder
from fastvideo.train.utils import training_config as config
from fastvideo.train.utils.config import RunConfig


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[4] / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MIB = 1024 * 1024


def choice(policy, time, *sizes, host=()):
    return memory.PolicyCost(policy, time, tuple(size * MIB for size in sizes), tuple(size * MIB for size in host))


def test_event_constraints_match_exhaustive_optimum_and_reuse_role_storage():
    profile = memory.MemoryProfile(
        ("student_forward", "student_backward", "critic_forward", "critic_backward"),
        (MIB,) * 4,
        (
            memory.BlockCosts("student", "blocks.0", (
                choice("save", 0, 8, 8, 0, 0), choice("recompute", 3, 1, 3, 0, 0),
                choice("offload", 2, 1, 4, 0, 0, host=(8, 8, 0, 0)),
            )),
            memory.BlockCosts("student", "blocks.1", (
                choice("save", 0, 7, 0, 0, 0), choice("recompute", 1, 1, 0, 0, 0),
            )),
            memory.BlockCosts("critic", "blocks.0", (
                choice("save", 0, 0, 0, 9, 9), choice("recompute", 4, 0, 0, 1, 4),
            )),
        ), {},
    )
    for budget in (6, 10, 17):
        plan = memory.plan_activation_memory(profile, budget * MIB)
        feasible = []
        for selected in itertools.product(*(block.choices for block in profile.blocks)):
            peak = max(fixed + sum(cost.device_bytes[event] for cost in selected)
                       for event, fixed in enumerate(profile.fixed_device_bytes))
            if peak <= budget * MIB:
                feasible.append(sum(cost.time_ms for cost in selected))
        assert plan.time_ms == min(feasible)
        assert max(plan.device_bytes) <= budget * MIB
    assert memory.plan_activation_memory(profile, 10 * MIB).policies["critic"]["blocks.0"] == "save"


def test_host_budget_and_infeasible_budget():
    profile = memory.MemoryProfile(("forward", "backward"), (MIB, MIB), (
        memory.BlockCosts("student", "blocks.0", (
            choice("save", 0, 9, 9), choice("recompute", 3, 2, 3),
            choice("offload", 1, 1, 2, host=(9, 9)),
        )),
    ), {})
    assert memory.plan_activation_memory(profile, 5 * MIB).policies["student"]["blocks.0"] == "offload"
    assert memory.plan_activation_memory(profile, 5 * MIB, host_budget_bytes=0).policies["student"]["blocks.0"] == "recompute"
    with pytest.raises(ValueError, match="No measured"):
        memory.plan_activation_memory(profile, 2 * MIB)


@pytest.mark.parametrize("host_budget", [None, 0, 8 * MIB])
@pytest.mark.parametrize("with_blocks", [False, True])
def test_profile_and_plan_json_round_trip(tmp_path, host_budget, with_blocks):
    profile = memory.MemoryProfile(("backward",), (MIB,), (
        memory.BlockCosts("student", "blocks.0", (choice("save", 0, 3),)),
    ) if with_blocks else (), {"shape": [1, 16, 64]})
    path = tmp_path / "profile.json"
    profile.save(path)
    assert memory.MemoryProfile.load(path) == profile
    plan = memory.plan_activation_memory(profile, 4 * MIB, host_budget_bytes=host_budget)
    plan.save(tmp_path / "plan.json")
    saved = json.loads((tmp_path / "plan.json").read_text())
    assert saved["budget_bytes"] == 4 * MIB
    assert saved["host_budget_bytes"] == host_budget
    assert saved == json.loads(json.dumps(asdict(plan)))
    saved["events"] = tuple(saved["events"])
    saved["device_bytes"] = tuple(saved["device_bytes"])
    saved["host_bytes"] = tuple(saved["host_bytes"])
    assert memory.MemoryPlan(**saved) == plan


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Dropout(0.2),
                                torch.nn.Linear(16, 8)) for _ in range(3)
        ])

    def forward(self, value):
        for block in self.blocks:
            value = value + block(value)
        return value


def update(model, optimizer, value, seed):
    torch.manual_seed(seed)
    optimizer.zero_grad()
    output = model(value)
    loss = output.square().mean()
    loss.backward()
    grads = [parameter.grad.clone() for parameter in model.parameters()]
    optimizer.step()
    return output.detach(), grads


@pytest.mark.parametrize("policies", [
    {"blocks.0": "save", "blocks.1": "recompute", "blocks.2": "offload"},
    {f"blocks.{index}": "recompute" for index in range(3)},
    {f"blocks.{index}": "offload" for index in range(3)},
    {f"blocks.{index}": "ops" for index in range(3)},
])
def test_mixed_policies_preserve_gradients_adamw_and_resume(policies):
    torch.manual_seed(23)
    baseline = Transformer().double()
    planned = copy.deepcopy(baseline)
    memory.apply_activation_memory_plan(planned, policies)
    optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
    planned_optimizer = torch.optim.AdamW(planned.parameters(), lr=1e-3)
    value = torch.randn(2, 4, 8, dtype=torch.float64)
    for step in range(2):
        ref_output, ref_grads = update(baseline, optimizer, value, step)
        output, grads = update(planned, planned_optimizer, value, step)
        torch.testing.assert_close(output, ref_output, atol=0, rtol=0)
        for actual, expected in zip(grads, ref_grads, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert planned.state_dict().keys() == baseline.state_dict().keys()
        for key, actual in planned.state_dict().items():
            torch.testing.assert_close(actual, baseline.state_dict()[key], atol=0, rtol=0)
    resumed = Transformer().double()
    memory.apply_activation_memory_plan(resumed, {f"blocks.{i}": "recompute" for i in range(3)})
    resumed.load_state_dict(planned.state_dict())
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(copy.deepcopy(planned_optimizer.state_dict()))
    update(baseline, optimizer, value, 3)
    update(resumed, resumed_optimizer, value, 3)
    for actual, expected in zip(resumed.parameters(), baseline.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_configuration_requires_budget_profile_pair_and_no_manual_policy():
    config.ModelTrainingConfig(activation_memory_budget_gib=32, activation_memory_profile="measured.json")
    with pytest.raises(ValueError, match="both"):
        config.ModelTrainingConfig(activation_memory_budget_gib=32)
    with pytest.raises(ValueError, match="manual"):
        config.ModelTrainingConfig(activation_memory_budget_gib=32, activation_memory_profile="measured.json",
                                   enable_gradient_checkpointing_type="full")


def test_yaml_budget_fields_and_cli_override_reach_typed_config(tmp_path):
    from fastvideo.train.utils.config import load_run_config

    path = tmp_path / "training.yaml"
    path.write_text("""models:
  student:
    _target_: fastvideo.train.models.wan.WanModel
    init_from: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
method:
  _target_: fastvideo.train.methods.distribution_matching.dmd2.DMD2Method
training:
  model:
    activation_memory_budget_gib: 24
    activation_memory_host_budget_gib: 0
    activation_memory_profile: measured.json
    activation_memory_plan_output: selected.json
""")
    loaded = load_run_config(str(path), ["--training.model.activation_memory_budget_gib", "25.5"])
    assert loaded.training.model.activation_memory_budget_gib == 25.5
    assert loaded.training.model.activation_memory_host_budget_gib == 0
    assert loaded.training.model.activation_memory_profile == "measured.json"
    assert loaded.training.model.activation_memory_plan_output == "selected.json"


def test_unknown_block_does_not_partially_wrap_model():
    model = Transformer()
    first = model.blocks[0]
    with pytest.raises(AttributeError):
        memory.apply_activation_memory_plan(model, {"blocks.0": "recompute", "blocks.8": "save"})
    assert model.blocks[0] is first


class PlannedAdapter(ModelBase):
    def __init__(self, *, training_config, activation_memory_plan=None):
        self.plan = activation_memory_plan

    def prepare_batch(self, *args, **kwargs):
        raise NotImplementedError

    def add_noise(self, *args, **kwargs):
        raise NotImplementedError

    def predict_noise(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError


class UnsupportedAdapter(PlannedAdapter):
    def __init__(self, *, training_config):
        super().__init__(training_config=training_config)


class KwargsAdapter(UnsupportedAdapter):
    def __init__(self, *, training_config, **kwargs):
        super().__init__(training_config=training_config)


class RoleMethod:
    def __init__(self, *, cfg, role_models):
        self.role_models = role_models


@pytest.fixture
def budget_config(tmp_path):
    profile = memory.MemoryProfile(("backward",), (MIB,), tuple(
        memory.BlockCosts(role, "blocks.0", (choice("save", 0, 3),))
        for role in ("student", "critic")
    ), {})
    profile.save(tmp_path / "profile.json")
    return RunConfig(
        models={
            "student": {"_target_": f"{__name__}.PlannedAdapter"},
            "critic": {"_target_": f"{__name__}.PlannedAdapter"},
            "teacher": {"_target_": f"{__name__}.UnsupportedAdapter"},
        },
        method={"_target_": f"{__name__}.RoleMethod"},
        training=config.TrainingConfig(model=config.ModelTrainingConfig(
            activation_memory_budget_gib=8 / 1024,
            activation_memory_host_budget_gib=0,
            activation_memory_profile=str(tmp_path / "profile.json"),
            activation_memory_plan_output=str(tmp_path / "plan.json"),
        )),
        callbacks={},
        raw={},
    )


@pytest.mark.parametrize("target", [
    f"{__name__}.UnsupportedAdapter",
    f"{__name__}.KwargsAdapter",
    "fastvideo.train.models.wan.WanCausalModel",
])
def test_builder_rejects_unsupported_role_before_export_or_model_loading(budget_config, tmp_path, monkeypatch, target):
    budget_config.models["critic"]["_target_"] = target
    instantiate = Mock(wraps=builder.instantiate)
    monkeypatch.setattr(builder, "instantiate", instantiate)
    with pytest.raises(ValueError, match=r"models\.critic\._target_.*activation_memory_plan"):
        builder.build_from_config(budget_config)
    instantiate.assert_not_called()
    assert not (tmp_path / "plan.json").exists()


def test_builder_delivers_exported_policies_to_each_planned_role(budget_config, tmp_path):
    _, method, _, _ = builder.build_from_config(budget_config)
    saved = json.loads((tmp_path / "plan.json").read_text())
    assert saved["host_budget_bytes"] == 0
    assert saved["policies"] == {"student": {"blocks.0": "save"}, "critic": {"blocks.0": "save"}}
    for role in ("student", "critic"):
        assert method.role_models[role].plan == saved["policies"][role]
    assert method.role_models["teacher"].plan is None


def test_builder_without_budget_preserves_existing_adapter_support(budget_config, tmp_path):
    budget_config.training.model = config.ModelTrainingConfig()
    budget_config.models["student"]["_target_"] = f"{__name__}.UnsupportedAdapter"
    _, method, _, _ = builder.build_from_config(budget_config)
    assert method.role_models["student"].plan is None
    assert not (tmp_path / "plan.json").exists()


def _gather_measurements_worker(rank, world_size, rendezvous, output_dir):
    import torch.distributed as dist

    measurements = load_source("activation_memory_measurements", "examples/train/activation_memory/measurements.py")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        records = [{
            "iteration": step,
            "step_ms": 10 + rank + step,
            "peak_allocated_bytes": (rank + 1) * MIB + step,
            "peak_reserved_bytes": (3 - rank) * MIB + step,
            "losses": {"student": rank + 0.5},
        } for step in range(2)]
        steps, rank_records = measurements.gather_step_records(records)
        Path(output_dir, f"rank{rank}.json").write_text(json.dumps({"steps": steps, "ranks": rank_records}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
def test_dmd_measurements_include_each_rank_and_step_maxima(tmp_path, world_size):
    torch.multiprocessing.spawn(
        _gather_measurements_worker,
        args=(world_size, f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    for rank in range(world_size):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert len(result["ranks"]) == world_size
        for index, steps in enumerate(result["ranks"]):
            assert steps[0]["peak_allocated_bytes"] == (index + 1) * MIB
            assert steps[0]["losses"] == {"student": index + 0.5}
        assert result["steps"] == [{
            "iteration": step,
            "step_ms": 10 + world_size - 1 + step,
            "peak_allocated_bytes": world_size * MIB + step,
            "peak_reserved_bytes": 3 * MIB + step,
            "losses": {"student": 0.5},
        } for step in range(2)]
