# SPDX-License-Identifier: Apache-2.0
"""Choose block activation policies from measured costs and live intervals."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class PolicyCost:
    """One block policy's additional time and storage at each execution event.

    Event storage includes retained activations, policy-dependent unsharded
    parameter copies, and temporary recomputation or transfer buffers. Fixed
    storage accounts for master parameters, optimizer state, gradients, and
    allocations outside the planned blocks.
    """

    policy: str
    time_ms: float
    device_bytes: tuple[int, ...]
    host_bytes: tuple[int, ...] = ()


@dataclass(frozen=True)
class BlockCosts:
    role: str
    block: str
    choices: tuple[PolicyCost, ...]


@dataclass(frozen=True)
class MemoryProfile:
    events: tuple[str, ...]
    fixed_device_bytes: tuple[int, ...]
    blocks: tuple[BlockCosts, ...]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> MemoryProfile:
        data = json.loads(Path(path).read_text())
        blocks = tuple(
            BlockCosts(
                item["role"], item["block"],
                tuple(
                    PolicyCost(choice["policy"], choice["time_ms"], tuple(choice["device_bytes"]),
                               tuple(choice.get("host_bytes", ()))) for choice in item["choices"]))
            for item in data["blocks"])
        return cls(tuple(data["events"]), tuple(data["fixed_device_bytes"]), blocks, data.get("metadata", {}))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


@dataclass(frozen=True)
class MemoryPlan:
    policies: dict[str, dict[str, str]]
    device_bytes: tuple[int, ...]
    host_bytes: tuple[int, ...]
    time_ms: float
    budget_bytes: int
    host_budget_bytes: int | None
    events: tuple[str, ...]
    profile_metadata: dict[str, Any]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def plan_activation_memory(
    profile: MemoryProfile,
    budget_bytes: int,
    *,
    host_budget_bytes: int | None = None,
) -> MemoryPlan:
    """Minimize measured policy time under every event's memory budget.

    A binary choice is made for each block. Keeping a separate constraint for
    each event preserves overlap between student and critic graphs, while
    allowing storage reuse between disjoint forward/backward phases.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix

    event_count = len(profile.events)
    if not event_count or len(profile.fixed_device_bytes) != event_count:
        raise ValueError("The profile must provide fixed device storage for every event")
    if budget_bytes <= 0 or (host_budget_bytes is not None and host_budget_bytes < 0):
        raise ValueError("Device budget must be positive and host budget non-negative")
    if any(value < 0 for value in profile.fixed_device_bytes):
        raise ValueError("Fixed storage must be non-negative")
    if max(profile.fixed_device_bytes) > budget_bytes:
        raise ValueError("The memory budget is smaller than the profile's fixed storage")
    if not profile.blocks:
        return MemoryPlan({}, profile.fixed_device_bytes, (0, ) * event_count, 0.0, budget_bytes, host_budget_bytes,
                          profile.events, profile.metadata)

    identifiers = [(block.role, block.block) for block in profile.blocks]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Each role/block must occur once in a memory profile")
    for block in profile.blocks:
        if not block.choices or len({choice.policy for choice in block.choices}) != len(block.choices):
            raise ValueError(f"{block.role}.{block.block} needs distinct policy choices")
        for choice in block.choices:
            if choice.policy not in {"save", "recompute", "offload", "ops"}:
                raise ValueError(f"Unsupported activation policy: {choice.policy}")
            if len(choice.device_bytes) != event_count or (choice.host_bytes and len(choice.host_bytes) != event_count):
                raise ValueError(f"{block.role}.{block.block} must cover every event")
            if not np.isfinite(choice.time_ms) or choice.time_ms < 0:
                raise ValueError("Policy times must be finite and non-negative")
            if any(value < 0 for value in (*choice.device_bytes, *choice.host_bytes)):
                raise ValueError("Policy storage must be non-negative")

    choices = [choice for block in profile.blocks for choice in block.choices]
    # MiB units keep solver feasibility tolerances well below a storage page.
    unit = 1024 * 1024
    device = np.asarray([choice.device_bytes for choice in choices], dtype=float).T / unit
    host = np.asarray([choice.host_bytes or (0, ) * event_count for choice in choices], dtype=float).T / unit
    select = np.zeros((len(profile.blocks), len(choices)))
    offset = 0
    for index, block in enumerate(profile.blocks):
        select[index, offset:offset + len(block.choices)] = 1
        offset += len(block.choices)
    matrices = [select, device]
    lower = [np.ones(len(profile.blocks)), np.full(event_count, -np.inf)]
    upper = [np.ones(len(profile.blocks)), (budget_bytes - np.asarray(profile.fixed_device_bytes)) / unit]
    if host_budget_bytes is not None:
        matrices.append(host)
        lower.append(np.full(event_count, -np.inf))
        upper.append(np.full(event_count, host_budget_bytes / unit))
    result = milp(
        c=np.asarray([choice.time_ms for choice in choices]),
        integrality=np.ones(len(choices)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(csc_matrix(np.vstack(matrices)), np.concatenate(lower), np.concatenate(upper)),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success:
        if result.status == 2:
            raise ValueError("No measured activation policy combination fits the memory budget")
        raise RuntimeError(f"Activation memory planning failed: {result.message}")

    policies: dict[str, dict[str, str]] = {}
    device_bytes = list(profile.fixed_device_bytes)
    host_bytes = [0] * event_count
    time_ms = 0.0
    offset = 0
    for block in profile.blocks:
        choice_index = int(np.argmax(result.x[offset:offset + len(block.choices)]))
        choice = block.choices[choice_index]
        offset += len(block.choices)
        policies.setdefault(block.role, {})[block.block] = choice.policy
        device_bytes = [value + extra for value, extra in zip(device_bytes, choice.device_bytes, strict=True)]
        if choice.host_bytes:
            host_bytes = [value + extra for value, extra in zip(host_bytes, choice.host_bytes, strict=True)]
        time_ms += choice.time_ms
    if max(device_bytes) > budget_bytes or (host_budget_bytes is not None and max(host_bytes) > host_budget_bytes):
        raise RuntimeError("The solver returned a policy outside the integer-byte budget")
    return MemoryPlan(policies, tuple(device_bytes), tuple(host_bytes), time_ms, budget_bytes, host_budget_bytes,
                      profile.events, profile.metadata)


def apply_activation_memory_plan(module: torch.nn.Module, policies: dict[str, str]) -> torch.nn.Module:
    """Apply a role's policies before FSDP, retaining checkpoint state keys."""
    import torch
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import ActivationWrapper, checkpoint_wrapper

    class DeviceOffloadWrapper(ActivationWrapper):

        def forward(self, *args, **kwargs):
            device_type = next(self._checkpoint_wrapped_module.parameters()).device.type
            with torch.autograd.graph.save_on_cpu(pin_memory=True, device_type=device_type):
                return self._checkpoint_wrapped_module(*args, **kwargs)

    # Resolve all paths before changing the model, including save-only entries.
    blocks = {name: module.get_submodule(name) for name in policies}
    for name, policy in policies.items():
        if policy not in {"save", "recompute", "offload", "ops"}:
            raise ValueError(f"Unsupported activation policy: {policy}")
        if isinstance(blocks[name], ActivationWrapper):
            raise ValueError(f"{name} already has an activation policy")
    for name, policy in policies.items():
        block = blocks[name]
        if policy == "save":
            continue
        if policy == "offload":
            wrapped = DeviceOffloadWrapper(block)
        else:
            kwargs = {}
            if policy == "ops":
                from fastvideo.train.utils.activation_checkpoint import _selective_checkpointing_context_fn

                kwargs["context_fn"] = _selective_checkpointing_context_fn
            wrapped = checkpoint_wrapper(block, preserve_rng_state=True, **kwargs)
        parent_path, _, child = name.rpartition(".")
        module.get_submodule(parent_path).register_module(child, wrapped)
    return module
