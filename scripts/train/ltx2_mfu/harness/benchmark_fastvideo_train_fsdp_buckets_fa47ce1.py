#!/usr/bin/env python3
"""Scratch-only LTX-2 FSDP2 communication-bucket benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard


BucketMode = int | str


def _chunks(items: Sequence[nn.Module], size: int) -> list[list[nn.Module]]:
    if size <= 0:
        raise ValueError("bucket size must be positive")
    return [list(items[start:start + size]) for start in range(0, len(items), size)]


def _unique_parameters(module: nn.Module) -> list[nn.Parameter]:
    return list({id(parameter): parameter for parameter in module.parameters()}.values())


def _numel(parameters: Sequence[nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _install_bucket_policy(mode: BucketMode) -> None:
    import fastvideo.models.loader.fsdp_load as fsdp_load

    original_shard_model = fsdp_load.shard_model

    def _shard_model(
        model: nn.Module,
        *,
        cpu_offload: bool,
        reshard_after_forward: bool = True,
        mp_policy: MixedPrecisionPolicy | None = MixedPrecisionPolicy(),
        mesh: Any = None,
        fsdp_shard_conditions: list[Any] = [],  # noqa: B006
        pin_cpu_memory: bool = True,
    ) -> None:
        if mode == 1:
            original_shard_model(
                model,
                cpu_offload=cpu_offload,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
                mesh=mesh,
                fsdp_shard_conditions=fsdp_shard_conditions,
                pin_cpu_memory=pin_cpu_memory,
            )
            if _rank_zero():
                print(
                    "BF16_BUCKET_PLAN " + json.dumps({
                        "mode": "current",
                        "block_count": 48,
                        "blocks_per_group": 1,
                        "communication_group_count": 49,
                        "public_fully_shard_list_api": False,
                    }, sort_keys=True),
                    flush=True,
                )
            return

        if os.environ.get("FASTVIDEO_FSDP2_AUTOWRAP", "0") == "1":
            raise RuntimeError("bucket benchmark is incompatible with FASTVIDEO_FSDP2_AUTOWRAP=1")
        if mp_policy is None:
            raise RuntimeError("bucket benchmark requires an explicit FSDP2 mixed-precision policy")
        if not fsdp_shard_conditions:
            raise RuntimeError("bucket benchmark requires the model's FSDP shard condition")

        named_modules = list(model.named_modules())
        matched = [
            (name, module)
            for name, module in named_modules
            if any(condition(name, module) for condition in fsdp_shard_conditions)
        ]
        expected_names = [f"model.transformer_blocks.{index}" for index in range(48)]
        names = [name for name, _ in matched]
        if names != expected_names:
            raise RuntimeError(f"expected the 48 ordered LTX-2 blocks, got {names}")
        blocks = [module for _, module in matched]

        all_parameters = _unique_parameters(model)
        block_parameter_ids: set[int] = set()
        block_numels: list[int] = []
        for block in blocks:
            parameters = _unique_parameters(block)
            parameter_ids = {id(parameter) for parameter in parameters}
            overlap = block_parameter_ids.intersection(parameter_ids)
            if overlap:
                raise RuntimeError(f"LTX-2 block parameters overlap across buckets: {len(overlap)}")
            block_parameter_ids.update(parameter_ids)
            block_numels.append(_numel(parameters))
        root_parameters = [parameter for parameter in all_parameters if id(parameter) not in block_parameter_ids]
        if not root_parameters or len(block_parameter_ids) + len(root_parameters) != len(all_parameters):
            raise RuntimeError("block and root buckets do not cover model parameters exactly once")

        default_param_dtype = getattr(mp_policy, "param_dtype", None)
        dtype_selector = getattr(model, "_get_parameter_dtype", None)
        ignored_params = set()
        if callable(dtype_selector) and default_param_dtype is not None:
            ignored_params = {
                parameter
                for name, parameter in model.named_parameters()
                if dtype_selector(name, default_param_dtype) != default_param_dtype
            }
        if ignored_params:
            raise RuntimeError("scratch LTX-2 bucket benchmark does not support ignored mixed-dtype parameters")

        fsdp_kwargs: dict[str, Any] = {
            "reshard_after_forward": reshard_after_forward,
            "mesh": mesh,
            "mp_policy": mp_policy,
        }
        if cpu_offload:
            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=pin_cpu_memory)

        if mode == "root":
            block_groups: list[list[nn.Module]] = []
            grouped_numels = [_numel(all_parameters)]
            fully_shard(model, **fsdp_kwargs)
            communication_group_count = 1
        else:
            if mode not in (2, 4):
                raise ValueError(f"unsupported bucket mode: {mode!r}")
            block_groups = _chunks(blocks, mode)
            grouped_numels = [
                sum(block_numels[index:index + mode])
                for index in range(0, len(block_numels), mode)
            ]
            for group in block_groups:
                # Public PyTorch FSDP2 supports a module list as one
                # communication group while retaining hooks on each module.
                fully_shard(group, **fsdp_kwargs)
            fully_shard(model, **fsdp_kwargs)
            communication_group_count = len(block_groups) + 1

        if _rank_zero():
            print(
                "BF16_BUCKET_PLAN " + json.dumps({
                    "mode": mode,
                    "block_count": len(blocks),
                    "blocks_per_group": None if mode == "root" else mode,
                    "block_communication_groups": len(block_groups),
                    "communication_group_count": communication_group_count,
                    "decorated_block_modules": 0 if mode == "root" else len(blocks),
                    "public_fully_shard_list_api": mode != "root",
                    "total_parameter_numel": _numel(all_parameters),
                    "root_straggler_numel": _numel(root_parameters),
                    "grouped_block_numel_min": min(grouped_numels),
                    "grouped_block_numel_max": max(grouped_numels),
                    "original_parameter_dtypes": sorted({str(parameter.dtype) for parameter in all_parameters}),
                    "working_parameter_dtype": str(mp_policy.param_dtype),
                    "reduction_dtype": str(mp_policy.reduce_dtype),
                    "initial_reshard_after_forward": reshard_after_forward,
                    "cpu_offload": cpu_offload,
                }, sort_keys=True),
                flush=True,
            )

    fsdp_load.shard_model = _shard_model


def _self_test() -> None:
    modules = [nn.Identity() for _ in range(8)]
    assert [len(group) for group in _chunks(modules, 1)] == [1] * 8
    assert [len(group) for group in _chunks(modules, 2)] == [2] * 4
    assert [len(group) for group in _chunks(modules, 4)] == [4] * 2
    print("FSDP_BUCKET_SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fsdp-blocks-per-group", choices=("1", "2", "4", "root"), required=False)
    parser.add_argument("--bucket-self-test", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.bucket_self_test:
        _self_test()
        return
    if args.fsdp_blocks_per_group is None:
        parser.error("--fsdp-blocks-per-group is required")

    mode: BucketMode = args.fsdp_blocks_per_group if args.fsdp_blocks_per_group == "root" else int(
        args.fsdp_blocks_per_group)
    sys.argv = [sys.argv[0], *remaining]
    _install_bucket_policy(mode)

    import benchmark_fastvideo_train_pack_d016 as benchmark

    benchmark.main()


if __name__ == "__main__":
    main()
