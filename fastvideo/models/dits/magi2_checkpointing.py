# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint loading for MAGI-2.

Weights ship as a single HuggingFace-style safetensors directory holding the
full, unsharded expert stack. Every expert-parallel rank reads that same
directory but pulls only the slice of each MoE tensor it owns, so one copy on
disk serves any ``ep_size``.
"""

import json
import os
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from safetensors import safe_open
from tqdm.auto import tqdm

from fastvideo.models.dits.magi2_runtime import psm

_INDEX_NAME = "model.safetensors.index.json"
_SINGLE_NAME = "model.safetensors"

# MoE tensors are concatenated over (head, expert) along dim 0 and split across
# expert-parallel ranks; everything else is replicated.
_EP_SHARDED_SUFFIXES = (
    "moe_mlp.gate",
    "moe_mlp.W_gate",
    "moe_mlp.W_up",
    "moe_mlp.W_down",
    "moe_mlp.router.expert_bias",
    "moe_mlp.router.expert_bias_ema",
)


def print_rank_0(message: str) -> None:
    """Print one checkpoint message from global rank zero."""
    if psm.get_global_rank() == 0:
        print(message)


def read_safetensors_weight_map(checkpoint_dir: str) -> dict[str, str]:
    """Map every tensor name in a HuggingFace-style directory to its file."""
    index_path = os.path.join(checkpoint_dir, _INDEX_NAME)
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return {key: os.path.join(checkpoint_dir, shard) for key, shard in weight_map.items()}

    single_path = os.path.join(checkpoint_dir, _SINGLE_NAME)
    if not os.path.exists(single_path):
        raise FileNotFoundError(f"No {_INDEX_NAME} or {_SINGLE_NAME} under {checkpoint_dir}")
    with safe_open(single_path, framework="pt") as f:
        return {key: single_path for key in f}


def load_safetensors_dir(checkpoint_dir: str, desc: str = "Loading shards") -> dict[str, torch.Tensor]:
    """Load every tensor from a HuggingFace-style safetensors directory."""
    return _load_by_shard(read_safetensors_weight_map(checkpoint_dir), reader=_read_whole, desc=desc)


def _is_ep_sharded_key(key: str) -> bool:
    return ".moe_mlp." in key and key.endswith(_EP_SHARDED_SUFFIXES)


def _read_whole(handle: Any, key: str, _target: torch.Tensor | None) -> torch.Tensor:
    return handle.get_tensor(key)


def _read_ep_slice(handle: Any, key: str, target: torch.Tensor | None) -> torch.Tensor:
    """Read only the (head, expert) rows this expert-parallel rank owns.

    ``num_heads`` is padded up to a multiple of ``ep_size``, so the trailing
    ranks may fall wholly or partly past the real experts and get zeros.
    """
    if target is None or not _is_ep_sharded_key(key):
        return _read_whole(handle, key, target)

    ep_size = psm.get_world_size("ep")
    view = handle.get_slice(key)
    real_rows = view.get_shape()[0]
    shard_rows = target.shape[0]
    if ep_size <= 1 or shard_rows == real_rows:
        return handle.get_tensor(key)

    start = psm.get_local_rank("ep") * shard_rows
    if start >= real_rows:
        return torch.zeros_like(target, device="cpu")

    chunk = view[start : min(start + shard_rows, real_rows)]
    if chunk.shape[0] == shard_rows:
        return chunk
    padding = torch.zeros((shard_rows - chunk.shape[0], *chunk.shape[1:]), dtype=chunk.dtype)
    return torch.cat([chunk, padding], dim=0)


def _load_by_shard(
    weight_map: Mapping[str, str],
    reader,
    desc: str,
    targets: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Read tensors grouped by file so each shard is opened exactly once."""
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, path in weight_map.items():
        by_shard[path].append(key)

    def read_shard(item: tuple[str, list[str]]) -> dict[str, torch.Tensor]:
        path, keys = item
        with safe_open(path, framework="pt") as handle:
            return {key: reader(handle, key, None if targets is None else targets.get(key)) for key in keys}

    state_dict: dict[str, torch.Tensor] = {}
    with ThreadPoolExecutor() as executor:
        futures = list(executor.map(read_shard, by_shard.items()))
        for shard_state in tqdm(futures, desc=desc, total=len(futures), disable=psm.get_global_rank() != 0):
            state_dict.update(shard_state)
    return state_dict


def _apply_router_bias_ema(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pair the router bias with exponential moving average (EMA) model weights.

    The checkpoint stores both ``expert_bias`` (unaveraged training buffer) and
    ``expert_bias_ema`` (EMA-smoothed buffer). Inference uses the EMA model
    weights, so the router must use the paired EMA bias; otherwise routing
    biases mismatch the weights and quality degrades. Set the env var
    ``MAGI2_ROUTER_BIAS_SOURCE=main`` to keep the raw ``expert_bias`` instead.
    """
    if (os.environ.get("MAGI2_ROUTER_BIAS_SOURCE") or "ema").strip().lower() == "main":
        print_rank_0("[magi2] Router bias source: expert_bias (raw, MAGI2_ROUTER_BIAS_SOURCE=main)")
        return state_dict

    copied = 0
    for key, value in list(state_dict.items()):
        if not key.endswith("moe_mlp.router.expert_bias_ema"):
            continue
        target_key = key[: -len("_ema")]
        target = state_dict.get(target_key)
        if target is None or tuple(target.shape) != tuple(value.shape):
            continue
        state_dict[target_key] = value.clone()
        copied += 1

    print_rank_0(f"[magi2] Router bias: expert_bias_ema -> expert_bias ({copied} tensor(s))")
    return state_dict


def load_magi2_model_state_dict(model: torch.nn.Module, checkpoint_dir: str) -> Mapping[str, Any]:
    """Load the rank-local preview state and select EMA router biases."""
    if not checkpoint_dir:
        raise ValueError("checkpoint_dir must point at a safetensors checkpoint directory")

    print_rank_0(f"[magi2] Loading preview weights from {checkpoint_dir}")
    targets = model.state_dict()
    weight_map = {key: path for key, path in read_safetensors_weight_map(checkpoint_dir).items() if key in targets}

    state_dict = _load_by_shard(weight_map, reader=_read_ep_slice, desc="Loading shards", targets=targets)
    for key, value in state_dict.items():
        target = targets[key]
        if value.dtype != target.dtype and value.is_floating_point() and target.is_floating_point():
            state_dict[key] = value.to(dtype=target.dtype)
    return _apply_router_bias_ema(state_dict)
