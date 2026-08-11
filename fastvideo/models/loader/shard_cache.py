# SPDX-License-Identifier: Apache-2.0
"""Per-rank sharded base-weight cache for fast training relaunches.

After a full checkpoint load, each shard rank persists its local DTensor
chunks (post-rename, post-cast — exactly what ``assign=True`` installed) as
one safetensors file in a cache directory (typically tmpfs). Subsequent
launches with an identical (checkpoint, mesh layout, dtype, name-mapping)
tuple rebuild the model from those chunks via ``DTensor.from_local`` —
skipping the full-tensor reads, per-rank H2D of the whole checkpoint, and
the distribute/scatter step.

Opt-in via ``FASTVIDEO_WEIGHT_SHARD_CACHE=<dir>`` (e.g. ``/dev/shm/fv-wcache``).
Any validation failure or exception degrades to the normal full load — the
cache can never fail a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from fastvideo.logger import init_logger

logger = init_logger(__name__)

_FORMAT_VERSION = 1
_ENV_DIR = "FASTVIDEO_WEIGHT_SHARD_CACHE"
_ENV_MAX_GB = "FASTVIDEO_WEIGHT_SHARD_CACHE_MAX_GB"
# Set to "1" when the cache dir is node-local (tmpfs) on a multi-node run so
# every replicate rank writes its own node's copy.
_ENV_PER_NODE = "FASTVIDEO_WEIGHT_SHARD_CACHE_PER_NODE"
# Mirrors fsdp_load's zero-init allowance for params absent from checkpoints.
_ALLOWED_NEW_PARAM_PATTERNS = ("gate_compress", "proj_l")
_WRITE_MARGIN_BYTES = 5 << 30


@dataclass
class ShardCacheContext:
    entry_dir: Path
    key: str
    shard_index: int
    num_shards: int
    is_writer: bool  # replicate-coordinate 0 writes; replicas are identical


def _expand_weight_files(weight_dir_list: list[str]) -> list[str]:
    files: list[str] = []
    for entry in weight_dir_list:
        if os.path.isdir(entry):
            files.extend(str(p) for p in sorted(Path(entry).glob("*.safetensors")))
        else:
            files.append(entry)
    return files


def _shard_file(entry_dir: Path, shard_index: int, num_shards: int) -> Path:
    return entry_dir / f"shard{shard_index}-of-{num_shards}.safetensors"


def shard_cache_context(
    *,
    weight_dir_list: list[str],
    device_mesh: Any,
    hsdp_replicate_dim: int,
    hsdp_shard_dim: int,
    default_dtype: torch.dtype,
    param_dtype: torch.dtype,
    param_names_mapping: dict[str, str] | None,
) -> ShardCacheContext | None:
    root = os.environ.get(_ENV_DIR)
    if not root:
        return None
    try:
        files = _expand_weight_files(weight_dir_list)
        if not files:
            return None
        stats = sorted((os.path.basename(p), os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in files)
        mapping_items = sorted((param_names_mapping or {}).items())
        key_material = json.dumps(
            [
                _FORMAT_VERSION,
                stats,
                int(hsdp_replicate_dim),
                int(hsdp_shard_dim),
                str(default_dtype),
                str(param_dtype),
                mapping_items,
            ],
            sort_keys=True,
        )
        key = hashlib.sha256(key_material.encode()).hexdigest()[:16]
        coordinate = device_mesh.get_coordinate()
        if coordinate is None:
            return None
        # mesh dims are ("replicate", "shard")
        replicate_index, shard_index = int(coordinate[0]), int(coordinate[1])
        # With a shared cache dir, replicate-coordinate 0 writes and replicas
        # (identical shards) skip to avoid same-file collisions. With a
        # node-local dir (e.g. /dev/shm on a multi-node HSDP run), every
        # replica must write its own node's copy — same-name collisions are
        # impossible across nodes, and a single-writer rule would leave every
        # non-zero replica's node permanently cold (all-reduce MIN then turns
        # that into a global miss).
        per_node_root = os.environ.get(_ENV_PER_NODE, "0") == "1"
        return ShardCacheContext(
            entry_dir=Path(root) / key,
            key=key,
            shard_index=shard_index,
            num_shards=int(hsdp_shard_dim),
            is_writer=per_node_root or replicate_index == 0,
        )
    except Exception as exc:  # noqa: BLE001 - cache must never fail a load
        logger.warning("shard cache disabled for this load (context error): %s", exc)
        return None


def _all_ranks_agree(local_ok: bool, device: torch.device) -> bool:
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return local_ok
    flag = torch.tensor([1 if local_ok else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _validate_entry(name: str, entry: dict[str, Any], meta_param: torch.Tensor) -> bool:
    if entry["dtype"] != str(meta_param.dtype):
        return False
    if list(entry["global_shape"]) != list(meta_param.shape):
        return False
    is_dtensor = isinstance(meta_param, DTensor)
    if entry["kind"] != ("dtensor" if is_dtensor else "tensor"):
        return False
    if is_dtensor:
        if entry["placements"] != [str(p) for p in meta_param.placements]:
            return False
        if list(entry["local_shape"]) != list(meta_param.to_local().shape):
            return False
    return True


def try_load_from_shard_cache(
    model: nn.Module,
    ctx: ShardCacheContext,
    device: torch.device,
    *,
    strict: bool = True,
) -> bool:
    """Assemble the model's state dict from cached local shards. Returns False
    (without mutating the model) on any mismatch."""
    try:
        manifest_path = ctx.entry_dir / "manifest.json"
        shard_path = _shard_file(ctx.entry_dir, ctx.shard_index, ctx.num_shards)
        local_ok = manifest_path.is_file() and shard_path.is_file()
        manifest: dict[str, Any] = {}
        if local_ok:
            manifest = json.loads(manifest_path.read_text())
            local_ok = (manifest.get("format_version") == _FORMAT_VERSION and manifest.get("key") == ctx.key)
        meta_sd = model.state_dict()
        if local_ok:
            params_table = manifest["params"]
            for name, meta_param in meta_sd.items():
                entry = params_table.get(name)
                if entry is None:
                    if not any(pattern in name for pattern in _ALLOWED_NEW_PARAM_PATTERNS):
                        local_ok = False
                        break
                    continue
                if not _validate_entry(name, entry, meta_param):
                    local_ok = False
                    break
        if not _all_ranks_agree(local_ok, device):
            if local_ok:
                logger.info("shard cache: another rank missed entry %s; falling back to full load", ctx.key)
            return False

        from safetensors import safe_open

        named_buffers = dict(model.named_buffers())
        dtype_selector = getattr(model, "_get_parameter_dtype", None)
        sharded_sd: dict[str, Any] = {}
        with safe_open(str(shard_path), framework="pt", device=str(device)) as f:
            cached_keys = set(f.keys())
            for name, meta_param in meta_sd.items():
                if name in cached_keys:
                    local = f.get_tensor(name)
                    if isinstance(meta_param, DTensor):
                        tensor: torch.Tensor = DTensor.from_local(
                            local,
                            meta_param.device_mesh,
                            meta_param.placements,
                            run_check=False,
                            shape=meta_param.shape,
                            stride=meta_param.stride(),
                        )
                    else:
                        tensor = local
                else:
                    # Zero-init new params exactly like the full-load path.
                    target_dtype = meta_param.dtype
                    if callable(dtype_selector):
                        target_dtype = dtype_selector(name, target_dtype)
                    if isinstance(meta_param, DTensor):
                        local = torch.zeros(meta_param.to_local().shape, device=device, dtype=target_dtype)
                        tensor = DTensor.from_local(
                            local,
                            meta_param.device_mesh,
                            meta_param.placements,
                            run_check=False,
                            shape=meta_param.shape,
                            stride=meta_param.stride(),
                        )
                    else:
                        tensor = torch.zeros(meta_param.shape, device=device, dtype=target_dtype)
                sharded_sd[name] = tensor if name in named_buffers else nn.Parameter(tensor)

        reverse_map = manifest.get("reverse_param_names_mapping", {})
        model.reverse_param_names_mapping = {k: tuple(v) for k, v in reverse_map.items()}
        model.load_state_dict(sharded_sd, strict=strict, assign=True)
        # Freshen mtimes so mtime-based tmpfs cleaners (and our own LRU GC)
        # treat actively used entries as recent.
        for p in (shard_path, manifest_path):
            try:
                os.utime(p)
            except OSError:
                pass
        logger.info(
            "shard cache HIT %s: %d tensors from %s",
            ctx.key,
            len(sharded_sd),
            shard_path,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - cache must never fail a load
        logger.warning("shard cache load failed (%s); falling back to full load", exc)
        return False


def write_shard_cache(model: nn.Module, ctx: ShardCacheContext) -> None:
    """Persist this rank's local shards after a successful full load."""
    try:
        from safetensors.torch import save_file

        tensors: dict[str, torch.Tensor] = {}
        params_table: dict[str, Any] = {}
        for name, value in model.state_dict().items():
            if isinstance(value, DTensor):
                local = value.to_local().detach().to("cpu", copy=True).contiguous()
                params_table[name] = {
                    "kind": "dtensor",
                    "dtype": str(value.dtype),
                    "global_shape": list(value.shape),
                    "placements": [str(p) for p in value.placements],
                    "local_shape": list(local.shape),
                }
            else:
                local = value.detach().to("cpu", copy=True).contiguous()
                params_table[name] = {
                    "kind": "tensor",
                    "dtype": str(value.dtype),
                    "global_shape": list(value.shape),
                }
            tensors[name] = local

        ctx.entry_dir.mkdir(parents=True, exist_ok=True)
        needed = sum(t.numel() * t.element_size() for t in tensors.values())
        free = shutil.disk_usage(ctx.entry_dir).free
        if free < needed + _WRITE_MARGIN_BYTES:
            logger.warning(
                "shard cache: skipping write (%.1f GiB needed, %.1f GiB free at %s)",
                needed / 2**30,
                free / 2**30,
                ctx.entry_dir,
            )
            _barrier_if_initialized()
            return

        if ctx.is_writer:
            shard_path = _shard_file(ctx.entry_dir, ctx.shard_index, ctx.num_shards)
            tmp_path = shard_path.with_suffix(".safetensors.tmp")
            save_file(tensors, str(tmp_path))
            os.replace(tmp_path, shard_path)
        _barrier_if_initialized()

        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        if rank == 0:
            reverse_map = {
                k: list(v)
                for k, v in getattr(model, "reverse_param_names_mapping", {}).items()
            }
            manifest = {
                "format_version": _FORMAT_VERSION,
                "key": ctx.key,
                "num_shards": ctx.num_shards,
                "params": params_table,
                "reverse_param_names_mapping": reverse_map,
            }
            manifest_tmp = ctx.entry_dir / "manifest.json.tmp"
            manifest_tmp.write_text(json.dumps(manifest))
            os.replace(manifest_tmp, ctx.entry_dir / "manifest.json")
            logger.info("shard cache WRITE %s: %d tensors -> %s", ctx.key, len(tensors), ctx.entry_dir)
            _gc_cache_root(ctx.entry_dir.parent, keep=ctx.entry_dir.name)
    except Exception as exc:  # noqa: BLE001 - cache must never fail a run
        logger.warning("shard cache write failed (non-fatal): %s", exc)
        _barrier_if_initialized()


def _barrier_if_initialized() -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def _gc_cache_root(root: Path, keep: str) -> None:
    """Drop least-recently-written entries beyond the size cap."""
    try:
        max_bytes = float(os.environ.get(_ENV_MAX_GB, "300")) * 2**30
        entries = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            manifest = child / "manifest.json"
            size = sum(f.stat().st_size for f in child.glob("*") if f.is_file())
            mtime = manifest.stat().st_mtime if manifest.is_file() else 0.0
            entries.append((mtime, size, child))
        total = sum(size for _, size, _ in entries)
        for mtime, size, child in sorted(entries):
            if total <= max_bytes:
                break
            if child.name == keep:
                continue
            shutil.rmtree(child, ignore_errors=True)
            total -= size
            logger.info("shard cache GC: evicted %s (%.1f GiB)", child, size / 2**30)
    except Exception as exc:  # noqa: BLE001
        logger.warning("shard cache GC failed (non-fatal): %s", exc)
