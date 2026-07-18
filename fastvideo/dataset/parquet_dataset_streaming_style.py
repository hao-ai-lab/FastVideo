# SPDX-License-Identifier: Apache-2.0
"""Low-amplification Parquet streaming for large, shared T2V datasets.

The map-style loader remains the default.  This opt-in loader keeps the source
dataset read-only, stores only JSON metadata in a caller-owned cache, projects
the requested T2V columns, and consumes each assigned row group sequentially.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.distributed import get_sp_world_size, get_world_rank, get_world_size
from fastvideo.logger import init_logger

logger = init_logger(__name__)

_MANIFEST_VERSION = 2
_STATE_VERSION = 2


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _real(path: str | os.PathLike[str]) -> str:
    return os.path.realpath(os.path.expanduser(os.fspath(path)))


def _assert_user_owned_manifest(dataset_root: str, manifest_path: str) -> None:
    manifest_parent = _real(os.path.dirname(manifest_path) or ".")
    if os.path.commonpath([dataset_root, manifest_parent]) == dataset_root:
        raise ValueError(
            f"streaming_manifest_path must be outside the read-only dataset root ({dataset_root}); got {manifest_path}"
        )


def _manifest_fingerprint(row_groups: list[dict[str, Any]]) -> str:
    payload = json.dumps(row_groups, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_file_path(dataset_root: str, relative_path: str) -> str:
    file_path = _real(os.path.join(dataset_root, relative_path))
    if os.path.commonpath([dataset_root, file_path]) != dataset_root:
        raise ValueError(f"Manifest file escapes dataset root: {relative_path!r}")
    return file_path


def _manifest_matches_source(
    dataset_root: str,
    manifest: dict[str, Any],
) -> bool:
    """Validate cached metadata without reopening every Parquet footer."""
    row_groups = manifest.get("row_groups")
    if not isinstance(row_groups, list) or not row_groups:
        return False
    expected_sample_start = 0
    stats: dict[str, os.stat_result] = {}
    try:
        for item in row_groups:
            relative_path = str(item["file"])
            rows = int(item["rows"])
            row_group = int(item["row_group"])
            sample_start = int(item["sample_start"])
            if rows <= 0 or row_group < 0 or sample_start != expected_sample_start:
                return False
            expected_sample_start += rows
            if relative_path not in stats:
                file_path = _manifest_file_path(dataset_root, relative_path)
                stats[relative_path] = os.stat(file_path)
            stat = stats[relative_path]
            if int(item["file_size"]) != stat.st_size or int(item["file_mtime_ns"]) != stat.st_mtime_ns:
                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return int(manifest.get("total_rows", -1)) == expected_sample_start and manifest.get(
        "fingerprint"
    ) == _manifest_fingerprint(row_groups)


def _scan_manifest(dataset_root: str, columns: tuple[str, ...]) -> dict[str, Any]:
    parquet_files: list[str] = []
    for root, _, files in os.walk(dataset_root):
        for name in sorted(files):
            if name.endswith(".parquet"):
                parquet_files.append(_real(os.path.join(root, name)))
    parquet_files.sort()
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under dataset path: {dataset_root}")

    row_groups: list[dict[str, Any]] = []
    sample_start = 0
    for file_path in parquet_files:
        file_stat = os.stat(file_path)
        parquet_file = pq.ParquetFile(file_path)
        missing = sorted(set(columns) - set(parquet_file.schema_arrow.names))
        if missing:
            raise ValueError(f"Parquet file {file_path} is missing projected T2V columns: {missing}")
        relative_path = os.path.relpath(file_path, dataset_root)
        for row_group_index in range(parquet_file.num_row_groups):
            rows = parquet_file.metadata.row_group(row_group_index).num_rows
            row_groups.append(
                {
                    "file": relative_path,
                    "row_group": row_group_index,
                    "rows": rows,
                    "sample_start": sample_start,
                    "file_size": file_stat.st_size,
                    "file_mtime_ns": file_stat.st_mtime_ns,
                }
            )
            sample_start += rows

    return {
        "version": _MANIFEST_VERSION,
        "dataset_root": dataset_root,
        "columns": list(columns),
        "total_rows": sample_start,
        "row_groups": row_groups,
        "fingerprint": _manifest_fingerprint(row_groups),
    }


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(dir=parent, prefix=f".{os.path.basename(path)}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_or_create_streaming_manifest(
    dataset_root: str, manifest_path: str, columns: tuple[str, ...]
) -> dict[str, Any]:
    if os.name != "posix":
        raise RuntimeError("The streaming Parquet loader currently requires POSIX file locks")
    import fcntl

    dataset_root = _real(dataset_root)
    manifest_path = _real(manifest_path)
    _assert_user_owned_manifest(dataset_root, manifest_path)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    lock_path = f"{manifest_path}.lock"
    with open(lock_path, "a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        manifest: dict[str, Any] | None = None
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as handle:
                candidate = json.load(handle)
            if (
                candidate.get("version") == _MANIFEST_VERSION
                and candidate.get("dataset_root") == dataset_root
                and candidate.get("columns") == list(columns)
                and _manifest_matches_source(dataset_root, candidate)
            ):
                manifest = candidate
        if manifest is None:
            logger.info("Building safe JSON Parquet manifest at %s", manifest_path)
            manifest = _scan_manifest(dataset_root, columns)
            _write_json_atomic(manifest_path, manifest)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return manifest


def _contiguous_row_group_shards(row_groups: list[dict[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
    """Split ordered row groups into row-balanced contiguous shards."""
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    remaining_rows = sum(int(item["rows"]) for item in row_groups)
    cursor = 0
    for shard_index in range(num_shards):
        shards_left = num_shards - shard_index
        target = remaining_rows / shards_left if shards_left else 0
        shard_rows = 0
        while cursor < len(row_groups):
            item = row_groups[cursor]
            item_rows = int(item["rows"])
            items_left = len(row_groups) - cursor
            if shard_rows and shard_rows + item_rows > target and items_left >= shards_left - 1:
                break
            shards[shard_index].append(item)
            shard_rows += item_rows
            remaining_rows -= item_rows
            cursor += 1
        if cursor >= len(row_groups):
            break
    return shards


def reconstruct_streaming_dataset_state(
    manifest: dict[str, Any],
    *,
    global_rank: int,
    world_size: int,
    sp_world_size: int,
    num_workers: int,
    batch_size: int,
    read_batch_size: int,
    seed: int,
    shuffle_row_groups: bool,
    yielded_samples: int,
    worker_id: int = 0,
) -> dict[str, Any]:
    """Reconstruct an epoch-zero streaming cursor without reading Parquet.

    This is intentionally strict and is meant for audited migration of legacy
    checkpoints whose rank-local dataloader state was incorrectly stored in a
    shared DCP key.  It refuses epoch rollover because the exact wrapper state
    at that boundary depends on whether the iterator has observed StopIteration.
    """
    world_size = int(world_size)
    sp_world_size = int(sp_world_size)
    global_rank = int(global_rank)
    num_workers = max(int(num_workers), 1)
    batch_size = int(batch_size)
    read_batch_size = int(read_batch_size)
    yielded_samples = int(yielded_samples)
    worker_id = int(worker_id)
    if world_size < 1 or sp_world_size < 1 or world_size % sp_world_size:
        raise ValueError("world_size must be positive and divisible by sp_world_size")
    if not 0 <= global_rank < world_size:
        raise ValueError("global_rank is outside the requested world")
    if not 0 <= worker_id < num_workers:
        raise ValueError("worker_id is outside the requested worker layout")
    if batch_size < 1 or read_batch_size < 1:
        raise ValueError("batch_size and read_batch_size must be positive")
    if yielded_samples < 0 or yielded_samples % batch_size:
        raise ValueError("yielded_samples must be a non-negative batch multiple")

    row_groups = manifest.get("row_groups")
    if not isinstance(row_groups, list) or not row_groups:
        raise ValueError("Streaming manifest has no row groups")
    num_sp_groups = world_size // sp_world_size
    shards = _contiguous_row_group_shards(
        row_groups,
        num_sp_groups * num_workers,
    )
    shard_rows = [sum(int(item["rows"]) for item in shard) for shard in shards]
    samples_per_worker = min(shard_rows) // batch_size * batch_size
    if samples_per_worker == 0:
        raise ValueError("Dataset is too small for the requested topology")
    if yielded_samples >= samples_per_worker:
        raise ValueError(
            "Legacy cursor reconstruction only supports the first epoch and "
            "requires yielded_samples < samples_per_worker"
        )

    sp_group_index = global_rank // sp_world_size
    shard_index = sp_group_index * num_workers + worker_id
    shard = list(shards[shard_index])
    if shuffle_row_groups:
        random.Random(int(seed)).shuffle(shard)

    row_group_position = 0
    row_offset = 0
    remaining = yielded_samples
    if remaining:
        for position, item in enumerate(shard):
            rows = int(item["rows"])
            if remaining <= rows:
                row_group_position = position
                row_offset = remaining
                break
            remaining -= rows
        else:
            raise ValueError("yielded_samples exceeds the selected streaming shard")

    return {
        "version": _STATE_VERSION,
        "manifest_fingerprint": str(manifest["fingerprint"]),
        "topology": {
            "global_rank": global_rank,
            "world_size": world_size,
            "sp_world_size": sp_world_size,
            "sp_group_index": sp_group_index,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "read_batch_size": read_batch_size,
            "seed": int(seed),
            "shuffle_row_groups": bool(shuffle_row_groups),
            "samples_per_worker": samples_per_worker,
        },
        "worker_id": worker_id,
        "epoch": 0,
        "row_group_position": row_group_position,
        "row_offset": row_offset,
        "yielded_samples": yielded_samples,
    }


class LatentsParquetStreamingDataset(IterableDataset):
    """Sequential row-group loader with deterministic DP/SP sharding.

    Ranks in the same sequence-parallel group receive identical samples.
    Different data-parallel groups receive disjoint contiguous row-group
    shards.  Worker zero is a real worker shard, so ``num_workers=0`` is fully
    supported and is the recommended shared-filesystem setting.
    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        parquet_schema: pa.Schema,
        manifest_path: str,
        cfg_rate: float = 0.0,
        num_workers: int = 0,
        drop_last: bool = True,
        text_padding_length: int = 512,
        seed: int = 42,
        read_batch_size: int = 8,
        shuffle_row_groups: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(path, str):
            raise TypeError("The streaming loader currently accepts one dataset root")
        if not drop_last:
            raise ValueError("drop_last must be True for distributed streaming")
        if batch_size < 1 or read_batch_size < 1:
            raise ValueError("batch_size and read_batch_size must be positive")
        if not manifest_path:
            raise ValueError(
                "streaming_manifest_path is required; place it in user-owned storage, never inside the shared dataset"
            )

        self.path = _real(path)
        self.batch_size = int(batch_size)
        self.parquet_schema = parquet_schema
        self.columns = tuple(parquet_schema.names)
        self.manifest_path = _real(manifest_path)
        self.cfg_rate = float(cfg_rate)
        self.num_workers = max(int(num_workers), 1)
        self.text_padding_length = int(text_padding_length)
        self.seed = int(seed)
        self.read_batch_size = int(read_batch_size)
        self.shuffle_row_groups = bool(shuffle_row_groups)

        if dist.is_available() and dist.is_initialized():
            self.global_rank = get_world_rank()
            self.world_size = get_world_size()
            self.sp_world_size = get_sp_world_size()
        else:
            # Keep standalone tests and small I/O benchmarks useful without
            # requiring the full FastVideo distributed bootstrap.
            self.global_rank = 0
            self.world_size = 1
            self.sp_world_size = 1
        if self.world_size % self.sp_world_size:
            raise ValueError("world_size must be divisible by sp_world_size")
        self.num_sp_groups = self.world_size // self.sp_world_size
        self.sp_group_index = self.global_rank // self.sp_world_size

        if self.global_rank == 0:
            load_or_create_streaming_manifest(self.path, self.manifest_path, self.columns)
        _barrier()
        with open(self.manifest_path, encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("dataset_root") != self.path:
            raise ValueError("Streaming manifest belongs to a different dataset root")
        if self.manifest.get("columns") != list(self.columns):
            raise ValueError("Streaming manifest uses a different column projection")
        self.manifest_fingerprint = str(self.manifest["fingerprint"])

        num_shards = self.num_sp_groups * self.num_workers
        self.shards = _contiguous_row_group_shards(self.manifest["row_groups"], num_shards)
        shard_rows = [sum(int(item["rows"]) for item in shard) for shard in self.shards]
        self.samples_per_worker = min(shard_rows) // self.batch_size * self.batch_size
        if self.samples_per_worker == 0:
            raise ValueError("Dataset is too small for the requested DP/SP/worker layout")

        self._epoch = 0
        self._row_group_position = 0
        self._row_offset = 0
        self._yielded_samples = 0
        self._loaded_state: dict[str, Any] | None = None
        logger.info(
            "Streaming parquet: %d rows, %d row groups, %d samples per worker",
            int(self.manifest["total_rows"]),
            len(self.manifest["row_groups"]),
            self.samples_per_worker,
        )

    def _worker_id(self) -> int:
        info = get_worker_info()
        return 0 if info is None else int(info.id)

    def _worker_shard(self, worker_id: int) -> list[dict[str, Any]]:
        shard_index = self.sp_group_index * self.num_workers + worker_id
        shard = list(self.shards[shard_index])
        if self.shuffle_row_groups:
            random.Random(self.seed + self._epoch).shuffle(shard)
        return shard

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_id = self._worker_id()
        if worker_id >= self.num_workers:
            raise RuntimeError(f"Unexpected DataLoader worker id: {worker_id}")
        if self._loaded_state is not None:
            saved_worker = int(self._loaded_state.get("worker_id", worker_id))
            if saved_worker != worker_id:
                raise ValueError(f"Dataloader state belongs to worker {saved_worker}, not {worker_id}")
            self._loaded_state = None

        shard = self._worker_shard(worker_id)
        rows: list[dict[str, Any]] = []
        while self._yielded_samples < self.samples_per_worker and self._row_group_position < len(shard):
            item = shard[self._row_group_position]
            file_path = os.path.join(self.path, item["file"])
            parquet_file = pq.ParquetFile(file_path)
            consumed_in_group = 0
            record_batches = parquet_file.iter_batches(
                batch_size=self.read_batch_size,
                row_groups=[int(item["row_group"])],
                columns=list(self.columns),
                use_threads=False,
            )
            for record_batch in record_batches:
                record_count = record_batch.num_rows
                record_end = consumed_in_group + record_count
                if record_end <= self._row_offset:
                    consumed_in_group = record_end
                    continue
                start = max(self._row_offset - consumed_in_group, 0)
                for local_index, row in enumerate(record_batch.slice(start).to_pylist(), start=start):
                    global_offset = consumed_in_group + local_index
                    row["_sample_index"] = int(item["sample_start"]) + global_offset
                    rows.append(row)
                    self._row_offset = global_offset + 1
                    if len(rows) == self.batch_size:
                        self._yielded_samples += self.batch_size
                        yield collate_rows_from_parquet_schema(
                            rows,
                            self.parquet_schema,
                            self.text_padding_length,
                            cfg_rate=self.cfg_rate,
                            seed=self.seed,
                        )
                        rows = []
                        if self._yielded_samples >= self.samples_per_worker:
                            break
                consumed_in_group = record_end
                if self._yielded_samples >= self.samples_per_worker:
                    break
            if self._yielded_samples >= self.samples_per_worker:
                break
            self._row_group_position += 1
            self._row_offset = 0

        if self._yielded_samples != self.samples_per_worker:
            raise RuntimeError("Streaming shard ended before producing the balanced sample count")
        self._epoch += 1
        self._row_group_position = 0
        self._row_offset = 0
        self._yielded_samples = 0

    def __len__(self) -> int:
        return self.samples_per_worker * self.num_workers // self.batch_size

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "manifest_fingerprint": self.manifest_fingerprint,
            "topology": self._resume_topology(),
            "worker_id": self._worker_id(),
            "epoch": self._epoch,
            "row_group_position": self._row_group_position,
            "row_offset": self._row_offset,
            "yielded_samples": self._yielded_samples,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if int(state_dict.get("version", -1)) != _STATE_VERSION:
            raise ValueError("Cannot resume: unsupported streaming dataset state")
        if state_dict.get("manifest_fingerprint") != self.manifest_fingerprint:
            raise ValueError("Cannot resume: streaming dataset manifest changed")
        if state_dict.get("topology") != self._resume_topology():
            raise ValueError("Cannot resume: streaming loader topology or sampling config changed")
        self._epoch = int(state_dict["epoch"])
        self._row_group_position = int(state_dict["row_group_position"])
        self._row_offset = int(state_dict["row_offset"])
        self._yielded_samples = int(state_dict["yielded_samples"])
        self._loaded_state = dict(state_dict)

    def _resume_topology(self) -> dict[str, Any]:
        return {
            "global_rank": self.global_rank,
            "world_size": self.world_size,
            "sp_world_size": self.sp_world_size,
            "sp_group_index": self.sp_group_index,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "read_batch_size": self.read_batch_size,
            "seed": self.seed,
            "shuffle_row_groups": self.shuffle_row_groups,
            "samples_per_worker": self.samples_per_worker,
        }

    def get_validation_negative_prompt(self) -> tuple[Any, Any, str]:
        first = self.manifest["row_groups"][0]
        parquet_file = pq.ParquetFile(os.path.join(self.path, first["file"]))
        first_batch = next(
            parquet_file.iter_batches(
                batch_size=1,
                row_groups=[int(first["row_group"])],
                columns=list(self.columns),
                use_threads=False,
            )
        )
        row = first_batch.to_pylist()[0]
        batch = collate_rows_from_parquet_schema(
            [row], self.parquet_schema, self.text_padding_length, cfg_rate=0.0, seed=self.seed
        )
        embedding = batch["text_embedding"]
        mask = batch["text_attention_mask"]
        prompt = batch["info_list"][0]["prompt"]
        return embedding, mask, prompt


def build_parquet_streaming_style_dataloader(
    path: str,
    batch_size: int,
    num_data_workers: int,
    parquet_schema: pa.Schema,
    manifest_path: str,
    cfg_rate: float = 0.0,
    drop_last: bool = True,
    text_padding_length: int = 512,
    seed: int = 42,
    read_batch_size: int = 8,
    shuffle_row_groups: bool = True,
) -> tuple[LatentsParquetStreamingDataset, StatefulDataLoader]:
    dataset = LatentsParquetStreamingDataset(
        path=path,
        batch_size=batch_size,
        parquet_schema=parquet_schema,
        manifest_path=manifest_path,
        cfg_rate=cfg_rate,
        num_workers=num_data_workers,
        drop_last=drop_last,
        text_padding_length=text_padding_length,
        seed=seed,
        read_batch_size=read_batch_size,
        shuffle_row_groups=shuffle_row_groups,
    )
    loader = StatefulDataLoader(
        dataset,
        batch_size=None,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader
