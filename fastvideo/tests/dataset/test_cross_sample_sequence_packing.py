# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for H3 cross-sample sequence packing."""

from __future__ import annotations

import pickle
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.dataset import parquet_dataset_map_style as parquet_dataset
from fastvideo.dataset.dataloader.schema import pyarrow_schema_ref2va, pyarrow_schema_t2va
from fastvideo.dataset.parquet_dataset_map_style import (
    DP_SP_BatchSampler,
    DP_SP_PackedBatchSampler,
    LatentsParquetMapStyleDataset,
    PackedBatchIndices,
    scan_h3_sequence_packing_metadata,
)
from fastvideo.dataset.sequence_capacity import (
    select_sequence_capacity,
    validate_fixed_sequence_capacities,
)


def _sampler(
    rank: int,
    *,
    sp_world_size: int = 1,
    compatibility_ids: list[str] | None = None,
) -> DP_SP_PackedBatchSampler:
    counts = [8, 2, 5, 3, 2]
    return DP_SP_PackedBatchSampler(
        sample_joint_token_counts=counts,
        sample_compatibility_ids=compatibility_ids or ["text-only"] * len(counts),
        max_joint_tokens=10,
        max_samples_per_pack=2,
        num_sp_groups=2,
        sp_world_size=sp_world_size,
        global_rank=rank,
        gradient_accumulation_steps=2,
        seed=19,
    )


def test_packed_sampler_is_restartable_and_weights_complete_accumulation_window() -> None:
    samplers = [_sampler(rank) for rank in range(2)]
    first = [list(sampler) for sampler in samplers]
    replay = [list(sampler) for sampler in samplers]

    assert [list(batch) for batches in first for batch in batches] == [
        list(batch) for batches in replay for batch in batches
    ]
    assert all(len(batches) == 2 for batches in first)
    window_document_count = first[0][0].window_document_count
    window_token_count = first[0][0].window_joint_token_count
    assert sum(len(batch) for batches in first for batch in batches) == window_document_count
    assert sum(batch.local_joint_token_count for batches in first for batch in batches) == window_token_count
    for batches in first:
        for batch in batches:
            assert batch.loss_scale == pytest.approx(
                2 * 2 * batch.local_effective_document_count / batch.window_effective_document_count)
            assert sum(batch.window_document_counts) == window_document_count
            assert len(batch.window_document_counts) == 4
            assert len(batch.local_window_document_counts) == 2
            assert sum([8, 2, 5, 3, 2][index] for index in batch) <= 10
            assert len(batch) <= 2

    restored = pickle.loads(pickle.dumps(first[0][0]))
    assert isinstance(restored, PackedBatchIndices)
    assert restored.loss_scale == first[0][0].loss_scale
    assert restored.window_document_count == first[0][0].window_document_count


def test_packed_sampler_keeps_sp_ranks_identical_and_never_mixes_compatibility() -> None:
    compatibility_ids = ["latent-bearing", "text-only", "latent-bearing", "text-only", "text-only"]
    samplers = [
        _sampler(rank, sp_world_size=2, compatibility_ids=compatibility_ids)
        for rank in range(4)
    ]
    batches = [list(sampler) for sampler in samplers]

    assert [list(batch) for batch in batches[0]] == [list(batch) for batch in batches[1]]
    assert [list(batch) for batch in batches[2]] == [list(batch) for batch in batches[3]]
    for rank_batches in batches:
        for batch in rank_batches:
            assert len({compatibility_ids[index] for index in batch}) == 1
    for dp0_batch, dp1_batch in zip(batches[0], batches[2], strict=True):
        assert ({compatibility_ids[index] for index in dp0_batch}
                == {compatibility_ids[index] for index in dp1_batch})


def test_packed_sampler_padding_preserves_compatibility_document_mix() -> None:
    compatibility_ids = ["text-only"] * 5 + ["latent-bearing"]
    samplers = [
        DP_SP_PackedBatchSampler(
            sample_joint_token_counts=[1] * 6,
            sample_compatibility_ids=compatibility_ids,
            max_joint_tokens=1,
            max_samples_per_pack=1,
            num_sp_groups=2,
            sp_world_size=1,
            global_rank=rank,
            gradient_accumulation_steps=2,
            seed=7,
        ) for rank in range(2)
    ]

    effective_by_compatibility = {"text-only": 0, "latent-bearing": 0}
    physical_by_compatibility = {"text-only": 0, "latent-bearing": 0}
    weights_by_compatibility: dict[str, list[float]] = {"text-only": [], "latent-bearing": []}
    for sampler in samplers:
        for batch in sampler:
            compatibility_id = compatibility_ids[batch[0]]
            physical_by_compatibility[compatibility_id] += len(batch)
            effective_by_compatibility[compatibility_id] += batch.local_effective_document_count
            weights_by_compatibility[compatibility_id].append(batch.local_effective_document_count)
            assert batch.loss_scale == pytest.approx(
                2 * 2 * batch.local_effective_document_count / batch.window_effective_document_count)

    # DP-round padding executes two duplicate documents, but each duplicate
    # family shares one source-document unit of weight and therefore cannot
    # change the original 5:1 mixture.
    assert physical_by_compatibility == {"text-only": 6, "latent-bearing": 2}
    assert effective_by_compatibility == {"text-only": 5, "latent-bearing": 1}
    assert sorted(weights_by_compatibility["latent-bearing"]) == [0.5, 0.5]
    assert sorted(weights_by_compatibility["text-only"]) == [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
    assert sum(samplers[0].global_pack_effective_document_counts) == 6


class _PackedIndexDataset(Dataset):

    def __len__(self) -> int:
        return 5

    def __getitems__(self, indices):
        return indices


def _stateful_packed_loader(
    *,
    gradient_accumulation_steps: int = 2,
    fixed_sequence_capacities: list[int] | None = None,
) -> StatefulDataLoader:
    sampler = DP_SP_PackedBatchSampler(
        sample_joint_token_counts=[8, 2, 5, 3, 2],
        sample_compatibility_ids=["text-only"] * 5,
        max_joint_tokens=10,
        max_samples_per_pack=2,
        num_sp_groups=1,
        sp_world_size=1,
        global_rank=0,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=19,
        data_identity=[("/data/part.parquet", 5, 1234, 5678)],
        fixed_sequence_capacities=fixed_sequence_capacities,
    )
    return StatefulDataLoader(
        _PackedIndexDataset(),
        batch_sampler=sampler,
        collate_fn=lambda value: value,
        num_workers=0,
    )


def test_stateful_dataloader_roundtrip_restores_exact_packed_accumulation_index() -> None:
    loader = _stateful_packed_loader()
    iterator = iter(loader)
    first = next(iterator)
    assert first.accumulation_index == 0
    state = loader.state_dict()
    expected = next(iterator)
    assert expected.accumulation_index == 1

    restored_loader = _stateful_packed_loader()
    restored_loader.load_state_dict(state)
    actual = next(iter(restored_loader))

    assert list(actual) == list(expected)
    assert actual.accumulation_index == expected.accumulation_index
    assert actual.schedule_fingerprint == expected.schedule_fingerprint


def test_stateful_dataloader_rejects_changed_packing_schedule_on_resume() -> None:
    loader = _stateful_packed_loader(gradient_accumulation_steps=2)
    next(iter(loader))
    state = loader.state_dict()

    changed_loader = _stateful_packed_loader(gradient_accumulation_steps=1)
    changed_loader.load_state_dict(state)
    with pytest.raises(ValueError, match="schedule fingerprint changed"):
        iter(changed_loader)


def test_h3_metadata_scan_projects_shapes_only_and_computes_joint_cost(monkeypatch) -> None:
    observed_columns: list[str] = []

    class _FakeParquetFile:
        schema_arrow = pa.schema([
            pa.field("text_embedding_bytes", pa.binary()),
            pa.field("text_embedding_shape", pa.list_(pa.int64())),
            pa.field("vae_latent_bytes", pa.binary()),
            pa.field("vae_latent_shape", pa.list_(pa.int64())),
            pa.field("audio_latent_bytes", pa.binary()),
            pa.field("audio_latent_shape", pa.list_(pa.int64())),
        ])

        def read(self, *, columns):
            observed_columns.extend(columns)
            return pa.table({
                "text_embedding_shape": [[100, 4096], [200, 4096]],
                "vae_latent_shape": [[24, 2, 2, 4], []],
                "audio_latent_shape": [[2, 32, 8], []],
            })

    monkeypatch.setattr(parquet_dataset.pq, "ParquetFile", lambda _path: _FakeParquetFile())
    path = "/fake/bucket=64x32-5f/part.parquet"
    metadata = scan_h3_sequence_packing_metadata(
        [path],
        [2],
        ["bucket=64x32-5f"] * 2,
        pyarrow_schema_t2va,
    )

    # 5 pixel frames -> 2 video latent frames and 8 stereo audio rows:
    # 2 * (32/32) * (64/32) + 2 * 8 = 20 non-text joint rows.
    assert metadata.joint_token_counts == (120, 220)
    assert metadata.compatibility_ids == ("latent-bearing", "text-only")
    assert observed_columns == ["text_embedding_shape", "vae_latent_shape", "audio_latent_shape"]
    assert not any(column.endswith("_bytes") for column in observed_columns)


@pytest.mark.parametrize("capacities", [None, [256, 512]])
def test_packed_dataset_collates_each_ragged_document_as_a_singleton(monkeypatch, capacities) -> None:
    collated_row_indices: list[int] = []

    def fake_read(_files, global_row_idx, _lengths, columns=None):
        return {"id": f"sample-{global_row_idx}"}

    def fake_collate(rows, *_args, **_kwargs):
        assert len(rows) == 1
        collated_row_indices.append(rows[0]["_sample_index"])
        return {"row_index": rows[0]["_sample_index"]}

    monkeypatch.setattr(parquet_dataset, "read_row_from_parquet_file", fake_read)
    monkeypatch.setattr(parquet_dataset, "collate_rows_from_parquet_schema", fake_collate)
    dataset = LatentsParquetMapStyleDataset.__new__(LatentsParquetMapStyleDataset)
    dataset.parquet_files = ("unused.parquet", )
    dataset.lengths = (2, )
    dataset.parquet_schema = pa.schema([])
    dataset.text_padding_length = None
    dataset.max_presentation_tokens = 4096
    dataset.cfg_rate = 0.0
    dataset.seed = 42
    dataset.sequence_packing_enabled = True
    dataset.fixed_sequence_capacities = validate_fixed_sequence_capacities(capacities)
    dataset.sample_bucket_ids = ("bucket=64x32-5f", "bucket=96x32-5f")
    dataset.sample_joint_token_counts = (120, 220)
    dataset.sample_compatibility_ids = ("text-only", "text-only")
    indices = PackedBatchIndices(
        [0, 1],
        local_joint_token_count=340,
        window_document_count=7,
        window_joint_token_count=1_100,
        window_document_counts=(2, 2, 1, 2),
        local_window_document_counts=(2, 1),
        accumulation_index=0,
        num_dp_groups=2,
        gradient_accumulation_steps=2,
    )

    batch = dataset.__getitems__(indices)

    assert collated_row_indices == [0, 1]
    assert batch["_packed_samples"] == [
        {"row_index": 0, "_shape_bucket_id": "bucket=64x32-5f"},
        {"row_index": 1, "_shape_bucket_id": "bucket=96x32-5f"},
    ]
    assert batch["_packed_sample_joint_token_counts"] == (120, 220)
    assert batch["_packed_joint_token_count"] == 340
    assert batch["_sequence_packing_local_document_count"] == 2
    assert batch["_sequence_packing_window_document_count"] == 7
    assert batch["_sequence_packing_window_document_counts"] == (2, 2, 1, 2)
    assert batch["_sequence_packing_local_window_document_counts"] == (2, 1)
    assert batch["_sequence_packing_accumulation_index"] == 0
    assert batch["_sequence_packing_loss_scale"] == pytest.approx(8 / 7)
    if capacities:
        assert batch["_fixed_sequence_capacity"] == 512
        assert batch["_sequence_padding_ratio"] == 512 / 340
    else:
        assert "_fixed_sequence_capacity" not in batch
        assert "_sequence_padding_ratio" not in batch


@pytest.mark.parametrize("values", [[], [0], [-256], [257], [True], [256.0], ["256"], "256", 256])
def test_fixed_sequence_capacities_reject_invalid_lists(values) -> None:
    with pytest.raises(ValueError, match="fixed_sequence_capacities"):
        validate_fixed_sequence_capacities(values)


def test_fixed_sequence_capacity_chooses_smallest_fitting_value() -> None:
    assert validate_fixed_sequence_capacities(None) == ()
    assert validate_fixed_sequence_capacities([512, 256, 512]) == (256, 512)
    assert select_sequence_capacity(1, [512, 256]) == 256
    assert select_sequence_capacity(256, [256, 512]) == 256
    assert select_sequence_capacity(257, [256, 512]) == 512
    assert select_sequence_capacity(512, [256, 512]) == 512
    with pytest.raises(ValueError, match="513.*exceeding"):
        select_sequence_capacity(513, [256, 512])


@pytest.mark.parametrize("real_tokens", [0, -1, True, 1.5])
def test_fixed_sequence_capacity_rejects_invalid_real_token_count(real_tokens) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        select_sequence_capacity(real_tokens, [256])


def test_ref2va_metadata_counts_all_presentation_and_reference_rows(monkeypatch) -> None:
    observed_columns: list[str] = []

    class _FakeParquetFile:
        schema_arrow = pyarrow_schema_ref2va

        def read(self, *, columns):
            observed_columns.extend(columns)
            return pa.table({
                "text_embedding_shape": [[100, 5120], [200, 5120]],
                "vae_latent_shape": [[24, 2, 2, 4], []],
                "audio_latent_shape": [[2, 32, 8], []],
                "reference_video_latent_rows_shape": [[40, 96], [52, 96]],
                "reference_audio_latent_rows_shape": [[6, 32], [0, 32]],
            })

    monkeypatch.setattr(parquet_dataset.pq, "ParquetFile", lambda _path: _FakeParquetFile())
    metadata = scan_h3_sequence_packing_metadata(
        ["/fake/bucket=64x32-5f/part.parquet"], [2], ["bucket=64x32-5f"] * 2, pyarrow_schema_ref2va)

    # Target rows: 4 video + 16 stereo audio. Reference tensors are already
    # flattened joint rows, so neither channel count nor stereo is multiplied.
    assert metadata.joint_token_counts == (166, 272)
    assert metadata.compatibility_ids == ("latent-bearing", "text-only")
    assert observed_columns == [
        "text_embedding_shape", "vae_latent_shape", "audio_latent_shape",
        "reference_video_latent_rows_shape", "reference_audio_latent_rows_shape",
    ]
    assert all(column.endswith("_shape") for column in observed_columns)


@pytest.mark.parametrize("bad_shape", [None, [], [1], [-1, 96], [2, 24], [2, 2, 96]])
def test_ref2va_metadata_scan_rejects_invalid_reference_shapes(monkeypatch, bad_shape) -> None:
    class _FakeParquetFile:
        schema_arrow = pyarrow_schema_ref2va

        def read(self, *, columns):
            return pa.table({
                "text_embedding_shape": [[100, 5120]],
                "vae_latent_shape": [[]],
                "audio_latent_shape": [[]],
                "reference_video_latent_rows_shape": [bad_shape],
                "reference_audio_latent_rows_shape": [[0, 32]],
            })

    monkeypatch.setattr(parquet_dataset.pq, "ParquetFile", lambda _path: _FakeParquetFile())
    with pytest.raises(ValueError, match="invalid reference_video_latent_rows_shape"):
        scan_h3_sequence_packing_metadata(
            ["/fake/bucket=64x32-5f/part.parquet"], [1], ["bucket=64x32-5f"], pyarrow_schema_ref2va)


def test_ref2va_metadata_scan_requires_reference_shapes_before_payload_reads(monkeypatch) -> None:
    class _FakeParquetFile:
        schema_arrow = pyarrow_schema_t2va

        def read(self, *, columns):
            pytest.fail("missing reference shape metadata must fail before reading tensor payloads")

    monkeypatch.setattr(parquet_dataset.pq, "ParquetFile", lambda _path: _FakeParquetFile())
    with pytest.raises(ValueError, match="missing reference shape"):
        scan_h3_sequence_packing_metadata(
            ["/fake/bucket=64x32-5f/part.parquet"], [1], ["bucket=64x32-5f"], pyarrow_schema_ref2va)


@pytest.mark.parametrize("batch_size,native,capacities", [(2, True, [256]), (1, False, [256]), (1, True, [257])])
def test_fixed_capacity_dataset_validates_options_before_scanning(monkeypatch, batch_size, native, capacities) -> None:
    def unexpected_scan(_path):
        pytest.fail("invalid capacities/options must be rejected before any dataset I/O")

    monkeypatch.setattr(parquet_dataset, "get_parquet_files_and_length", unexpected_scan)
    with pytest.raises(ValueError, match="fixed_sequence_capacities"):
        LatentsParquetMapStyleDataset("/unused", batch_size, pyarrow_schema_ref2va,
                                      native_shape_bucketing=native, fixed_sequence_capacities=capacities)


def _capacity_dataset(
    tmp_path: Path,
    monkeypatch,
    *,
    capacities: list[int] | None = None,
    text_lengths_by_bucket: dict[str, list[int]] | None = None,
    rank: int = 0,
    world_size: int = 1,
    sp_size: int = 1,
) -> LatentsParquetMapStyleDataset:
    text_lengths_by_bucket = text_lengths_by_bucket or {
        "bucket=64x32-5f": [190, 191, 200],
        "bucket=96x32-5f": [100],
    }
    files, lengths = [], []
    shape_schema = pa.schema([field for field in pyarrow_schema_ref2va if field.name.endswith("_shape")])
    for bucket, text_lengths in text_lengths_by_bucket.items():
        path = tmp_path / bucket / "data.parquet"
        # Do not rewrite files on resume: file identity is part of the guard.
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist([{
                "text_embedding_shape": [length, 5120],
                "text_token_tags_shape": [length],
                "vae_latent_shape": [],
                "audio_latent_shape": [],
                "reference_video_latent_rows_shape": [40, 96],
                "reference_audio_latent_rows_shape": [6, 32],
            } for length in text_lengths], schema=shape_schema), path)
        files.append(str(path))
        lengths.append(len(text_lengths))

    def cached_files(_path, *, require_existing_parquet_cache=False):
        assert require_existing_parquet_cache is False
        return files, lengths

    monkeypatch.setattr(parquet_dataset, "get_parquet_files_and_length", cached_files)
    monkeypatch.setattr(parquet_dataset, "load_h3_sequence_packing_metadata", scan_h3_sequence_packing_metadata)
    monkeypatch.setattr(parquet_dataset, "load_parquet_data_identity",
                        lambda files, lengths: parquet_dataset.scan_parquet_data_identity(files, lengths))
    monkeypatch.setattr(parquet_dataset, "get_world_rank", lambda: rank)
    monkeypatch.setattr(parquet_dataset, "get_world_size", lambda: world_size)
    monkeypatch.setattr(parquet_dataset, "get_sp_world_size", lambda: sp_size)
    monkeypatch.setattr(parquet_dataset, "collate_rows_from_parquet_schema", lambda rows, *_args, **_kwargs: {
        "sample_indices": tuple(row["_sample_index"] for row in rows),
    })
    return LatentsParquetMapStyleDataset(
        str(tmp_path), 1, pyarrow_schema_ref2va, native_shape_bucketing=True,
        fixed_sequence_capacities=capacities, seed=19,
    )


def test_fixed_capacity_startup_overflow_reports_every_offending_bucket(tmp_path, monkeypatch) -> None:
    def unexpected_sampler(**_kwargs):
        pytest.fail("capacity overflow must be rejected before sampler/iterator construction")

    monkeypatch.setattr(parquet_dataset, "DP_SP_BatchSampler", unexpected_sampler)
    with pytest.raises(ValueError, match="largest capacity 512") as error:
        _capacity_dataset(tmp_path, monkeypatch, capacities=[256, 512], text_lengths_by_bucket={
            "bucket=64x32-5f": [600, 800],
            "bucket=96x32-5f": [700],
            "bucket=128x32-5f": [100],
        })
    assert "bucket=64x32-5f: 866 tokens" in str(error.value)
    assert "bucket=96x32-5f: 768 tokens" in str(error.value)
    assert "bucket=128x32-5f" not in str(error.value)


def test_fixed_capacity_single_document_batches_carry_exact_metadata(tmp_path, monkeypatch) -> None:
    dataset = _capacity_dataset(tmp_path, monkeypatch, capacities=[512, 256])
    assert type(dataset.sampler) is DP_SP_BatchSampler
    assert dataset.sequence_packing_enabled is False
    assert dataset.text_padding_length is None  # never truncate the scanned presentation
    assert dataset.sample_joint_token_counts == (256, 257, 266, 168)
    for index, real_tokens in enumerate(dataset.sample_joint_token_counts):
        batch = dataset.__getitems__([index])
        capacity = 256 if real_tokens <= 256 else 512
        assert batch["_fixed_sequence_capacity"] == capacity
        assert batch["_packed_joint_token_count"] == real_tokens
        assert batch["_sequence_padding_ratio"] == capacity / real_tokens
        assert batch["_shape_bucket_id"] == dataset.sample_bucket_ids[index]
        assert "_packed_samples" not in batch
        assert "_sequence_packing_loss_scale" not in batch


def test_fixed_capacity_preserves_native_shape_grouping_and_sp_schedule(tmp_path, monkeypatch) -> None:
    datasets = [
        _capacity_dataset(tmp_path, monkeypatch, capacities=[256, 512], rank=rank, world_size=4, sp_size=2)
        for rank in range(4)
    ]
    schedules = [list(dataset.sampler) for dataset in datasets]
    assert schedules[0] == schedules[1]
    assert schedules[2] == schedules[3]
    plain = _capacity_dataset(tmp_path, monkeypatch, rank=0, world_size=4, sp_size=2)
    assert type(plain.sampler) is DP_SP_BatchSampler
    assert schedules[0] == list(plain.sampler)
    assert datasets[0].sampler.bucket_padding == plain.sampler.bucket_padding
    for round_indices in zip(*schedules, strict=True):
        assert all(len(indices) == 1 for indices in round_indices)
        assert len({datasets[rank].sample_bucket_ids[indices[0]]
                    for rank, indices in enumerate(round_indices)}) == 1


def test_fixed_capacity_native_loader_restores_cursor_and_rejects_changed_capacities(tmp_path, monkeypatch) -> None:
    def loader(capacities):
        dataset = _capacity_dataset(tmp_path, monkeypatch, capacities=capacities)
        return StatefulDataLoader(dataset, batch_sampler=dataset.sampler,
                                  collate_fn=parquet_dataset.passthrough, num_workers=0)

    original = loader([256, 512])
    iterator = iter(original)
    next(iterator)
    state = original.state_dict()
    expected = next(iterator)
    restored = loader([512, 256, 512])
    restored.load_state_dict(state)
    assert next(iter(restored)) == expected
    for capacities in ([512], None):
        changed = loader(capacities)
        changed.load_state_dict(state)
        with pytest.raises(ValueError, match="capacity fingerprint changed"):
            next(iter(changed))


def test_fixed_capacity_packed_sampler_preserves_weights_but_guards_resume() -> None:
    unpadded = list(_stateful_packed_loader())
    loader = _stateful_packed_loader(fixed_sequence_capacities=[256])
    padded = list(loader)
    assert [list(batch) for batch in padded] == [list(batch) for batch in unpadded]
    assert [batch.loss_scale for batch in padded] == [batch.loss_scale for batch in unpadded]
    assert [batch.local_effective_document_count for batch in padded] == [
        batch.local_effective_document_count for batch in unpadded]
    next(iter(loader))
    state = loader.state_dict()
    for capacities in ([512], None):
        changed = _stateful_packed_loader(fixed_sequence_capacities=capacities)
        changed.load_state_dict(state)
        with pytest.raises(ValueError, match="schedule fingerprint changed"):
            iter(changed)
