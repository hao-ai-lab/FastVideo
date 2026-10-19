# SPDX-License-Identifier: Apache-2.0
"""Prompt-only preprocessing and native loader contract, using synthetic embeddings."""

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from fastvideo.dataset import parquet_dataset_map_style as data
from fastvideo.dataset.dataloader.schema import pyarrow_schema_text_only
from fastvideo.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.pipelines.preprocess import preprocess_minimax_h3_text_only as preprocessing


def prompt_row(**updates):
    return {"id": "one", "prompt": "  Water flows.\nBirds sing.  ", "width": 1344, "height": 768,
            "num_frames": 124, **updates}


def write_input(tmp_path, rows):
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def fake_world(monkeypatch):
    monkeypatch.setattr(data, "get_world_rank", lambda: 0)
    monkeypatch.setattr(data, "get_world_size", lambda: 2)
    monkeypatch.setattr(data, "get_sp_world_size", lambda: 2)
    monkeypatch.setattr(data, "get_world_group", lambda: SimpleNamespace(broadcast_object=lambda value, src: value))


def test_prompt_text_ids_and_geometry_survive_sharding(tmp_path):
    rows = [prompt_row(id=f"row-{index}") for index in range(5)]
    path = write_input(tmp_path, rows)
    first = preprocessing.load_prompt_records(path, shard_index=0, num_shards=2)
    second = preprocessing.load_prompt_records(path, shard_index=1, num_shards=2)
    assert [record.id for record in first] == ["row-0", "row-2", "row-4"]
    assert [record.id for record in second] == ["row-1", "row-3"]
    assert first[0].prompt == rows[0]["prompt"]
    assert first[0].geometry.bucket_id == "bucket=1344x768-124f"


@pytest.mark.parametrize("changes", [
    {"id": ""}, {"prompt": "  "}, {"prompt": 42}, {"width": True}, {"height": 768.0},
    {"width": 1345}, {"height": 0}, {"num_frames": 123}, {"num_frames": -12}, {"num_frames": 1},
    {"reference_image": "unexpected.png"},
])
def test_invalid_input_fails_before_encoding(tmp_path, changes):
    with pytest.raises(ValueError):
        preprocessing.load_prompt_records(write_input(tmp_path, [prompt_row(**changes)]))


def test_duplicate_ids_rejected_even_if_in_different_shards(tmp_path):
    path = write_input(tmp_path, [prompt_row(), prompt_row()])
    with pytest.raises(ValueError, match="duplicate id"):
        preprocessing.load_prompt_records(path, shard_index=0, num_shards=2)


def test_prompt_only_parquets_round_trip_through_native_strict_loader(tmp_path, monkeypatch):
    rows = [prompt_row(), prompt_row(id="two", prompt="Music.", width=768, height=1344, num_frames=243)]
    records = preprocessing.load_prompt_records(write_input(tmp_path, rows))
    vectors = {row["prompt"]: np.full((index + 2, 5120), index + 0.5, dtype=np.float32)
               for index, row in enumerate(rows)}
    output = tmp_path / "encoded"
    assert preprocessing.encode_records(records, output, vectors.__getitem__, samples_per_file=1) == 2
    files, lengths = pickle.loads((output / "map_style_cache/file_info.pkl").read_bytes())
    assert sum(lengths) == 2
    for file in files:
        assert pq.ParquetFile(file).schema_arrow == pyarrow_schema_text_only
    fake_world(monkeypatch)
    dataset = data.LatentsParquetMapStyleDataset(
        str(output), 1, pyarrow_schema_text_only, native_shape_bucketing=True,
        require_existing_parquet_cache=True, text_padding_length=None, max_presentation_tokens=16)
    seen = set()
    for indices in dataset.sampler:
        batch = dataset.__getitems__(indices)
        info = batch["info_list"][0]
        source = next(row for row in rows if row["id"] == info["id"])
        assert info["caption"] == source["prompt"]
        assert batch["_shape_bucket_id"] == f"bucket={source['width']}x{source['height']}-{source['num_frames']}f"
        assert torch.equal(batch["text_embedding"][0], torch.from_numpy(vectors[source["prompt"]]))
        assert batch["text_attention_mask"].all()
        assert "vae_latent" not in batch and "audio_latent" not in batch
        seen.add(info["id"])
    assert seen == {"one", "two"}


@pytest.mark.parametrize("embedding", [
    np.zeros((0, 5120), np.float32), np.zeros((2, 4), np.float32),
    np.zeros((4, 5120), np.float32), np.full((1, 5120), np.nan, np.float32),
])
def test_bad_embeddings_do_not_publish_cache(tmp_path, embedding):
    records = preprocessing.load_prompt_records(write_input(tmp_path, [prompt_row()]))
    output = tmp_path / "encoded"
    with pytest.raises(ValueError):
        preprocessing.encode_records(records, output, lambda prompt: embedding, max_tokens=3)
    assert not (output / "map_style_cache/file_info.pkl").exists()


def test_existing_output_fails_before_encoder_call(tmp_path):
    records = preprocessing.load_prompt_records(write_input(tmp_path, [prompt_row()]))
    with pytest.raises(FileExistsError):
        preprocessing.encode_records(records, tmp_path, lambda prompt: pytest.fail("encoder should not run"))


def test_text_only_projection_does_not_read_target_payloads(tmp_path, monkeypatch):
    path = tmp_path / "superset.parquet"
    pq.write_table(pa.table({"id": ["one"], "caption": ["A scene."], "text_embedding_bytes": [b"abcd"],
                            "text_embedding_shape": [[1, 1]], "text_embedding_dtype": ["float32"],
                            "unused_target_payload": [b"not a valid tensor"]}), path)
    original = data.pq.ParquetFile
    projected = []

    class RecordingParquet:
        def __init__(self, file):
            self.reader = original(file)
            self.metadata = self.reader.metadata
            self.num_row_groups = self.reader.num_row_groups

        def read_row_group(self, index, columns=None):
            projected.append(columns)
            return self.reader.read_row_group(index, columns=columns)

    monkeypatch.setattr(data.pq, "ParquetFile", RecordingParquet)
    row = data.read_row_from_parquet_file([str(path)], 0, [1], columns=pyarrow_schema_text_only.names)
    assert projected == [pyarrow_schema_text_only.names]
    assert set(row) == set(pyarrow_schema_text_only.names)


def test_dynamic_collate_keeps_long_text_and_fixed_collate_keeps_legacy_crop():
    embedding = np.arange(1030 * 2, dtype=np.float32).reshape(1030, 2)
    row = {"id": "long", "caption": "Prompt", "text_embedding_bytes": embedding.tobytes(),
           "text_embedding_shape": list(embedding.shape), "text_embedding_dtype": "float32"}
    dynamic = collate_rows_from_parquet_schema([row], pyarrow_schema_text_only, None, max_presentation_tokens=2048)
    fixed = collate_rows_from_parquet_schema([row], pyarrow_schema_text_only, 1024)
    assert torch.equal(dynamic["text_embedding"][0], torch.from_numpy(embedding))
    assert torch.equal(fixed["text_embedding"][0], torch.from_numpy(embedding[:1024]))
    with pytest.raises(ValueError, match="will not be truncated"):
        collate_rows_from_parquet_schema([row], pyarrow_schema_text_only, None, max_presentation_tokens=1024)
    with pytest.raises(ValueError, match="positive integer"):
        collate_rows_from_parquet_schema([row], pyarrow_schema_text_only, None)


@pytest.mark.parametrize("payload", [None, b"", b"short"])
def test_dynamic_collate_rejects_missing_or_malformed_embeddings(payload):
    row = {"id": "broken", "caption": "Prompt", "text_embedding_bytes": payload,
           "text_embedding_shape": [1, 5120], "text_embedding_dtype": "float32"}
    with pytest.raises(ValueError, match="matching its declared shape and dtype"):
        collate_rows_from_parquet_schema([row], pyarrow_schema_text_only, None, max_presentation_tokens=65536)
