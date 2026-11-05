# SPDX-License-Identifier: Apache-2.0
"""CPU serialization and bounded-presentation contracts for packed Ref2VA."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from fastvideo.dataset.dataloader.record_schema import ref2va_record_creator
from fastvideo.dataset.dataloader.schema import pyarrow_schema_ref2va
from fastvideo.dataset.utils import collate_rows_from_parquet_schema


def record(length=3, **overrides):
    args = dict(
        text_embedding=np.ones((length, 5120), dtype=np.float16),
        text_token_tags=np.arange(length) % 2,
        reference_video_latent_rows=np.ones((2, 96), dtype=np.float16),
        reference_audio_latent_rows=np.ones((4, 32), dtype=np.float32),
        reference_media_types=['image', 'audio'],
        reference_has_audio=[False, True],
        reference_num_latent_frames=[1, 0],
        reference_latent_heights=[2, 0],
        reference_latent_widths=[4, 0],
        reference_num_audio_latents=[0, 2],
        reference_noise_seed=42,
    )
    args.update(overrides)
    return ref2va_record_creator('sample', 'caption', **args)


def test_clean_reference_record_roundtrips_and_dynamic_tags_align(tmp_path):
    rows = [record(3), record(5)]
    path = tmp_path / 'rows.parquet'
    pq.write_table(pa.Table.from_pylist(rows, schema=pyarrow_schema_ref2va), path)
    rows = pq.ParquetFile(path).read().to_pylist()
    batch = collate_rows_from_parquet_schema(rows, pyarrow_schema_ref2va, None, max_presentation_tokens=8)
    assert batch['text_embedding'].shape == (2, 5, 5120)
    assert batch['text_embedding'].dtype == torch.float32
    assert batch['text_attention_mask'].sum(dim=1).tolist() == [3, 5]
    assert batch['text_token_tags'].tolist() == [[0, 1, 0, 0, 0], [0, 1, 0, 1, 0]]
    assert torch.equal(batch['reference_video_latent_rows'], torch.ones(2, 2, 96))
    assert batch['info_list'][0]['reference_media_types'] == ['image', 'audio']


@pytest.mark.parametrize('overrides,match', [
    ({'text_token_tags': np.array([0, 2, 1])}, 'only text'),
    ({'reference_video_latent_rows': np.ones((3, 96))}, 'wrong row count'),
    ({'reference_audio_latent_rows': np.ones((2, 32))}, 'wrong row count'),
    ({'reference_has_audio': [False]}, 'equal lengths'),
    ({'reference_noise_seed': True}, 'non-negative integer'),
    ({'vae_latent': np.ones((24, 2, 2, 2))}, 'both be present'),
])
def test_record_rejects_inconsistent_ordered_reference_contract(overrides, match):
    with pytest.raises(ValueError, match=match):
        record(**overrides)


def test_dynamic_presentation_limit_rejects_before_payload_decode():
    row = record(3)
    row['text_embedding_shape'] = [100, 5120]
    row['text_token_tags_shape'] = [100]
    row['vae_latent_bytes'] = b'invalid-if-decoded'
    with pytest.raises(ValueError, match='will not be truncated'):
        collate_rows_from_parquet_schema([row], pyarrow_schema_ref2va, None, max_presentation_tokens=99)


def test_dynamic_presentation_uses_persisted_dtype_size():
    row = record(3)
    row['text_embedding_dtype'] = 'float16'
    row['text_embedding_bytes'] = np.ones((3, 5120), dtype=np.float16).tobytes()
    batch = collate_rows_from_parquet_schema([row], pyarrow_schema_ref2va, None, max_presentation_tokens=3)
    assert batch['text_embedding'].dtype == torch.float16
    assert torch.equal(batch['text_embedding'], torch.ones(1, 3, 5120, dtype=torch.float16))
