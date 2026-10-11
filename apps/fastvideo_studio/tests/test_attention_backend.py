# SPDX-License-Identifier: Apache-2.0
"""Per-job attention selections must survive transport, restart and caching."""
from types import SimpleNamespace
import sqlite3

import pytest
from fastapi.testclient import TestClient
from fastvideo_studio.database import Database
from fastvideo_studio.job_runner import JobRunner
from fastvideo_studio import server


@pytest.fixture
def runner(tmp_path):
    result = JobRunner(str(tmp_path / 'out'), str(tmp_path / 'logs'), Database(tmp_path / 'jobs.db'))
    yield result
    result._shutdown()


def test_create_edit_duplicate_reload_preserves_attention(runner):
    job = runner.create_job('original', 'model', 'cat', attention_backend=' video_sparse_attn_h3 ', vsa_tile_size=64)
    assert job.to_dict()['attention_backend'] == 'VIDEO_SPARSE_ATTN_H3'
    runner.update_job_config(job.id, {'attention_backend': 'TORCH_SDPA', 'vsa_tile_size': 256,
                                    'seed': 7, 'vsa_sparsity': 0.8, 'dit_layerwise_offload': True,
                                    'references': [{'source': '/tmp/ref.png', 'media_type': 'image'}]})
    dup = runner.duplicate_job(job.id, 'copy')
    runner._jobs.clear()
    runner._load_jobs()
    for key in (job.id, dup.id):
        restored = runner.get_job(key)
        assert restored.attention_backend == 'TORCH_SDPA'
        assert restored.vsa_tile_size == 256
        assert restored.seed == 7
        assert restored.vsa_sparsity == 0.8
        assert restored.dit_layerwise_offload is True
        assert restored.references == [{'source': '/tmp/ref.png', 'media_type': 'image'}]
    runner.update_job_config(job.id, {'attention_backend': None})
    runner._load_jobs()
    assert runner.get_job(job.id).attention_backend is None


def test_reject_invalid_selection_without_mutation(runner):
    with pytest.raises(ValueError, match='attention backend'):
        runner.create_job('bad', 'model', 'cat', attention_backend='typo')
    assert runner.get_job('bad') is None
    job = runner.create_job('valid', 'model', 'cat')
    with pytest.raises(ValueError):
        runner.update_job_config(job.id, {'seed': 8, 'attention_backend': 'typo'})
    assert job.seed == 1024


def test_generator_cache_keys_and_forwards_attention(runner, monkeypatch):
    from fastvideo import VideoGenerator
    calls = []
    def build(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(shutdown=lambda: None)
    monkeypatch.setattr(VideoGenerator, 'from_pretrained', build)
    a = runner._get_or_create_generator('m', 't2v', 1, attention_backend='TORCH_SDPA', vsa_tile_size=64)
    assert runner._get_or_create_generator('m', 't2v', 1, attention_backend='TORCH_SDPA', vsa_tile_size=64) is a
    assert runner._get_or_create_generator('m', 't2v', 1, attention_backend='FLASH_ATTN', vsa_tile_size=64) is not a
    assert runner._get_or_create_generator('m', 't2v', 1, attention_backend='TORCH_SDPA', vsa_tile_size=256) is not a
    runner._get_or_create_generator('m', 't2v', 1)
    assert calls[0]['attention_backend'] == 'TORCH_SDPA'
    assert calls[0]['VSA_tile_size'] == 64
    assert calls[-1].get('attention_backend') is None


def test_old_database_defaults_to_auto(runner):
    job = runner.create_job('old', 'm', 'cat')
    conn = sqlite3.connect(runner._db._path)
    for col in ('attention_backend', 'vsa_tile_size'):
        conn.execute(f'ALTER TABLE jobs DROP COLUMN {col}')
    conn.commit()
    conn.close()
    migrated = Database(runner._db._path)
    row = migrated.get_all_jobs()[0]
    assert row['attention_backend'] is None
    assert row['vsa_tile_size'] is None


def test_api_validates_and_persists_selection(runner, monkeypatch):
    monkeypatch.setattr(server, 'job_runner', runner, raising=False)
    monkeypatch.setattr(server, 'database', runner._db)
    monkeypatch.setattr(server, '_available_models', [{'id': 'model', 'label': 'Model'}])
    client = TestClient(server.app)
    payload = {'model_id': 'model', 'prompt': 'cat', 'attention_backend': 'TORCH_SDPA', 'vsa_tile_size': 64}
    response = client.post('/api/jobs', json=payload)
    assert response.status_code == 201
    job_id = response.json()['id']
    assert response.json()['attention_backend'] == 'TORCH_SDPA'
    assert client.patch(f'/api/jobs/{job_id}', json={'attention_backend': 'bad'}).status_code == 400
    assert client.post('/api/jobs', json={**payload, 'attention_backend': 'bad'}).status_code == 400
    assert client.post('/api/jobs', json={**payload, 'vsa_tile_size': 0}).status_code == 422


def test_options_use_checked_in_recipe():
    response = TestClient(server.app).get('/api/inference-options', params={'model_id': 'FastVideo/FastVideo-FastH3-8-Step-V2'})
    assert response.status_code == 200
    data = response.json()
    assert 'TORCH_SDPA' in data['attention_backends']
    assert data['defaults']['attention_backend'] == 'VIDEO_SPARSE_ATTN_H3'
    assert data['defaults']['vsa_tile_size'] == 64
    assert data['defaults']['num_inference_steps'] == 9
    assert data['defaults']['guidance_scale'] == 1
    assert TestClient(server.app).get('/api/inference-options', params={'model_id': 'unknown'}).json()['defaults'] == {}


def test_invalid_edit_does_not_poison_restart(runner):
    job = runner.create_job('safe', 'model', 'cat')
    with pytest.raises(ValueError):
        runner.update_job_config(job.id, {'vsa_sparsity': 'invalid'})
    assert job.vsa_sparsity == 0
    runner._jobs.clear()
    runner._load_jobs()
    assert runner.get_job('safe').vsa_sparsity == 0


def test_auto_tile_defers_to_checkpoint(runner, monkeypatch):
    from fastvideo import VideoGenerator
    seen = []
    def build(*args, **kwargs):
        seen.append(kwargs.get('VSA_tile_size'))
        return SimpleNamespace(shutdown=lambda: None)
    monkeypatch.setattr(VideoGenerator, 'from_pretrained', build)
    job = runner.create_job('pdd', 'model', 'cat')
    runner._get_or_create_generator(job.model_id, 't2v', 1, vsa_tile_size=job.vsa_tile_size)
    assert seen == [None]


def test_recipes_requiring_external_options_are_not_advertised():
    client = TestClient(server.app)
    for model in ('FastVideo/FastWan2.1-T2V-1.3B-Diffusers', './CompactH3'):
        data = client.get('/api/inference-options', params={'model_id': model}).json()
        assert data['defaults'] == {}
