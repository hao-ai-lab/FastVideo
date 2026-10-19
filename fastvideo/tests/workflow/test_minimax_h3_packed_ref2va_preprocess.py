# SPDX-License-Identifier: Apache-2.0
"""CPU tests for native geometry and transactional packed Ref2VA preprocessing."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastvideo.dataset.dataloader.record_schema import ref2va_record_creator
from fastvideo.pipelines.preprocess import preprocess_minimax_h3_packed_ref2va as prep


def manifest(tmp_path, **extra):
    (tmp_path / 'target.mp4').touch()
    (tmp_path / 'reference.png').touch()
    row = dict(schema_version='minimax_h3_ref2va_raw_v1', id='sample', target={'video': 'target.mp4'},
               caption='a caption', references=[{'type': 'image', 'image': 'reference.png'}], **extra)
    path = tmp_path / 'input.jsonl'
    path.write_text(json.dumps(row) + '\n')
    return path


def record(geometry):
    return ref2va_record_creator(
        'sample', 'a caption', file_name='target.mp4', width=geometry.width, height=geometry.height,
        num_frames=geometry.num_frames,
        text_embedding=np.ones((3, 5120)), text_token_tags=np.ones(3),
        reference_video_latent_rows=np.ones((1, 96)), reference_audio_latent_rows=None,
        reference_media_types=['image'], reference_has_audio=[False], reference_num_latent_frames=[1],
        reference_latent_heights=[2], reference_latent_widths=[2], reference_num_audio_latents=[0],
        reference_noise_seed=42,
        vae_latent=np.ones((24, prep.video_latent_num_frames(geometry.num_frames), geometry.height // 16,
                            geometry.width // 16)),
        audio_latent=np.ones((2, 32, prep.audio_latent_num_frames(geometry.num_frames))),
    )


def test_manifest_geometry_is_per_record_and_paths_resolve(tmp_path):
    path = manifest(tmp_path, target_geometry={'width': 64, 'height': 32, 'num_frames': 22})
    sample, geometry = prep.load_samples(path, prep.TargetGeometry())[0]
    assert geometry.bucket == 'bucket=64x32-22f'
    assert sample.target_video_path == tmp_path / 'target.mp4'
    assert sample.references[0].image_path == tmp_path / 'reference.png'


@pytest.mark.parametrize('kwargs', [{'width': 33}, {'height': True}, {'num_frames': 24}, {'num_frames': -1}])
def test_geometry_rejects_invalid_vae_or_patch_dimensions(kwargs):
    with pytest.raises(ValueError):
        prep.TargetGeometry(**kwargs)


def test_target_media_preserves_pair_duration_and_resizes(monkeypatch):
    geometry = prep.TargetGeometry(64, 32, 22)
    expected = round(22 / 24 * 32000)
    frames = np.zeros((23, 16, 32, 3), dtype=np.uint8)
    waveform = torch.ones(1, expected + 500)
    monkeypatch.setattr(prep, 'decode_reference_video', lambda _: (frames, 24.0, (waveform, 32000)))
    video, audio = prep.load_target_media(Path('unused'), geometry)
    assert video.shape == (22, 32, 64, 3)
    assert audio.shape == (2, expected)
    assert torch.equal(audio[0], audio[1])
    monkeypatch.setattr(prep, 'decode_reference_video', lambda _: (frames, 24.0, (waveform[:, :4], 32000)))
    with pytest.raises(ValueError, match='shorter'):
        prep.load_target_media(Path('unused'), geometry)


def setup_encoder(tmp_path, monkeypatch):
    path = manifest(tmp_path)
    model = tmp_path / 'model'
    model.mkdir()
    monkeypatch.setattr(prep.encoding, '_init_single_process_distributed', lambda: None)
    monkeypatch.setattr(prep.encoding, '_build_fastvideo_args', lambda _: (object(), (1, 2, 2)))
    monkeypatch.setattr(prep, 'verify_model_config_and_directory', lambda _: {
        k: [] for k in ('vae', 'audio_vae', 'tokenizer', 'processor', 'text_encoder', 'transformer_ref')})
    return path, model


def test_native_bucket_output_is_validated_and_never_overwritten(tmp_path, monkeypatch):
    path, model = setup_encoder(tmp_path, monkeypatch)
    geometry = prep.TargetGeometry(32, 32, 22)
    monkeypatch.setattr(prep, 'build_record', lambda *_args, **_kwargs: record(geometry))
    output = tmp_path / 'output'
    paths = prep.preprocess(manifest_path=path, model_path=model, output_dir=output, geometry=geometry)
    assert paths == [output / 'bucket=32x32-22f/data_00000000.parquet']
    assert paths[0].is_file()
    previous = paths[0].read_bytes()
    with pytest.raises(FileExistsError):
        prep.preprocess(manifest_path=path, model_path=model, output_dir=output, geometry=geometry)
    assert paths[0].read_bytes() == previous


def test_nonfinite_encoder_output_does_not_publish_dataset(tmp_path, monkeypatch):
    path, model = setup_encoder(tmp_path, monkeypatch)
    geometry = prep.TargetGeometry(32, 32, 22)
    row = record(geometry)
    row['text_embedding_bytes'] = np.full((3, 5120), np.nan, dtype=np.float32).tobytes()
    monkeypatch.setattr(prep, 'build_record', lambda *_args, **_kwargs: row)
    with pytest.raises(ValueError, match='Non-finite'):
        prep.preprocess(manifest_path=path, model_path=model, output_dir=tmp_path / 'output', geometry=geometry)
    assert not (tmp_path / 'output').exists()
    assert not list(tmp_path.glob('.output.staging-*'))


def test_build_record_requests_clean_visual_rows(monkeypatch):
    geometry = prep.TargetGeometry(32, 32, 22)
    ref = SimpleNamespace(media_type='image', has_audio=False, num_latent_frames=1, latent_height=2,
                          latent_width=2, num_audio_latents=0)
    sample = SimpleNamespace(target_video_path=Path('target'), target_file='target', caption='caption',
                             sample_id='sample', references=[SimpleNamespace(media_type='image')])
    monkeypatch.setattr(prep, 'load_target_media', lambda *_: (object(), object()))
    monkeypatch.setattr(prep, 'build_minimax_h3_references', lambda _: [ref])
    monkeypatch.setattr(prep, 'prepare_reference', lambda *_: ref)
    monkeypatch.setattr(prep.encoding, 'encode_video_latents', lambda *_: torch.ones(24, 7, 2, 2))
    monkeypatch.setattr(prep.encoding, 'encode_audio_latents', lambda *_: torch.ones(2, 32, 23))
    def visual(*_args, augment):
        assert augment is False
        return torch.full((1, 96), 2.0)
    monkeypatch.setattr(prep.encoding, 'encode_ref_visual_anchor', visual)
    monkeypatch.setattr(prep.encoding, 'encode_ref_audio_anchor', lambda *_: torch.empty(0, 32))
    monkeypatch.setattr(prep.encoding, 'encode_ref2va_conditioning', lambda *_: (torch.ones(3, 5120), torch.ones(3)))
    row = prep.build_record(sample, geometry, model_path=Path('model'), model_index={}, fastvideo_args=object(),
                           patch_size=(1, 2, 2))
    assert np.all(np.frombuffer(row['reference_video_latent_rows_bytes'], dtype=np.float32) == 2.0)
    assert row['reference_noise_seed'] == 42


def test_visual_encoder_clean_opt_in_preserves_default_augmentation(monkeypatch):
    from PIL import Image
    from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
    original_device = torch.device
    monkeypatch.setattr(prep.encoding.torch, 'device', lambda value: original_device('cpu') if value == 'cuda:0' else original_device(value))
    monkeypatch.setattr(prep.encoding.torch.cuda, 'empty_cache', lambda: None)
    class Posterior:
        def sample(self, generator):
            return torch.full((1, 24, 1, 2, 2), 2.0)
    vae = SimpleNamespace(latent_channels=24, normalize_pixels=lambda x: x, normalize_latents=lambda x: x,
                          encode_keyframe=lambda _: SimpleNamespace(latent_dist=Posterior()))
    monkeypatch.setattr(prep.encoding, '_load_component', lambda *_: vae)
    monkeypatch.setattr(prep.encoding, 'keyframe_condition_noise', lambda *_args, **_kwargs: torch.full((1, 96), 4.0))
    reference = MiniMaxH3PreparedReference(media_type='image', image=Image.new('RGB', (32, 32)))
    clean = prep.encoding.encode_ref_visual_anchor([reference], Path('model'), {}, object(), (1, 2, 2), augment=False)
    default = prep.encoding.encode_ref_visual_anchor([reference], Path('model'), {}, object(), (1, 2, 2))
    assert torch.equal(clean, torch.full((1, 96), 2.0))
    assert torch.allclose(default, torch.full((1, 96), 2.002))
