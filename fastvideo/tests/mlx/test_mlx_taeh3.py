# SPDX-License-Identifier: Apache-2.0
"""TAEH3 port parity. Requires Apple MLX and an explicit upstream checkout.

TAEH3_REFERENCE_DIR=/path/to/taehv python -m pytest fastvideo/tests/mlx/test_mlx_taeh3.py
No weights or source code are downloaded by this test.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from fastvideo.mlx_runtime.minimax_h3_taeh3 import MLXTAEH3Decoder, ensure_taeh3_checkpoint


@pytest.fixture(scope='module')
def models():
    torch = pytest.importorskip('torch')
    location = os.environ.get('TAEH3_REFERENCE_DIR')
    if not location:
        pytest.skip('Set TAEH3_REFERENCE_DIR to an upstream taehv checkout')
    root = Path(location)
    spec = importlib.util.spec_from_file_location('taeh3_reference', root / 'taehv.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # FP64 avoids the CPU FP32 convolution's batch-size-dependent rounding.
    # The original FP32 comparison is recorded separately in the validation report.
    reference = module.TAEHV(str(root / 'taeh3.pth')).double().eval()
    actual = MLXTAEH3Decoder(root / 'safetensors/taeh3.safetensors')
    return torch, reference, actual


@pytest.mark.parametrize('reference_mode', ['serial_fp32', 'parallel_fp64'])
@pytest.mark.parametrize('batch,time,chunk', [(1, 2, 5), (1, 7, 1), (2, 7, 3), (1, 12, 5), (1, 37, 5)])
def test_mlx_taeh3_matches_upstream(models, batch, time, chunk, reference_mode):
    torch, reference, actual = models
    latent = np.random.default_rng(123).standard_normal((batch, time, 24, 2, 3)).astype(np.float32)
    with torch.no_grad():
        dtype = torch.float64 if reference_mode == "parallel_fp64" else torch.float32
        reference = reference.to(dtype=dtype)
        expected = reference.decode_video(torch.from_numpy(latent).to(dtype=dtype),
                                          parallel=reference_mode == "parallel_fp64",
                                          show_progress_bar=False).float().numpy()
    output = np.asarray(actual.decode_ntchw(mx.array(latent), chunk_size=chunk))
    assert output.shape == expected.shape
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output, expected, atol=1e-5, rtol=1e-5)


def test_invalid_latent_length_rejected(models):
    _, _, actual = models
    with pytest.raises(ValueError, match='5\\*k-3'):
        actual.decode_ntchw(mx.zeros((1, 5, 24, 2, 2)))


def test_cache_tampering_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    cache = tmp_path / '.cache/fastvideo/taehv/taeh3.safetensors'
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='SHA-256'):
        ensure_taeh3_checkpoint()


def test_local_checkpoint_requires_safetensors(tmp_path):
    path = tmp_path / 'weights.pth'
    path.touch()
    with pytest.raises(ValueError, match='safetensors'):
        ensure_taeh3_checkpoint(path)


def test_chunk_size_does_not_reset_temporal_state(models):
    _, _, actual = models
    latent = mx.array(np.random.default_rng(11).standard_normal((1, 12, 24, 2, 3)).astype(np.float32))
    expected = np.asarray(actual.decode_ntchw(latent, chunk_size=12))
    for size in (1, 3, 5):
        np.testing.assert_array_equal(np.asarray(actual.decode_ntchw(latent, chunk_size=size)), expected)


def test_strided_latents_match_contiguous(models):
    _, _, actual = models
    latent = mx.array(np.random.default_rng(9).standard_normal((1, 7, 24, 2, 6)).astype(np.float32))[:, :, :, :, ::2]
    expected = actual.decode_ntchw(mx.contiguous(latent))
    np.testing.assert_array_equal(np.asarray(actual.decode_ntchw(latent)), np.asarray(expected))


def test_incompatible_checkpoint_fails_before_decode(tmp_path):
    path = tmp_path / 'bad.safetensors'
    mx.save_safetensors(str(path), {'decoder.1.weight': mx.zeros((1, 1, 1, 1))})
    with pytest.raises(ValueError, match='keys mismatch'):
        MLXTAEH3Decoder(path)


def test_failed_download_does_not_leave_cache(monkeypatch, tmp_path):
    import io
    from fastvideo.mlx_runtime import minimax_h3_taeh3 as module
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *args, **kwargs: io.BytesIO(b'corrupt'))
    with pytest.raises(RuntimeError, match='SHA-256'):
        ensure_taeh3_checkpoint()
    assert list((tmp_path / '.cache/fastvideo/taehv').iterdir()) == []


def test_pipeline_taeh3_skips_full_vae_and_denormalization(monkeypatch):
    from fastvideo.mlx_runtime import minimax_h3_taeh3 as module
    from fastvideo.mlx_runtime import minimax_h3_video_vae as full_vae
    from fastvideo.mlx_runtime.minimax_h3_pipeline import MiniMaxH3MLXPipeline

    pipeline = MiniMaxH3MLXPipeline.__new__(MiniMaxH3MLXPipeline)
    pipeline.video_decode_backend = 'taeh3'
    pipeline.taeh3_checkpoint = None
    pipeline.taeh3_chunk_size = 5
    pipeline.vae_dtype = 'fp32'
    pipeline._dit_in_channels = 24
    pipeline._dit_patch_size = (1, 2, 2)

    def fail(*args, **kwargs):
        raise AssertionError('Full VAE must not load for TAEH3')

    def fake_decode(latents, **kwargs):
        assert latents.shape == (1, 24, 2, 2, 2)
        np.testing.assert_array_equal(latents, np.full_like(latents, 2.0))
        return np.full((1, 5, 32, 32, 3), 0.5, dtype=np.float32)

    monkeypatch.setattr(full_vae, 'mlx_h3_video_vae_from_dir', fail)
    monkeypatch.setattr(module, 'decode_latents_taeh3_mlx', fake_decode)
    frames = pipeline.decode_video(np.full((2, 96), 2.0, dtype=np.float32), height=32, width=32, num_frames=5)
    assert frames.shape == (5, 32, 32, 3)
    assert frames.dtype == np.uint8
    assert np.all(frames == 127)
