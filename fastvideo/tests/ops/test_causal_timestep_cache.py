# SPDX-License-Identifier: Apache-2.0
"""Owned timestep frequency cache must retain the original numerical basis."""

import pytest
import torch

from fastvideo.layers.visual_embedding import TimestepEmbedder


def _embedder(dim=256, freq_dtype=torch.float32):
    embedder = TimestepEmbedder(16, frequency_embedding_size=dim, freq_dtype=freq_dtype, dtype=torch.float32).eval()
    # FastVideo linears reserve empty weights for the checkpoint loader.
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in embedder.parameters():
            parameter.normal_(mean=0, std=0.02, generator=generator)
    return embedder


@pytest.mark.parametrize("dim", [16, 17, 256])
def test_owned_frequency_cache_preserves_outputs_and_checkpoint_keys(dim):
    embedder = _embedder(dim)
    timesteps = torch.tensor([0., 0.125, 499.5, 1000.])
    keys = set(embedder.state_dict())
    reference = embedder(timesteps)
    assert torch.isfinite(reference).all()
    embedder.cache_frequencies = True
    torch.testing.assert_close(embedder(timesteps), reference, atol=0, rtol=0)
    frequency_storage = embedder._frequency_cache[1]
    torch.testing.assert_close(embedder(timesteps), reference, atol=0, rtol=0)
    assert embedder._frequency_cache[1] is frequency_storage
    assert set(embedder.state_dict()) == keys
    embedder.max_period = 20000
    embedder(timesteps)
    assert embedder._frequency_cache[1] is not frequency_storage


@pytest.mark.parametrize("initial_dtype,new_dtype", [(torch.float32, torch.float64), (torch.float64, torch.float32)])
def test_frequency_dtype_changes_rebuild_the_cache_and_preserve_outputs(initial_dtype, new_dtype):
    embedder = _embedder(17, freq_dtype=initial_dtype)
    timesteps = torch.tensor([0., 0.125, 499.5, 1000.])
    embedder.cache_frequencies = True
    embedder(timesteps)
    old_storage = embedder._frequency_cache[1]
    embedder.freq_dtype = new_dtype
    embedder.cache_frequencies = False
    expected = embedder(timesteps)
    embedder.cache_frequencies = True
    torch.testing.assert_close(embedder(timesteps), expected, atol=0, rtol=0)
    assert embedder._frequency_cache[1] is not old_storage
    assert embedder._frequency_cache[1].dtype == new_dtype


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
def test_frequency_cache_moves_from_cpu_to_gpu_without_changing_values():
    embedder = _embedder(17)
    timesteps = torch.tensor([0., 0.125, 499.5, 1000.])
    with torch.no_grad():
        embedder.cache_frequencies = True
        embedder(timesteps)
        cpu_storage = embedder._frequency_cache[1]
        embedder.cuda()
        timesteps = timesteps.cuda()
        embedder.cache_frequencies = False
        expected = embedder(timesteps)
        embedder.cache_frequencies = True
        torch.testing.assert_close(embedder(timesteps), expected, atol=0, rtol=0)
        assert embedder._frequency_cache[1].device == timesteps.device
        torch.testing.assert_close(embedder._frequency_cache[1].cpu(), cpu_storage, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
def test_missing_frequency_cache_is_rejected_before_recording(monkeypatch):
    embedder = _embedder().cuda()
    embedder.cache_frequencies = True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="Prepare timestep frequencies"):
        embedder(torch.ones(1, device="cuda"))
    assert embedder._frequency_cache is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
def test_gpu_frequency_cache_is_exact_and_capture_safe():
    embedder = _embedder().cuda()
    timesteps = torch.tensor([0., 0.125, 499.5, 1000.], device="cuda")
    with torch.no_grad():
        reference = embedder(timesteps)
        embedder.cache_frequencies = True
        torch.testing.assert_close(embedder(timesteps), reference, atol=0, rtol=0)
        storage = embedder._frequency_cache[1]
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                embedder(timesteps)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            result = embedder(timesteps)
        graph.replay()
        torch.testing.assert_close(result, reference, atol=0, rtol=0)
        assert embedder._frequency_cache[1] is storage
