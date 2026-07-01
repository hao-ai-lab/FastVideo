from types import SimpleNamespace

import pytest
import torch

from fastvideo.worker.multiproc_executor import _prepare_worker_output_for_parent


def test_prepare_worker_output_keeps_non_latent_tensor_on_device():
    result = torch.empty((1, 2))
    args = SimpleNamespace(output_type="pil")

    assert _prepare_worker_output_for_parent(result, args) is result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_prepare_worker_output_moves_latent_cuda_tensor_to_cpu():
    result = torch.empty((1, 16, 1, 8, 8), device="cuda")
    args = SimpleNamespace(output_type="latent")

    prepared = _prepare_worker_output_for_parent(result, args)

    assert prepared.device.type == "cpu"
    assert prepared.shape == result.shape
