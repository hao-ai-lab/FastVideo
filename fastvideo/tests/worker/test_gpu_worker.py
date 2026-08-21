from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ForwardBatch
from fastvideo.worker.gpu_worker import Worker, _log_cuda_device_uuid


def test_cuda_device_uuid_receipt_is_disabled_without_nvtx_profiling(monkeypatch) -> None:
    """Avoid NVIDIA property access during ordinary worker initialization."""
    get_device_properties = Mock()
    monkeypatch.setenv("FASTVIDEO_NVTX_PROFILE", "0")
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)

    _log_cuda_device_uuid(0, torch.device("cuda:0"))

    get_device_properties.assert_not_called()


def test_cuda_device_uuid_receipt_identifies_profiled_worker(monkeypatch) -> None:
    """Bind one profiled worker rank to its NVIDIA device UUID in logs."""
    log_info = Mock()
    monkeypatch.setenv("FASTVIDEO_NVTX_PROFILE", "1")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="device-uuid"))
    monkeypatch.setattr("fastvideo.worker.gpu_worker.logger.info", log_info)

    _log_cuda_device_uuid(2, torch.device("cuda:0"))

    log_info.assert_called_once_with(
        "Worker %d CUDA device UUID: GPU-%s",
        2,
        "device-uuid",
        local_main_process_only=False,
    )


def _worker_returning(output_batch: ForwardBatch) -> Worker:
    worker = Worker.__new__(Worker)
    worker.fastvideo_args = SimpleNamespace()
    worker.pipeline = SimpleNamespace(forward=lambda batch, args: output_batch)
    return worker


def test_execute_forward_drops_metadata_only_output_before_transport():
    output = torch.ones((1, 3, 2, 4, 4))
    output_batch = ForwardBatch(data_type="video", output=output)
    worker = _worker_returning(output_batch)
    request_batch = ForwardBatch(data_type="video", save_video=False, return_frames=False)

    result = worker.execute_forward(request_batch, FastVideoArgs(model_path="test"))

    assert result.output is not None
    assert result.output.device.type == "cpu"
    assert result.output.numel() == 0


def test_execute_forward_preserves_missing_metadata_only_output():
    output_batch = ForwardBatch(data_type="video", output=None)
    worker = _worker_returning(output_batch)
    request_batch = ForwardBatch(data_type="video", save_video=False, return_frames=False)

    result = worker.execute_forward(request_batch, FastVideoArgs(model_path="test"))

    assert result.output is None


def test_execute_forward_drops_save_only_latent_output():
    output = torch.ones((1, 16, 1, 2, 2))
    output_batch = ForwardBatch(data_type="video", output=output)
    worker = _worker_returning(output_batch)
    request_batch = ForwardBatch(data_type="video", save_video=True, return_frames=False)

    result = worker.execute_forward(request_batch, FastVideoArgs(model_path="test", output_type="latent"))

    assert result.output is not None
    assert result.output.device.type == "cpu"
    assert result.output.numel() == 0


def test_execute_forward_drops_save_only_audio_placeholder():
    output = torch.ones((1, 3, 1, 8, 8))
    output_batch = ForwardBatch(data_type="audio", output=output, extra={"audio_only": True})
    worker = _worker_returning(output_batch)
    request_batch = ForwardBatch(data_type="audio", save_video=True, return_frames=False)

    result = worker.execute_forward(request_batch, FastVideoArgs(model_path="test"))

    assert result.output is not None
    assert result.output.device.type == "cpu"
    assert result.output.numel() == 0


@pytest.mark.parametrize(
    ("save_video", "return_frames"),
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_execute_forward_preserves_requested_output(save_video, return_frames):
    output = torch.ones((1, 3, 2, 4, 4))
    output_batch = ForwardBatch(data_type="video", output=output)
    worker = _worker_returning(output_batch)
    request_batch = ForwardBatch(data_type="video", save_video=save_video, return_frames=return_frames)

    result = worker.execute_forward(request_batch, FastVideoArgs(model_path="test"))

    assert result.output is output
