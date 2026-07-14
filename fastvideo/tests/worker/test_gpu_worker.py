from types import SimpleNamespace

import pytest
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ForwardBatch
from fastvideo.worker.gpu_worker import Worker


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
