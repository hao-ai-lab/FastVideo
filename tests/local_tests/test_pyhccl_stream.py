# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from fastvideo.distributed.device_communicators.pyhccl import PyHcclCommunicator


@pytest.mark.parametrize("operation", ["all_reduce", "broadcast"])
def test_pyhccl_uses_communicator_device_current_stream(monkeypatch, operation):
    stream = SimpleNamespace(npu_stream=42)
    current_stream = Mock(return_value=stream)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=current_stream), raising=False)
    communicator = object.__new__(PyHcclCommunicator)
    communicator.disabled = False
    communicator.device = torch.device("cpu")
    communicator.comm = 1
    communicator.hccl = SimpleNamespace(hcclAllReduce=Mock(), hcclBroadcast=Mock())
    tensor = torch.ones(4)

    if operation == "all_reduce":
        communicator.all_reduce(tensor)
        call = communicator.hccl.hcclAllReduce.call_args
    else:
        communicator.broadcast(tensor, src=0)
        call = communicator.hccl.hcclBroadcast.call_args

    current_stream.assert_called_once_with(communicator.device)
    assert call.args[-1].value == stream.npu_stream
