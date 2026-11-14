# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from fastvideo.distributed.device_communicators import pyhccl
from fastvideo.distributed.device_communicators.pyhccl import PyHcclCommunicator
from fastvideo.distributed.utils import StatelessProcessGroup


@pytest.mark.parametrize("operation", ["all_reduce", "broadcast"])
@pytest.mark.parametrize("explicit_stream", [False, True])
def test_pyhccl_uses_communicator_device_current_stream(monkeypatch, operation, explicit_stream):
    stream = SimpleNamespace(npu_stream=42)
    current_stream = Mock(return_value=stream)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=current_stream), raising=False)
    communicator = object.__new__(PyHcclCommunicator)
    communicator.disabled = False
    communicator.device = torch.device("cpu")
    communicator.comm = 1
    communicator.hccl = SimpleNamespace(hcclAllReduce=Mock(), hcclBroadcast=Mock())
    tensor = torch.ones(4)

    kwargs = {"stream": stream} if explicit_stream else {}
    if operation == "all_reduce":
        communicator.all_reduce(tensor, **kwargs)
        call = communicator.hccl.hcclAllReduce.call_args
    else:
        communicator.broadcast(tensor, src=0, **kwargs)
        call = communicator.hccl.hcclBroadcast.call_args

    if explicit_stream:
        current_stream.assert_not_called()
    else:
        current_stream.assert_called_once_with(communicator.device)
    assert call.args[-1].value == stream.npu_stream


def test_pyhccl_warmup_uses_and_synchronizes_communicator_device_stream(monkeypatch):
    stream = SimpleNamespace(npu_stream=42, synchronize=Mock())
    current_stream = Mock(return_value=stream)
    device_context = Mock(side_effect=lambda device: nullcontext())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=current_stream, device=device_context), raising=False)
    hccl = SimpleNamespace(hcclGetUniqueId=Mock(return_value=object()),
                           hcclCommInitRank=Mock(return_value=1), hcclAllReduce=Mock())
    monkeypatch.setattr(pyhccl, "HCCLLibrary", Mock(return_value=hccl))
    group = StatelessProcessGroup(rank=0, world_size=2, store=Mock())
    monkeypatch.setattr(group, "broadcast_obj", lambda value, src: value)
    device = torch.device("cpu")

    communicator = PyHcclCommunicator(group, device)

    assert communicator.available
    assert current_stream.call_args_list == [call(device), call(device)]
    assert device_context.call_args_list == [call(device), call(device)]
    hccl.hcclCommInitRank.assert_called_once_with(2, communicator.unique_id, 0)
    hccl.hcclAllReduce.assert_called_once()
    assert hccl.hcclAllReduce.call_args.args[-1].value == stream.npu_stream
    stream.synchronize.assert_called_once_with()
