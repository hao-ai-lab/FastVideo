# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/device_communicators/pymccl.py

# ===================== import region =====================
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from fastvideo.v1.distributed.device_communicators.pymccl_wrapper import (
    MCCLLibrary, buffer_type, musaStream_t, mcclComm_t, mcclDataTypeEnum,
    mcclRedOpTypeEnum, mcclUniqueId)
from fastvideo.v1.distributed.utils import StatelessProcessGroup
from fastvideo.v1.logger import init_logger
from fastvideo.v1.utils import current_stream

logger = init_logger(__name__)


class PyNcclCommunicator:

    def __init__(
        self,
        group: ProcessGroup | StatelessProcessGroup,
        device: int | str | torch.device,
        library_path: str | None = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyNcclCommunicator to. If None,
                it will be bind to f"musa:{local_rank}".
            library_path: the path to the MCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert dist.get_backend(group) != dist.Backend.MCCL, (
                "PyNcclCommunicator should be attached to a non-MCCL group.")
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            return
        try:
            self.mccl = MCCLLibrary(library_path)
        except Exception:
            # disable because of missing MCCL library
            # e.g. in a non-GPU environment
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        logger.info("FastVideo is using mccl==%s", self.mccl.mcclGetVersion())

        if self.rank == 0:
            # get the unique id from MCCL
            self.unique_id = self.mccl.mcclGetUniqueId()
        else:
            # construct an empty unique id
            self.unique_id = mcclUniqueId()

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            # arg `src` in `broadcast` is the global rank
            dist.broadcast(tensor, src=ranks[0], group=group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)
        if isinstance(device, int):
            device = torch.device(f"musa:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        # mccl communicator and stream will use this device
        # `torch.musa.device` is a context manager that changes the
        # current musa device to the specified one
        with torch.musa.device(device):
            self.comm: mcclComm_t = self.mccl.mcclCommInitRank(
                self.world_size, self.unique_id, self.rank)

            stream = current_stream()
            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            stream.synchronize()
            del data

    def all_reduce(self,
                   in_tensor: torch.Tensor,
                   op: ReduceOp = ReduceOp.SUM,
                   stream=None) -> torch.Tensor:
        if self.disabled:
            return None
        # mccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert in_tensor.device == self.device, (
            f"this mccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}")

        out_tensor = torch.empty_like(in_tensor)

        if stream is None:
            stream = current_stream()
        self.mccl.mcclAllReduce(buffer_type(in_tensor.data_ptr()),
                                buffer_type(out_tensor.data_ptr()),
                                in_tensor.numel(),
                                mcclDataTypeEnum.from_torch(in_tensor.dtype),
                                mcclRedOpTypeEnum.from_torch(op), self.comm,
                                musaStream_t(stream.musa_stream))
        return out_tensor

    def all_gather(self,
                   output_tensor: torch.Tensor,
                   input_tensor: torch.Tensor,
                   stream=None):
        if self.disabled:
            return
        # mccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this mccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")
        if stream is None:
            stream = current_stream()
        self.mccl.mcclAllGather(buffer_type(input_tensor.data_ptr()),
                                buffer_type(output_tensor.data_ptr()),
                                input_tensor.numel(),
                                mcclDataTypeEnum.from_torch(input_tensor.dtype),
                                self.comm, musaStream_t(stream.musa_stream))

    def reduce_scatter(self,
                       output_tensor: torch.Tensor,
                       input_tensor: torch.Tensor,
                       op: ReduceOp = ReduceOp.SUM,
                       stream=None):
        if self.disabled:
            return
        # mccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this mccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")
        if stream is None:
            stream = current_stream()
        self.mccl.mcclReduceScatter(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()), output_tensor.numel(),
            mcclDataTypeEnum.from_torch(input_tensor.dtype),
            mcclRedOpTypeEnum.from_torch(op), self.comm,
            musaStream_t(stream.musa_stream))

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this mccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()
        self.mccl.mcclSend(buffer_type(tensor.data_ptr()), tensor.numel(),
                           mcclDataTypeEnum.from_torch(tensor.dtype), dst,
                           self.comm, musaStream_t(stream.musa_stream))

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this mccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()
        self.mccl.mcclRecv(buffer_type(tensor.data_ptr()), tensor.numel(),
                           mcclDataTypeEnum.from_torch(tensor.dtype), src,
                           self.comm, musaStream_t(stream.musa_stream))

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this mccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()
        if src == self.rank:
            sendbuff = buffer_type(tensor.data_ptr())
            # MCCL requires the sender also to have a receive buffer
            recvbuff = buffer_type(tensor.data_ptr())
        else:
            sendbuff = buffer_type()
            recvbuff = buffer_type(tensor.data_ptr())
        self.mccl.mcclBroadcast(sendbuff, recvbuff, tensor.numel(),
                                mcclDataTypeEnum.from_torch(tensor.dtype), src,
                                self.comm, musaStream_t(stream.musa_stream))
