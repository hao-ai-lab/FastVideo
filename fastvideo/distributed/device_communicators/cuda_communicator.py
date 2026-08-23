# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/device_communicators/cuda_communicator.py

import torch
from torch.distributed import ProcessGroup

from fastvideo.distributed.device_communicators.base_device_communicator import (DeviceCommunicatorBase)
from fastvideo.distributed.device_communicators.ulysses_a2a import maybe_create_helper


class CudaCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: torch.device | None = None,
                 device_group: ProcessGroup | None = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)

        from fastvideo.distributed.device_communicators.pynccl import (PyNcclCommunicator)

        self.pynccl_comm: PyNcclCommunicator | None = None
        if self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        # Capability is agreed once; the persistent window arms on first use.
        self.ulysses_a2a = maybe_create_helper(self.cpu_group, self.device_group, self.world_size, self.device,
                                               self.pynccl_comm)

    def all_reduce(self, input_, op: torch.distributed.ReduceOp | None = None):
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        out = pynccl_comm.all_reduce(input_, op=op)
        if out is None:
            # fall back to the default all-reduce using PyTorch.
            # this usually happens during testing.
            # when we run the model, allreduce only happens for the TP
            # group, where we always have either custom allreduce or pynccl.
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group, op=op)
        return out

    def all_to_all_4D(self, input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1) -> torch.Tensor:
        """All-to-all over the sequence parallel group, fused when available."""
        if self.ulysses_a2a is not None:
            output = self.ulysses_a2a.try_all_to_all_4D(input_, scatter_dim, gather_dim)
            if output is not None:
                return output
        return super().all_to_all_4D(input_, scatter_dim, gather_dim)

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(self, size: torch.Size, dtype: torch.dtype, src: int | None = None) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self) -> None:
        if self.ulysses_a2a is not None:
            # The helper still needs its NCCL communicator during teardown.
            self.ulysses_a2a.close()
            self.ulysses_a2a = None
        if self.pynccl_comm is not None:
            self.pynccl_comm = None
