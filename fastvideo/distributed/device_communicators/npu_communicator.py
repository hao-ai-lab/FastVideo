import torch
from torch.distributed import ProcessGroup

from fastvideo.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase)


class NpuCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: torch.device | None = None,
                 device_group: ProcessGroup | None = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)

        from fastvideo.distributed.device_communicators.pyhccl import (
            PyHcclCommunicator)

        self.pyhccl_comm: PyHcclCommunicator | None = None
        if self.world_size > 1:
            self.pyhccl_comm = PyHcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

    def all_reduce(self, input_, op: torch.distributed.ReduceOp | None = None):
        pyhccl_comm = self.pyhccl_comm
        assert pyhccl_comm is not None, "pyhccl_comm should not be None"
        out = pyhccl_comm.all_reduce(input_, op=op)
        if out is None:
            # fall back to the default all-reduce using PyTorch.
            # this usually happens during testing.
            # when we run the model, allreduce only happens for the TP
            # group, where we always have either custom allreduce or pyhccl.
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group, op=op)
        return out

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is not None and not pyhccl_comm.disabled:
            pyhccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(self,
             size: torch.Size,
             dtype: torch.dtype,
             src: int | None = None) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is not None and not pyhccl_comm.disabled:
            pyhccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self) -> None:
        if self.pyhccl_comm is not None:
            self.pyhccl_comm = None
