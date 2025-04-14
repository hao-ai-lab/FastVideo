# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/device_communicators/base_device_communicator.py

from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


class DeviceCommunicatorBase:
    """
    Base class for device-specific communicator.
    It can use the `cpu_group` to initialize the communicator.
    If the device has PyTorch integration (PyTorch can recognize its
    communication backend), the `device_group` will also be given.
    """

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        self.device = device or torch.device("cpu")
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name
        self.rank = dist.get_rank(cpu_group)
        self.world_size = dist.get_world_size(cpu_group)
        self.ranks = dist.get_process_group_ranks(cpu_group)
        self.global_rank = dist.get_rank()
        self.global_world_size = dist.get_world_size()
        self.rank_in_group = dist.get_group_rank(self.cpu_group,
                                                 self.global_rank)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size, ) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(output_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # All-gather.
        dist.all_gather_into_tensor(output_tensor,
                                    input_,
                                    group=self.device_group)
        # Reshape
        output_tensor = output_tensor.reshape((self.world_size, ) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (self.world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor

    def gather(self,
               input_: torch.Tensor,
               dst: int = 0,
               dim: int = -1) -> Optional[torch.Tensor]:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # Gather.
        torch.distributed.gather(input_,
                                 gather_list,
                                 dst=self.ranks[dst],
                                 group=self.device_group)
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def all_to_all_4D(self,
                      input_: torch.Tensor,
                      scatter_dim: int = 2,
                      gather_dim: int = 1) -> torch.Tensor:
        """Specialized all-to-all operation for 4D tensors (e.g., for QKV matrices).
        
        Args:
            input_ (torch.Tensor): 4D input tensor to be scattered and gathered.
            scatter_dim (int, optional): Dimension along which to scatter. Defaults to 2.
            gather_dim (int, optional): Dimension along which to gather. Defaults to 1.
            
        Returns:
            torch.Tensor: Output tensor after all-to-all operation.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_

        assert input_.dim(
        ) == 4, f"input must be 4D tensor, got {input_.dim()} and shape {input_.shape}"

        if scatter_dim == 2 and gather_dim == 1:
            # input: (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
            bs, shard_seqlen, hc, hs = input_.shape
            seqlen = shard_seqlen * self.world_size
            shard_hc = hc // self.world_size

            # Reshape and transpose for scattering
            input_t = (input_.reshape(bs, shard_seqlen, self.world_size,
                                      shard_hc, hs).transpose(0,
                                                              2).contiguous())

            output = torch.empty_like(input_t)

            torch.distributed.all_to_all_single(output,
                                                input_t,
                                                group=self.device_group)
            torch.cuda.synchronize()

            # Reshape and transpose back
            output = output.reshape(seqlen, bs, shard_hc,
                                    hs).transpose(0, 1).contiguous().reshape(
                                        bs, seqlen, shard_hc, hs)

            return output

        elif scatter_dim == 1 and gather_dim == 2:
            # input: (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
            bs, seqlen, shard_hc, hs = input_.shape
            hc = shard_hc * self.world_size
            shard_seqlen = seqlen // self.world_size

            # Reshape and transpose for scattering
            input_t = (input_.reshape(bs, self.world_size, shard_seqlen,
                                      shard_hc, hs).transpose(0, 3).transpose(
                                          0, 1).contiguous().reshape(
                                              self.world_size, shard_hc,
                                              shard_seqlen, bs, hs))
            output = torch.empty_like(input_t)

            torch.distributed.all_to_all_single(output,
                                                input_t,
                                                group=self.device_group)
            torch.cuda.synchronize()

            # Reshape and transpose back
            output = output.reshape(hc, shard_seqlen, bs,
                                    hs).transpose(0, 2).contiguous().reshape(
                                        bs, shard_seqlen, hc, hs)

            return output
        else:
            raise RuntimeError(
                "scatter_dim must be 1 or 2 and gather_dim must be 1 or 2")

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(self,
             size: torch.Size,
             dtype: torch.dtype,
             src: Optional[int] = None) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self) -> None:
        pass
