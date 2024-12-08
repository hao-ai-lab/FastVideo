from typing import Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

import genmo.mochi_preview.dit.joint_model.context_parallel as cp


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def cp_pass_frames(x: torch.Tensor, frames_to_send: int) -> torch.Tensor:
    """
    Forward pass that handles communication between ranks for inference.
    Args:
        x: Tensor of shape (B, C, T, H, W)
        frames_to_send: int, number of frames to communicate between ranks
    Returns:
        output: Tensor of shape (B, C, T', H, W)
    """
    cp_rank, cp_world_size = cp.get_cp_rank_size()
    if frames_to_send == 0 or cp_world_size == 1:
        return x

    group = cp.get_cp_group()
    global_rank = dist.get_rank()

    # Send to next rank
    if cp_rank < cp_world_size - 1:
        assert x.size(2) >= frames_to_send
        tail = x[:, :, -frames_to_send:].contiguous()
        dist.send(tail, global_rank + 1, group=group)

    # Receive from previous rank
    if cp_rank > 0:
        B, C, _, H, W = x.shape
        recv_buffer = torch.empty(
            (B, C, frames_to_send, H, W),
            dtype=x.dtype,
            device=x.device,
        )
        dist.recv(recv_buffer, global_rank - 1, group=group)
        x = torch.cat([recv_buffer, x], dim=2)

    return x


def _pad_to_max(x: torch.Tensor, max_T: int) -> torch.Tensor:
    if max_T > x.size(2):
        pad_T = max_T - x.size(2)
        pad_dims = (0, 0, 0, 0, 0, pad_T)
        return F.pad(x, pad_dims)
    return x


def gather_all_frames(x: torch.Tensor) -> torch.Tensor:
    """
    Gathers all frames from all processes for inference.
    Args:
        x: Tensor of shape (B, C, T, H, W)
    Returns:
        output: Tensor of shape (B, C, T_total, H, W)
    """
    cp_rank, cp_size = cp.get_cp_rank_size()
    if cp_size == 1:
        return x

    cp_group = cp.get_cp_group()

    # Ensure the tensor is contiguous for collective operations
    x = x.contiguous()

    # Get the local time dimension size
    local_T = x.size(2)
    local_T_tensor = torch.tensor([local_T], device=x.device, dtype=torch.int64)

    # Gather all T sizes from all processes
    all_T = [torch.zeros(1, dtype=torch.int64, device=x.device) for _ in range(cp_size)]
    dist.all_gather(all_T, local_T_tensor, group=cp_group)
    all_T = [t.item() for t in all_T]

    # Pad the tensor at the end of the time dimension to match max_T
    max_T = max(all_T)
    x = _pad_to_max(x, max_T).contiguous()

    # Prepare a list to hold the gathered tensors
    gathered_x = [torch.zeros_like(x).contiguous() for _ in range(cp_size)]

    # Perform the all_gather operation
    dist.all_gather(gathered_x, x, group=cp_group)

    # Slice each gathered tensor back to its original T size
    for idx, t_size in enumerate(all_T):
        gathered_x[idx] = gathered_x[idx][:, :, :t_size]

    return torch.cat(gathered_x, dim=2)


def excessive_memory_usage(input: torch.Tensor, max_gb: float = 2.0) -> bool:
    """Estimate memory usage based on input tensor size and data type."""
    element_size = input.element_size()  # Size in bytes of each element
    memory_bytes = input.numel() * element_size
    memory_gb = memory_bytes / 1024**3
    return memory_gb > max_gb


class ContextParallelCausalConv3d(torch.nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]],
        **kwargs,
    ):
        kernel_size = cast_tuple(kernel_size, 3)
        stride = cast_tuple(stride, 3)
        height_pad = (kernel_size[1] - 1) // 2
        width_pad = (kernel_size[2] - 1) // 2

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=(1, 1, 1),
            padding=(0, height_pad, width_pad),
            **kwargs,
        )

    def forward(self, x: torch.Tensor):
        cp_rank, cp_world_size = cp.get_cp_rank_size()

        context_size = self.kernel_size[0] - 1
        if cp_rank == 0:
            mode = "constant" if self.padding_mode == "zeros" else self.padding_mode
            x = F.pad(x, (0, 0, 0, 0, context_size, 0), mode=mode)

        if cp_world_size == 1:
            return super().forward(x)

        if all(s == 1 for s in self.stride):
            # Receive some frames from previous rank.
            x = cp_pass_frames(x, context_size)
            return super().forward(x)

        # Less efficient implementation for strided convs.
        # All gather x, infer and chunk.
        x = gather_all_frames(x)  # [B, C, k - 1 + global_T, H, W]
        x = super().forward(x)
        x_chunks = x.tensor_split(cp_world_size, dim=2)
        assert len(x_chunks) == cp_world_size
        return x_chunks[cp_rank]
