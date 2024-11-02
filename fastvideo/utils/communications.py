import torch
import torch.distributed as dist
from einops import rearrange
from fastvideo.utils.parallel_states import nccl_info

def broadcast(input_: torch.Tensor):
    sp_size = nccl_info.world_size
    src = nccl_info.rank // sp_size * sp_size
    dist.broadcast(input_, src=src, group=nccl_info.group)

_COUNT = 0
def _single_all_to_all(
    input_: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    enable_HCCL=False,
):
    assert scatter_dim == 0 or scatter_dim == 1, "scatter_dim should be 0 or 1"
    sp_size = nccl_info.world_size
    inp_shape = list(input_.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // sp_size
    if scatter_dim < 1:
        input_t = input_.reshape(
            [sp_size, inp_shape[scatter_dim]] + \
            inp_shape[scatter_dim + 1:]
        )
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        input_t = input_.reshape(
            [-1, sp_size, inp_shape[scatter_dim]] + \
            inp_shape[scatter_dim + 1:]
        ).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=nccl_info.group)
    # if scattering the seq-dim, transpose the heads back to the original dimension
    if scatter_dim < 1:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[: gather_dim] + [inp_shape[gather_dim] * sp_size, ] + inp_shape[gather_dim + 1:])


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, scatter_dim, gather_dim, all_to_all_func):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.all_to_all = all_to_all_func
        output = ctx.all_to_all(input_, scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = ctx.all_to_all(
            grad_output,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )



def all_to_all_SBH(
    input_: torch.Tensor,
    scatter_dim: int = 1,
    gather_dim: int = 0,
):
    return _AllToAll.apply(input_, scatter_dim, gather_dim, _single_all_to_all)



class _AllGather(torch.autograd.Function):
    """All-gather communication with autograd support.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        world_size = nccl_info.world_size
        group = nccl_info.group
        input_size = list(input_.size())

        ctx.input_size = input_size[dim]

        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        input_ = input_.contiguous()
        dist.all_gather(tensor_list, input_, group=group)

        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size = nccl_info.world_size
        rank = nccl_info.rank
        dim = ctx.dim
        input_size = ctx.input_size

        sizes = [input_size] * world_size

        grad_input_list = torch.split(grad_output, sizes, dim=dim)
        grad_input = grad_input_list[rank]

        return grad_input, None

def all_gather_BHSD(input_: torch.Tensor, dim: int = 1):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather.apply(input_, dim)


def prepare_parallel_data(hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask):
    def all_to_all(hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask):
        hidden_states = _single_all_to_all(hidden_states, scatter_dim=2, gather_dim=0, enable_HCCL=True)
        encoder_hidden_states = _single_all_to_all(encoder_hidden_states, scatter_dim=1, gather_dim=0, enable_HCCL=True)
        attention_mask = _single_all_to_all(attention_mask, scatter_dim=1, gather_dim=0, enable_HCCL=True)
        encoder_attention_mask = _single_all_to_all(encoder_attention_mask, scatter_dim=1, gather_dim=0, enable_HCCL=True)
        return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask

    sp_size = nccl_info.world_size
    frame = hidden_states.shape[2]
    assert frame % sp_size == 0, "frame should be a multiple of sp_size"

    encoder_hidden_states = rearrange(encoder_hidden_states, 'b 1 (n x) h -> b n x h',
                                     n=sp_size, x=encoder_hidden_states.shape[2]//sp_size).contiguous()
    hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask = all_to_all(hidden_states,
                                                                                            encoder_hidden_states,
                                                                                            attention_mask.repeat(1, sp_size, 1, 1),
                                                                                            encoder_attention_mask.repeat(1, sp_size, 1))

    return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask