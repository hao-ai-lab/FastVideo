import torch
import torch.distributed as dist
from einops import rearrange
from fastvideo.utils.parallel_states import nccl_info

def broadcast(input_: torch.Tensor):
    sp_size = nccl_info.world_size
    src = nccl_info.rank // sp_size * sp_size
    dist.broadcast(input_, src=src, group=nccl_info.group)
    
    
def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(input_, ctx.world_size, process_group, scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, nccl_info.group, scatter_dim, gather_dim)



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

def all_gather(input_: torch.Tensor, dim: int = 1):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather.apply(input_.contiguous(), dim)


def prepare_parallel_data(hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask):
    def prepare(hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask):
        
        # hidden_states
        # bs = hidden_states.shape[0] * world_size
        # hidden_states = rearrange(hidden_states, 'b c s h w -> (b c) s h w')
        # hidden_states = _single_all_to_all(hidden_states, scatter_dim=1, gather_dim=0)
        # hidden_states = rearrange(hidden_states, '(b c) s h w -> b c s h w', b=bs)
        hidden_states = all_to_all(hidden_states, scatter_dim=2, gather_dim=0)
        encoder_hidden_states = all_to_all(encoder_hidden_states, scatter_dim=1, gather_dim=0)
        attention_mask = all_to_all(attention_mask, scatter_dim=1, gather_dim=0)
        encoder_attention_mask = all_to_all(encoder_attention_mask, scatter_dim=1, gather_dim=0)
        return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask

    sp_size = nccl_info.world_size
    frame = hidden_states.shape[2]
    assert frame % sp_size == 0, "frame should be a multiple of sp_size"
    # print all share
    # if nccl_info.rank == 0:
    #     # torch.Size([1, 12, 28, 60, 106]) torch.Size([1, 256, 4096]) torch.Size([1, 28, 60, 106]) torch.Size([1, 256])
    #     print(hidden_states.shape, encoder_hidden_states.shape, attention_mask.shape, encoder_attention_mask.shape)
    # encoder_hidden_states = rearrange(encoder_hidden_states, 'b (n x) h -> b n x h',
    #                                  n=sp_size, x=encoder_hidden_states.shape[1]//sp_size).contiguous()
    # if nccl_info.rank == 0:
    #     print(encoder_attention_mask.tolist())
    # if nccl_info.rank == 0:
    #     print("-------")
    #     rank_1_sum = torch.mean(hidden_states[:, :, hidden_states.shape[2]//nccl_info.world_size : hidden_states.shape[2]//nccl_info.world_size*2, :, :])
        # print("ok", rank_1_sum)
        # print(encoder_hidden_states.mean())
    hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask = prepare(hidden_states,
                                                                                            encoder_hidden_states.repeat(1, sp_size,  1),
                                                                                            attention_mask.repeat(1, sp_size, 1, 1),
                                                                                            encoder_attention_mask.repeat(1, sp_size))
    # if nccl_info.rank == 0:
    #     # torch.Size([4, 12, 7, 60, 106]) torch.Size([4, 256, 4096]) torch.Size([4, 28, 60, 106]) torch.Size([4, 256])
    #     print(hidden_states.shape, encoder_hidden_states.shape, attention_mask.shape, encoder_attention_mask.shape)
    #     assert False

    return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask