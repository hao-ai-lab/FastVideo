import torch
import torch.distributed as dist

from fastvideo.attention.yunchang.globals import PROCESS_GROUP


def stripe_extract_local(value, rank, world_size, rd, ud, *args, **kwargs):
    # ud at the highest dim
    input_dim = value.dim()
    assert input_dim >= 2

    batch_size, seqlen, *rest = value.shape

    assert dist.get_world_size(group=PROCESS_GROUP.RING_PG) == rd
    assert dist.get_world_size(group=PROCESS_GROUP.ULYSSES_PG) == ud

    value = value.reshape(batch_size, seqlen // rd, rd, -1).contiguous()
    value = value.transpose(1, 2).reshape(batch_size, seqlen, -1).contiguous()
    value = value.chunk(world_size, dim=1)[rank]

    new_shape = [batch_size, seqlen // world_size] + rest
    return value.reshape(new_shape)


def basic_extract_local(value, rank, world_size, *args, **kwargs):
    return value.chunk(world_size, dim=1)[rank].detach().clone()


def zigzag_extract_local(value,
                         rank,
                         world_size,
                         rd,
                         ud,
                         dim=1,
                         *args,
                         **kwargs):
    """
    value is a tensor of shape (bs, seqlen, ...)
    """
    input_dim = value.dim()
    assert input_dim >= 2
    batch_size, seqlen, *rest = value.shape

    value_chunks = value.chunk(2 * rd, dim=dim)
    r_rank = dist.get_rank(group=PROCESS_GROUP.RING_PG)
    u_rank = dist.get_rank(group=PROCESS_GROUP.ULYSSES_PG)

    assert dist.get_world_size(group=PROCESS_GROUP.RING_PG) == rd
    assert dist.get_world_size(group=PROCESS_GROUP.ULYSSES_PG) == ud

    local_value = torch.cat(
        [value_chunks[r_rank], value_chunks[2 * rd - r_rank - 1]],
        dim=dim).chunk(ud, dim=dim)[u_rank]

    new_shape = [batch_size, seqlen // world_size] + rest
    return local_value.reshape(new_shape).contiguous()


EXTRACT_FUNC_DICT = {
    "basic": basic_extract_local,
    "strip": stripe_extract_local,
    "zigzag": zigzag_extract_local,
    "basic_pytorch": basic_extract_local,
    "basic_flashinfer": basic_extract_local,
    "basic_npu": basic_extract_local,
}
