import torch

from fastvideo.utils.communications import all_gather
from fastvideo.utils.parallel_states import nccl_info


def parallel_forward(fn_):

    def wrapTheFunction(_, hidden_states, *args, **kwargs):
        if kwargs['parallel']:
            hidden_states = torch.chunk(hidden_states, nccl_info.sp_size, dim=-2)[nccl_info.rank_within_group]
            kwargs['attn_mask'] = torch.chunk(kwargs['attn_mask'], nccl_info.sp_size,
                                              dim=-2)[nccl_info.rank_within_group]
        output = fn_(_, hidden_states, *args, **kwargs)

        if kwargs['parallel']:
            output = all_gather(output.contiguous(), dim=-2)

        return output

    return wrapTheFunction
