import torch

from fastvideo.utils.communications import all_gather
from fastvideo.utils.parallel_states import nccl_info


def parallel_forward(fn_):

    def wrapTheFunction(_, x, *args, **kwargs):
        if kwargs['parallel']:
            x = torch.chunk(x, nccl_info.sp_size, dim=1)[nccl_info.rank_within_group]

        output = fn_(_, x, *args, **kwargs)

        if kwargs['parallel']:
            output = all_gather(output.contiguous(), dim=1)

        return output

    return wrapTheFunction
