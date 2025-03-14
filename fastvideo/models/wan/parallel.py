import torch

from fastvideo.utils.communications import all_gather
from fastvideo.utils.parallel_states import nccl_info


def parallel_forward(fn_):

    def wrapTheFunction(_, x, **kwargs):
        parallel = kwargs['parallel']
        kwargs.pop('parallel', None)
        if parallel:
            x = torch.chunk(x, nccl_info.sp_size, dim=1)[nccl_info.rank_within_group]

        output = fn_(_, x, **kwargs)

        if parallel:
            output = all_gather(output.contiguous(), dim=1)

        return output

    return wrapTheFunction
