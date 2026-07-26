#!/usr/bin/env python3
import json
import os

import torch
import torch.distributed as dist

from fastvideo.distributed.parallel_state import (
    get_world_group,
    maybe_init_distributed_environment_and_model_parallel,
)


local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
torch.cuda.set_device(local_rank)
maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
dist.all_reduce(value, group=get_world_group().device_group)
torch.cuda.synchronize(local_rank)
print(
    "NCCL_SMOKE "
    + json.dumps(
        {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": dist.get_world_size(),
            "sum": value.item(),
            "host": os.uname().nodename,
        },
        sort_keys=True,
    ),
    flush=True,
)
dist.barrier()
