import os
import shutil

import pytest
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from fastvideo.models.mochi_hf.modeling_mochi import MochiTransformer3DModel
from fastvideo.utils.checkpoint import save_checkpoint
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs


@pytest.fixture(scope="module", autouse=True)
def setup_distributed():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"

    dist.init_process_group("nccl")
    yield
    dist.destroy_process_group()


def test_save_and_remove_checkpoint():
    transformer = MochiTransformer3DModel(num_layers=0)
    fsdp_kwargs, _ = get_dit_fsdp_kwargs(transformer, "none")
    transformer = FSDP(transformer, **fsdp_kwargs)

    test_folder = "./test_checkpoint"
    save_checkpoint(transformer, 0, test_folder, 0)

    assert os.path.exists(test_folder), "Checkpoint folder was not created."

    shutil.rmtree(test_folder)
    assert not os.path.exists(test_folder), "Checkpoint folder still exists."
