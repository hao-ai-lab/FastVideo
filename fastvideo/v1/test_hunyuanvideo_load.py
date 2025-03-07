import os
import torch
import torch.nn as nn
import argparse
import numpy as np

from fastvideo.logger import init_logger
from fastvideo.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    get_sequence_model_parallel_rank,
    get_sequence_model_parallel_world_size,
    destroy_model_parallel,
    destroy_distributed_environment,
    cleanup_dist_env_and_memory
)
import json
from fastvideo.models.dits.hunyuanvideo import HunyuanVideoDiT
from fastvideo.models.hunyuan.modules.models import HYVideoDiffusionTransformer
from fastvideo.loader.fsdp_load import load_fsdp_model
import glob
logger = init_logger(__name__)


def setup_args():
    parser = argparse.ArgumentParser(description='Distributed HunyuanVideo Test')
    parser.add_argument('--sequence_model_parallel_size', type=int, default=1,
                        help='Degree of sequence model parallelism')
    return parser.parse_args()

def test_hunyuanvideo_distributed():
    args = setup_args()
    
    # Initialize distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    logger.info(f"Initializing process: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    
    # Initialize distributed environment
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank
    )
    
    # Initialize tensor model parallel groups
    initialize_model_parallel(
        sequence_model_parallel_size=args.sequence_model_parallel_size
    )
    # load data/hunyuanvideo_community/transformer/config.json
    with open("/mbz/users/hao.zhang/peiyuan/FastVideo/data/hunyuanvideo_community/transformer/config.json", "r") as f:
        config = json.load(f)
    # remove   "_class_name": "HunyuanVideoTransformer3DModel",   "_diffusers_version": "0.32.0.dev0",
    # TODO: write normalize config function
    config.pop("_class_name")
    config.pop("_diffusers_version")
    # load data/hunyuanvideo_community/transformer/*.safetensors
    weight_dir_list = glob.glob("/mbz/users/hao.zhang/peiyuan/FastVideo/data/hunyuanvideo_community/transformer/*.safetensors")
    # to str
    torch.cuda.set_device(f"cuda:{local_rank}")
    weight_dir_list = [str(path) for path in weight_dir_list]
    model = load_fsdp_model(
        model_name="HunyuanVideoTransformer3DModel",
        init_params=config,
        weight_dir_list=weight_dir_list,
        device=torch.device(f"cuda:{local_rank}"),
        cpu_offload=False
    )
    
if __name__ == "__main__":
    test_hunyuanvideo_distributed()