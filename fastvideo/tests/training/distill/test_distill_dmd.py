import os

import fastvideo.envs as envs

envs.setdefault_external("MASTER_ADDR", "localhost")
envs.setdefault_external("MASTER_PORT", "29512")
import sys
import subprocess
from pathlib import Path
import torch
import json
from huggingface_hub import snapshot_download
from fastvideo.utils import logger
# Import the training pipeline
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from fastvideo.training.wan_training_pipeline import main
from fastvideo.api.training_schema import resolve_training_config
from fastvideo.training.wan_distillation_pipeline import WanDistillationPipeline

wandb_name = "test_distill_dmd"

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "2"


def run_worker():
    """Worker function that will be run on each GPU"""
    # Set the arguments as they are in finetune_v1_test.sh
    training_run_config = {
        "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "mode": "distillation",
        "engine": {
            "num_gpus": 2,
            "parallelism": {
                "sp_size": 2,
                "tp_size": 1,
                "hsdp_replicate_dim": 1,
                "hsdp_shard_dim": 2,
            },
            "precision": {
                "dit": "fp32",
            },
        },
        "pipeline": {
            "flow_shift": 8.0,
            "dmd_denoising_steps": [1000, 757, 522],
        },
        "training": {
            "data": {
                "data_path": "data/crush-smol_processed_t2v/combined_parquet_dataset",
                "train_batch_size": 1,
                "num_latent_t": 4,
                "train_sp_batch_size": 1,
                "dataloader_num_workers": 1,
                "training_cfg_rate": 0.0,
                "num_height": 480,
                "num_width": 832,
                "num_frames": 13,
            },
            "optimizer": {
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
            },
            "loop": {
                "gradient_accumulation_steps": 2,
                "max_train_steps": 2,
            },
            "checkpoint": {
                "output_dir": "data/wan_finetune_test",
                "training_state_checkpointing_steps": 30,
                "weight_only_checkpointing_steps": 30,
                "checkpoints_total_limit": 3,
            },
            "tracker": {
                "project_name": "wan_finetune_ci",
                "run_name": wandb_name,
            },
            "validation": {
                "enabled": True,
                "dataset_file": "examples/training/finetune/wan_t2v_1.3B/crush_smol/validation.json",
                "every_steps": 10,
                "sampling_steps": [3],
            },
            "distillation": {
                "real_score_model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "fake_score_model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "generator_update_interval": 5,
                "min_timestep_ratio": 0.02,
                "max_timestep_ratio": 0.98,
                "real_score_guidance_scale": 3.5,
            },
            "ema": {
                "start_step": 0,
            },
            "model": {
                "enable_gradient_checkpointing_type": "full",
            },
        },
    }
    resolved_config = resolve_training_config(training_run_config)
    # Call the main training function
    pipeline = WanDistillationPipeline.from_pretrained(resolved_config.model_path, resolved_config=resolved_config)
    resolved_config = pipeline.resolved_config
    pipeline.train()
    logger.info("Training pipeline done")


def test_distributed_training():
    """Test the distributed training setup"""
    data_dir = Path("data/crush-smol_processed_t2v")

    if not data_dir.exists():
        print(f"Downloading test dataset to {data_dir}...")
        snapshot_download(repo_id="wlsaidhi/crush-smol_processed_t2v",
                          local_dir=str(data_dir),
                          repo_type="dataset",
                          local_dir_use_symlinks=False)

    # Get the current file path
    current_file = Path(__file__).resolve()

    # Run torchrun command
    cmd = [
        "torchrun", "--nnodes", NUM_NODES, "--nproc_per_node", NUM_GPUS_PER_NODE, "--master_port",
        os.environ["MASTER_PORT"],
        str(current_file)
    ]
    with envs.override_external("WANDB_MODE", "offline"):
        process = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr for debugging
    if process.stdout:
        print("STDOUT:", process.stdout)
    if process.stderr:
        print("STDERR:", process.stderr)

    # Check if the process failed
    if process.returncode != 0:
        print(f"Process failed with return code: {process.returncode}")
        raise subprocess.CalledProcessError(process.returncode, cmd, process.stdout, process.stderr)


if __name__ == "__main__":
    if os.environ.get("LOCAL_RANK") is not None:
        # We're being run by torchrun
        run_worker()
    else:
        # We're being run directly
        test_distributed_training()
