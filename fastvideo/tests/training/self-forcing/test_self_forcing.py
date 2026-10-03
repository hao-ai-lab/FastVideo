import os

import fastvideo.envs as envs

envs.setdefault_external("MASTER_ADDR", "localhost")
envs.setdefault_external("MASTER_PORT", "29513")
import sys
import subprocess
from pathlib import Path
import torch
import json
from huggingface_hub import snapshot_download
from fastvideo.utils import logger
# Import the training pipeline
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from fastvideo.training.wan_self_forcing_distillation_pipeline import WanSelfForcingDistillationPipeline
from fastvideo.api.training_schema import resolve_training_config

wandb_name = "test_self_forcing_distill"

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "2"


def run_worker():
    """Worker function that will be run on each GPU"""
    # Set the arguments based on the distill_dmd_t2v_1.3B.sh script
    training_run_config = {
        "model_path": "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers",
        "mode": "distillation",
        "engine": {
            "num_gpus": 2,
            "parallelism": {
                "sp_size": 1,
                "tp_size": 1,
                "hsdp_replicate_dim": 1,
                "hsdp_shard_dim": 2,
            },
            "precision": {
                "dit": "fp32",
            },
        },
        "pipeline": {
            "flow_shift": 5.0,
            "dmd_denoising_steps": [1000, 750, 500],  # Reduced steps for testing
        },
        "training": {
            "data": {
                "data_path": "data/crush-smol_processed_t2v/combined_parquet_dataset",
                "train_batch_size": 1,
                "num_latent_t": 21,
                "train_sp_batch_size": 1,
                "dataloader_num_workers": 1,
                "training_cfg_rate": 0.0,
                "num_height": 480,
                "num_width": 832,
                "num_frames": 21,
            },
            "optimizer": {
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
            },
            "loop": {
                "gradient_accumulation_steps": 1,
                "max_train_steps": 2,
            },
            "checkpoint": {
                "output_dir": "data/wan_self_forcing_test",
                "training_state_checkpointing_steps": 30,
                "weight_only_checkpointing_steps": 30,
                "checkpoints_total_limit": 3,
            },
            "tracker": {
                "project_name": "wan_self_forcing_ci",
                "run_name": wandb_name,
            },
            "validation": {
                "enabled": True,
                "dataset_file": "examples/training/finetune/wan_t2v_1.3B/crush_smol/validation.json",
                "every_steps": 10,
                "sampling_steps": [3],
                "log_visualization": True,
            },
            # DMD args
            "distillation": {
                "real_score_model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "fake_score_model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "min_timestep_ratio": 0.02,
                "max_timestep_ratio": 0.98,
                "real_score_guidance_scale": 3.0,
                "fake_score_learning_rate": 8e-6,
                "fake_score_betas": [0.0, 0.999],
                "warp_denoising_step": True,
                "simulate_generator_forward": True,
            },
            # Self-forcing specific args
            "self_forcing": {
                "dfake_gen_update_ratio": 5,
                "num_frame_per_block": 3,
                "independent_first_frame": False,
                "same_step_across_blocks": True,
                "last_step_only": False,
                "context_noise": 0,
            },
            "ema": {
                "enabled": True,
                "decay": 0.99,
                "start_step": 100,
            },
            "model": {
                "enable_gradient_checkpointing_type": "full",
            },
        },
    }
    resolved_config = resolve_training_config(training_run_config)

    # Call the main training function
    pipeline = WanSelfForcingDistillationPipeline.from_pretrained(resolved_config.model_path,
                                                                  resolved_config=resolved_config)
    resolved_config = pipeline.resolved_config
    pipeline.train()
    logger.info("Self-forcing distillation training pipeline done")


def test_distributed_training():
    """Test the distributed self-forcing training setup"""
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
