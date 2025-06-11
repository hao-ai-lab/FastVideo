import os
import sys
import subprocess
from pathlib import Path
import torch.distributed.elastic.multiprocessing.errors as errors
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
import torch
import pytest
import wandb
import json

# Import the training pipeline
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from fastvideo.v1.training.wan_training_pipeline import main
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.utils import FlexibleArgumentParser

wandb_name = "test_training_loss"

reference_wandb_summary_file = "fastvideo/v1/tests/training/reference_wandb_summary.json"


def run_worker():
    """Worker function that will be run on each GPU"""
    # Create and populate args
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    
    # Set the arguments as they are in finetune_v1_test.sh
    args = parser.parse_args([
        "--model_path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "--inference_mode", "False",
        "--pretrained_model_name_or_path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "--cache_dir", "/home/ray/.cache",
        "--data_path", "data/crush-smol_parq/combined_parquet_dataset",
        "--validation_prompt_dir", "data/crush-smol_parq/validation_parquet_dataset",
        "--train_batch_size", "4",
        "--num_latent_t", "8",
        "--num_gpus", "8",
        "--sp_size", "4",
        "--tp_size", "4",
        "--hsdp_replicate_dim", "2",
        "--hsdp_shard_dim", "4",
        "--train_sp_batch_size", "1",
        "--dataloader_num_workers", "1",
        "--gradient_accumulation_steps", "1",
        "--max_train_steps", "5",
        "--learning_rate", "1e-6",
        "--mixed_precision", "bf16",
        "--checkpointing_steps", "30",
        "--validation_steps", "10",
        "--validation_sampling_steps", "8",
        "--log_validation",
        "--checkpoints_total_limit", "3",
        "--allow_tf32",
        "--ema_start_step", "0",
        "--cfg", "0.0",
        "--output_dir", "data/wan_finetune_test",
        "--tracker_project_name", "wan_finetune_ci",
        "--wandb_run_name", wandb_name,
        "--num_height", "480",
        "--num_width", "832",
        "--num_frames", "81",
        "--shift", "3",
        "--validation_guidance_scale", "1.0",
        "--num_euler_timesteps", "50",
        "--multi_phased_distill_schedule", "4000-1",
        "--weight_decay", "0.01",
        "--not_apply_cfg_solver",
        "--master_weight_type", "fp32",
        "--max_grad_norm", "1.0"
    ])
    
    # Call the main training function
    main(args)

@pytest.mark.usefixtures("test_dataset_smol_crush")
def test_distributed_training(test_dataset_smol_crush):
    """Test the distributed training setup"""
    os.environ["WANDB_API_KEY"] = "8d9f4b39abd68eb4e29f6fc010b7ee71a2207cde"
    os.environ["WANDB_MODE"] = "online"

    # Get the current file path
    current_file = Path(__file__).resolve()
    
    # Run torchrun command
    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "8",
        str(current_file)
    ]
    
    process = subprocess.run(cmd, check=True)

    wandb_dir = f"wandb/latest-run"
    summary_file = f"{wandb_dir}/files/wandb-summary.json"

    reference_wandb_summary = json.load(open(reference_wandb_summary_file))
    wandb_summary = json.load(open(summary_file))

    fields_and_thresholds = {
        'avg_step_time': 0.1,
        'grad_norm': 0.01,
        'step_time': 0.1,
        'train_loss': 0.001
    }

    for field, threshold in fields_and_thresholds.items():
        ref_value = reference_wandb_summary[field]
        current_value = wandb_summary[field]
        diff = abs(ref_value - current_value)
        print(f"{field}, diff: {diff}, threshold: {threshold}, reference: {ref_value}, current: {current_value}")
        assert diff <= threshold, f"{field} difference {diff} exceeds threshold of {threshold} (reference: {ref_value}, current: {current_value})"


if __name__ == "__main__":
    if os.environ.get("LOCAL_RANK") is not None:
        # We're being run by torchrun
        run_worker()
    else:
        # We're being run directly
        test_distributed_training(None)
