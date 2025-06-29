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
from huggingface_hub import snapshot_download

# Import the training pipeline
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from fastvideo.v1.training.wan_training_pipeline import main
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.utils import FlexibleArgumentParser

wandb_name = "test_training_loss"
a40_reference_wandb_summary_file = "fastvideo/v1/tests/training/Vanilla/a40_reference_wandb_summary.json"
l40s_reference_wandb_summary_file = "fastvideo/v1/tests/training/Vanilla/l40s_reference_wandb_summary.json"

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "4"


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
        "--data_path", "data/crush-smol_processed_t2v/combined_parquet_dataset",
        "--validation_preprocessed_path", "data/crush-smol_processed_t2v/validation_parquet_dataset",
        "--train_batch_size", "2",
        "--num_latent_t", "4",
        "--num_gpus", "4",
        "--sp_size", "4",
        "--tp_size", "4",
        "--hsdp_replicate_dim", "1",
        "--hsdp_shard_dim", "4",
        "--train_sp_batch_size", "1",
        "--dataloader_num_workers", "1",
        "--gradient_accumulation_steps", "2",
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
        "--training_cfg_rate", "0.0",
        "--output_dir", "data/wan_finetune_test",
        "--tracker_project_name", "wan_finetune_ci",
        "--wandb_run_name", wandb_name,
        "--num_height", "480",
        "--num_width", "832",
        "--num_frames", "81",
        "--flow_shift", "3",
        "--validation_guidance_scale", "1.0",
        "--num_euler_timesteps", "50",
        "--multi_phased_distill_schedule", "4000-1",
        "--weight_decay", "0.01",
        "--not_apply_cfg_solver",
        "--dit_precision", "fp32",
        "--max_grad_norm", "1.0"
    ])
    
    # Call the main training function
    main(args)

def test_distributed_training():
    """Test the distributed training setup"""
    os.environ["WANDB_MODE"] = "online"

    data_dir = Path("data/crush-smol_processed_t2v")
    
    if not data_dir.exists():
        print(f"Downloading test dataset to {data_dir}...")
        snapshot_download(
            repo_id="wlsaidhi/crush-smol_processed_t2v",
            local_dir=str(data_dir),
            repo_type="dataset",
            local_dir_use_symlinks=False
        )

    # Get the current file path
    current_file = Path(__file__).resolve()
    
    # Run torchrun command
    cmd = [
        "torchrun",
        "--nnodes", NUM_NODES,
        "--nproc_per_node", NUM_GPUS_PER_NODE,
        str(current_file)
    ]
    
    process = subprocess.run(cmd, check=True)

    summary_file = 'wandb/latest-run/files/wandb-summary.json'

    device_name = torch.cuda.get_device_name()
    if "A40" in device_name:
        reference_wandb_summary_file = a40_reference_wandb_summary_file
    elif "L40S" in device_name:
        reference_wandb_summary_file = l40s_reference_wandb_summary_file
    else:
        raise ValueError(f"Unknown device: {device_name}")

    reference_wandb_summary = json.load(open(reference_wandb_summary_file))
    wandb_summary = json.load(open(summary_file))

    fields_and_thresholds = {
        'avg_step_time': 6.0,
        'grad_norm': 0.3,
        'step_time': 6.0,
        'train_loss': 0.0025
    }

    failures = []
    for field, threshold in fields_and_thresholds.items():
        ref_value = reference_wandb_summary[field]
        current_value = wandb_summary[field]
        diff = abs(ref_value - current_value)
        print(f"INFO: {field}, diff: {diff}, threshold: {threshold}, reference: {ref_value}, current: {current_value}")
        if diff > threshold:
            failures.append(f"FAILED: {field} difference {diff} exceeds threshold of {threshold} (reference: {ref_value}, current: {current_value})")

    if failures:
        raise AssertionError("\n".join(failures))

if __name__ == "__main__":
    if os.environ.get("LOCAL_RANK") is not None:
        # We're being run by torchrun
        run_worker()
    else:
        # We're being run directly
        test_distributed_training()
