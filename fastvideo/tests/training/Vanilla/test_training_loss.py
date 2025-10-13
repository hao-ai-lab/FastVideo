import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29512"
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
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.utils import FlexibleArgumentParser
from fastvideo.training.wan_training_pipeline import WanTrainingPipeline

wandb_name = "test_training_loss"
a40_reference_wandb_summary_file = "fastvideo/tests/training/Vanilla/a40_reference_wandb_summary.json"
l40s_reference_wandb_summary_file = "fastvideo/tests/training/Vanilla/l40s_reference_wandb_summary.json"

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "2"


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
        "--validation_dataset_file", "examples/training/finetune/wan_t2v_1.3B/crush_smol/validation.json",
        "--train_batch_size", "2",
        "--num_latent_t", "4",
        "--num_gpus", "2",
        "--sp_size", "2",
        "--tp_size", "2",
        "--hsdp_replicate_dim", "1",
        "--hsdp_shard_dim", "2",
        "--train_sp_batch_size", "1",
        "--dataloader_num_workers", "1",
        "--gradient_accumulation_steps", "2",
        "--max_train_steps", "5",
        "--learning_rate", "1e-6",
        "--mixed_precision", "bf16",
        "--weight_only_checkpointing_steps", "30",
        "--training_state_checkpointing_steps", "30",
        "--validation_steps", "10",
        "--validation_sampling_steps", "8",
        "--log_validation",
        "--checkpoints_total_limit", "3",
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
    pipeline = WanTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")

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
        "--master_port", os.environ["MASTER_PORT"],
        str(current_file)
    ]
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

    summary_file = 'data/wan_finetune_test/tracker/wandb/latest-run/files/wandb-summary.json'

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
