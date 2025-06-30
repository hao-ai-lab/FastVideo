import os
import sys
import subprocess
from pathlib import Path
import json
from huggingface_hub import snapshot_download

# Import the training pipeline
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from fastvideo.v1.training.wan_training_pipeline import main
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.utils import FlexibleArgumentParser

wandb_name = "test_training_loss_VSA"
reference_wandb_summary_file = "fastvideo/v1/tests/training/VSA/reference_wandb_summary_VSA.json"

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "2"

os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"

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
        "--data_path", "data/mini_dataset_i2v_VSA/combined_parquet_dataset",
        "--validation_preprocessed_path", "data/mini_dataset_i2v_VSA/validation_parquet_dataset",
        "--train_batch_size", "1",
        "--num_latent_t", "4",
        "--num_gpus", "2",
        "--sp_size", "2",
        "--tp_size", "2",
        "--hsdp_replicate_dim", "1",
        "--hsdp_shard_dim", "2",
        "--train_sp_batch_size", "1",
        "--dataloader_num_workers", "4",
        "--gradient_accumulation_steps", "2",
        "--max_train_steps", "5",
        "--learning_rate", "1e-5",
        "--mixed_precision", "bf16",
        "--checkpointing_steps", "30",
        "--validation_steps", "10",
        "--validation_sampling_steps", "50",
        "--log_validation",
        "--checkpoints_total_limit", "3",
        "--allow_tf32",
        "--ema_start_step", "0",
        "--training_cfg_rate", "0.0",
        "--output_dir", "data/wan_finetune_test_VSA",
        "--tracker_project_name", "wan_finetune_ci_VSA",
        "--wandb_run_name", wandb_name,
        "--num_height", "384",
        "--num_width", "512",
        "--num_frames", "13",
        "--flow_shift", "3",
        "--validation_guidance_scale", "1.0",
        "--num_euler_timesteps", "50",
        "--weight_decay", "0.01",
        "--dit_precision", "fp32",
        "--max_grad_norm", "1.0",
        "--VSA_decay_rate", "0.01",
        "--VSA_decay_interval_steps", "1",
        "--VSA_sparsity", "0.9"
    ])
    
    # Call the main training function
    main(args)

def test_distributed_training():
    """Test the distributed training setup"""
    os.environ["WANDB_MODE"] = "online"

    data_dir = Path("data/mini_dataset_i2v_VSA")
    
    if not data_dir.exists():
        print(f"Downloading test dataset to {data_dir}...")
        snapshot_download(
            repo_id="BrianChen1129/mini_dataset_i2v_VSA",
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

    reference_wandb_summary = json.load(open(reference_wandb_summary_file))
    wandb_summary = json.load(open(summary_file))

    fields_and_thresholds = {
        'avg_step_time': 1.0,
        'grad_norm': 0.1,
        'step_time': 1.0,
        'train_loss': 0.001
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
