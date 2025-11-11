import os
import sys
from pathlib import Path

# Set Python path to current folder
current_dir = str(Path(__file__).parent.parent.parent.parent.parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
os.environ["PYTHONPATH"] = current_dir + ":" + os.environ.get("PYTHONPATH", "")

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29512"
import subprocess
import torch
import json
from huggingface_hub import snapshot_download
from fastvideo.utils import logger
# Import the training pipeline
from fastvideo.training.wan_training_pipeline import main
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.utils import FlexibleArgumentParser
from fastvideo.training.wan_training_pipeline import WanTrainingPipeline

wandb_name = "mfu_calculation"


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
        "--train_batch_size", "4",
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

    summary_file = 'data/mfu_calculation/tracker/wandb/latest-run/files/wandb-summary.json'

    wandb_summary = json.load(open(summary_file))
    
    # Calculate and print MFU metrics
    device_name = torch.cuda.get_device_name()
    try:
        # Get actual values from training run (logged from training_batch.raw_latent_shape)
        batch_size = wandb_summary.get("batch_size", 2)
        latent_t = wandb_summary.get("latent_shape_t", 4)
        latent_h = wandb_summary.get("latent_shape_h", 60)
        latent_w = wandb_summary.get("latent_shape_w", 104)
        avg_step_time = wandb_summary.get("avg_step_time", 1.0)
        
        seq_len = latent_t * latent_h * latent_w

        
        # Model config (Wan 1.3B)
        hidden_dim = 768
        num_layers = 24

        
        # Calculate achieved FLOPs (forward + backward)
        # Per layer, per batch, forward pass:
        # - Linear ops (QKV, out_proj, MLP): 12 * hidden_dim^2 * seq_len
        # - Attention ops (QK^T, Attn*V): 2 * seq_len^2 * hidden_dim
        # Training multiplier: 3x (1 forward + 2 backward)
        linear_flops = 12 * hidden_dim * hidden_dim * seq_len
        attention_flops = 2 * seq_len * seq_len * hidden_dim
        flops_per_layer = linear_flops + attention_flops
        achieved_flops = batch_size * flops_per_layer * num_layers * 3

        
        # Account for gradient accumulation (from config)
        grad_accum = 2  # gradient_accumulation_steps = 2
        achieved_flops *= grad_accum  # gradient_accumulation_steps = 2

        # Peak FLOPs based on device
        if "H100" in device_name:
            peak_flops_per_gpu = 1979e12
        elif "A100" in device_name:
            peak_flops_per_gpu = 312e12
        elif "A40" in device_name:
            peak_flops_per_gpu = 312e12
        elif "L40S" in device_name:
            peak_flops_per_gpu = 362e12
        else:
            raise ValueError(f"Device {device_name} not supported")
        
        # Total peak (2 GPUs)
        world_size = int(NUM_GPUS_PER_NODE)
        total_peak_flops = peak_flops_per_gpu * world_size
        
        # Calculate MFU
        achieved_flops_per_sec = achieved_flops / avg_step_time if avg_step_time > 0 else 0
        mfu = (achieved_flops_per_sec / total_peak_flops * 100) if total_peak_flops > 0 else 0

        print(f"Per-Step MFU: {mfu:.4f}%")
    except Exception as e:
        print(f"Could not calculate MFU: {e}")
    

if __name__ == "__main__":
    if os.environ.get("LOCAL_RANK") is not None:
        # We're being run by torchrun
        run_worker()
    else:
        # We're being run directly
        test_distributed_training()
