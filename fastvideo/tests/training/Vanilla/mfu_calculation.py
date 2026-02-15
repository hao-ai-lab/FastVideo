import os
import sys
from pathlib import Path

# Set Python path to current folder
current_dir = str(Path(__file__).parent.parent.parent.parent.parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
os.environ["PYTHONPATH"] = current_dir + ":" + os.environ.get("PYTHONPATH", "")

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

MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_PATH = "data/crush-smol_processed_t2v/combined_parquet_dataset/worker_0/"
VALIDATION_DATASET_FILE = "examples/training/finetune/wan_t2v_1.3B/crush_smol/validation.json"
OUTPUT_DIR = Path("checkpoints/wan_t2v_finetune")
PROFILER_TRACE_ROOT = Path("/mnt/fast-disks/hao_lab/ohm/profiler_traces/wan_t2v_finetune")
WANDB_SUMMARY_FILE = OUTPUT_DIR / "tracker/wandb/latest-run/files/wandb-summary.json"

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "2"
GRAD_ACCUM = "1"
MASTER_PORT = "29510"

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = MASTER_PORT
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"


def run_worker():
    """Worker function that will be run on each GPU"""
    # Create and populate args
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    
    # Set the arguments as they are in finetune_t2v.sh
    args = parser.parse_args([
        "--model_path", MODEL_PATH,
        "--inference_mode", "False",
        "--pretrained_model_name_or_path", MODEL_PATH,
        "--data_path", DATA_PATH,
        "--dataloader_num_workers", "1",
        "--train_batch_size", "8",
        "--train_sp_batch_size", "1",
        "--gradient_accumulation_steps", GRAD_ACCUM,
        "--num_latent_t", "20",
        "--num_height", "720",
        "--num_width", "1280",
        "--num_frames", "77",
        "--enable_gradient_checkpointing_type", "full",
        "--max_train_steps", "5",
        "--learning_rate", "5e-5",
        "--mixed_precision", "bf16",
        "--weight_only_checkpointing_steps", "250",
        "--training_state_checkpointing_steps", "250",
        "--weight_decay", "1e-4",
        "--max_grad_norm", "1.0",
        "--num_euler_timesteps", "50",
        "--multi_phased_distill_schedule", "4000-1",
        "--not_apply_cfg_solver",
        "--training_cfg_rate", "0.1",
        "--ema_start_step", "0",
        "--dit_precision", "fp32",
        "--output_dir", str(OUTPUT_DIR),
        "--tracker_project_name", "wan_t2v_finetune",
        "--checkpoints_total_limit", "3",
        "--validation_dataset_file", VALIDATION_DATASET_FILE,
        "--validation_steps", "200",
        "--validation_sampling_steps", "50",
        "--validation_guidance_scale", "6.0",
        #"--enable_torch_compile",
        #"--log_validation",
        "--num_gpus", NUM_GPUS_PER_NODE,
        "--sp_size", NUM_GPUS_PER_NODE,
        "--tp_size", "1",
        "--hsdp_replicate_dim", NUM_GPUS_PER_NODE,
        "--hsdp_shard_dim", "1"
    ])
    # Call the main training function
    pipeline = WanTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")

def test_distributed_training(profile=False):
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
    cmd = []
    if profile:
        import datetime
        PROFILER_TRACE_ROOT.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_filename = f"{PROFILER_TRACE_ROOT}/trace_{timestamp}"
        cmd.extend([
            "nsys", "profile",
            "--trace=cuda,nvtx",
            f"--output={trace_filename}",
            "--force-overwrite=true",
        ])

    # Set up rendezvous arguments for multinode/multigpu
    rdvz_backend = os.environ.get("TORCH_RDZV_BACKEND", "c10d")
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", MASTER_PORT)
    rdvz_id = os.environ.get("TORCH_RDZV_ID", "default")

    cmd.extend([
        "torchrun",
        "--nnodes", NUM_NODES,
        "--nproc_per_node", NUM_GPUS_PER_NODE,
        "--rdzv_backend", rdvz_backend,
        "--rdzv_endpoint", f"{master_addr}:{master_port}",
        "--rdzv_id", rdvz_id,
        "--master_addr", master_addr,
        "--master_port", master_port,
        str(current_file)
    ])
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

    summary_file = WANDB_SUMMARY_FILE

    with summary_file.open() as f:
        wandb_summary = json.load(f)
    
    # Calculate and print MFU metrics
    device_name = torch.cuda.get_device_name()
    try:
        # Get actual values from training run (logged from training_batch.raw_latent_shape)
        batch_size = wandb_summary.get("batch_size")
        seq_len = wandb_summary.get("dit_seq_len")
        context_len = wandb_summary.get("context_len")
        avg_step_time = wandb_summary.get("avg_step_time")
        hidden_dim = wandb_summary.get("hidden_dim")
        num_layers = wandb_summary.get("num_layers")
        ffn_dim = wandb_summary.get("ffn_dim")

        logger.info(f"batch_size: {batch_size}, seq_len: {seq_len}, context_len: {context_len}, avg_step_time: {avg_step_time}, hidden_dim: {hidden_dim}, num_layers: {num_layers}, ffn_dim: {ffn_dim}")


        

        # FLOPs per layer (forward pass)
        # - QKV + out proj: 8 * hidden_dim^2 * seq_len
        # - Cross-attn proj: 4 * hidden_dim^2 * seq_len + 4 * hidden_dim^2 * context_len
        # - MLP: 4 * hidden_dim * ffn_dim * seq_len
        # - Self-attn matmuls: 4 * seq_len^2 * hidden_dim
        # - Cross-attn matmuls: 4 * seq_len * context_len * hidden_dim
        qkv_out_flops = 8 * hidden_dim * hidden_dim * seq_len
        cross_attn_proj_flops = (
            (4 * hidden_dim * hidden_dim * seq_len) +
            (4 * hidden_dim * hidden_dim * context_len) 
        )
        mlp_flops = 4 * hidden_dim * ffn_dim * seq_len
        self_attn_flops = 4 * seq_len * seq_len * hidden_dim
        cross_attn_flops = 4 * seq_len * context_len * hidden_dim
        flops_per_layer = (
            qkv_out_flops + cross_attn_proj_flops + mlp_flops + self_attn_flops + cross_attn_flops
        )

        # With full activation checkpointing: 1 forward + 3 backward (1 recompute + 2 gradient)
        achieved_flops = batch_size * flops_per_layer * num_layers * 4

        
        # Account for gradient accumulation (from config)
        grad_accum = int(GRAD_ACCUM)
        achieved_flops *= grad_accum  

        # Peak FLOPs based on device
        if "H100" in device_name:
            peak_flops_per_gpu = 989e12
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
        import argparse
        parser = argparse.ArgumentParser(description="Run MFU calculation test")
        parser.add_argument("--profile", action="store_true", help="Enable Nsight profiling with CUDA and NVTX tracing")
        args = parser.parse_args()
        
        test_distributed_training(profile=args.profile)
