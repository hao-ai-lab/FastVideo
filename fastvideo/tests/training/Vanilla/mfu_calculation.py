import os
import sys
from pathlib import Path

# Set Python path to current folder
current_dir = str(Path(__file__).parent.parent.parent.parent.parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import subprocess
import torch
import json
from huggingface_hub import snapshot_download
import fastvideo.envs as envs
from fastvideo.utils import logger
# Import the training pipeline
from fastvideo.training.wan_training_pipeline import main
from fastvideo.api.training_schema import resolve_training_config
from fastvideo.training.wan_training_pipeline import WanTrainingPipeline

MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_PATH = "data/crush-smol_processed_t2v/training_dataset/worker_1/worker_0/"
VALIDATION_DATASET_FILE = "examples/training/finetune/wan_t2v_1.3B/crush_smol/validation.json"
OUTPUT_DIR = Path("checkpoints/wan_t2v_finetune")
PROFILER_TRACE_ROOT = Path("/mnt/fast-disks/hao_lab/ohm/profiler_traces/wan_t2v_finetune")
WANDB_SUMMARY_FILE = OUTPUT_DIR / "tracker/wandb/latest-run/files/wandb-summary.json"

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "2"
GRAD_ACCUM = "1"
MASTER_PORT = os.environ.get("MASTER_PORT", "29504")

envs.setdefault_external("MASTER_ADDR", "localhost")
envs.setdefault_external("MASTER_PORT", MASTER_PORT)


def run_worker():
    """Worker function that will be run on each GPU"""
    # Set the arguments as they are in finetune_t2v.sh
    training_run_config = {
        "model_path": MODEL_PATH,
        "engine": {
            "num_gpus": int(NUM_GPUS_PER_NODE),
            "parallelism": {
                "sp_size": int(NUM_GPUS_PER_NODE),
                "tp_size": 1,
                "hsdp_replicate_dim": int(NUM_GPUS_PER_NODE),
                "hsdp_shard_dim": 1,
            },
            #"compile": {"enabled": True},
            "precision": {
                "dit": "fp32",
            },
        },
        "training": {
            "data": {
                "data_path": DATA_PATH,
                "dataloader_num_workers": 1,
                "train_batch_size": 4,
                "train_sp_batch_size": 1,
                "num_latent_t": 20,
                "num_height": 720,
                "num_width": 1280,
                "num_frames": 77,
                "training_cfg_rate": 0.1,
            },
            "optimizer": {
                "learning_rate": 5e-5,
                "weight_decay": 1e-4,
                "max_grad_norm": 1.0,
            },
            "loop": {
                "gradient_accumulation_steps": int(GRAD_ACCUM),
                "max_train_steps": 20,
            },
            "checkpoint": {
                "weight_only_checkpointing_steps": 250,
                "training_state_checkpointing_steps": 250,
                "output_dir": str(OUTPUT_DIR),
                "checkpoints_total_limit": 3,
            },
            "tracker": {
                "project_name": "wan_t2v_finetune",
            },
            "validation": {
                "dataset_file": VALIDATION_DATASET_FILE,
                "every_steps": 200,
                "sampling_steps": [50],
                "guidance_scale": 6.0,
                #"enabled": True,
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
    pipeline = WanTrainingPipeline.from_pretrained(resolved_config.model_path, resolved_config=resolved_config)
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
        "torchrun", "--nnodes", NUM_NODES, "--nproc_per_node", NUM_GPUS_PER_NODE, "--master_port", MASTER_PORT,
        str(current_file)
    ]
    # The torchrun workers run this file and import fastvideo from the repository root.
    worker_pythonpath = current_dir + ":" + os.environ.get("PYTHONPATH", "")
    with envs.override_external("WANDB_MODE", "offline"), envs.override_external("PYTHONPATH", worker_pythonpath):
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

        # FLOPs per layer (forward pass)
        # - QKV + out proj: 8 * hidden_dim^2 * seq_len
        # - Cross-attn proj: 4 * hidden_dim^2 * seq_len + 4 * hidden_dim^2 * context_len
        # - MLP: 4 * hidden_dim * ffn_dim * seq_len
        # - Self-attn matmuls: 4 * seq_len^2 * hidden_dim
        # - Cross-attn matmuls: 4 * seq_len * context_len * hidden_dim
        qkv_out_flops = 8 * hidden_dim * hidden_dim * seq_len
        cross_attn_proj_flops = ((4 * hidden_dim * hidden_dim * seq_len) + (4 * hidden_dim * hidden_dim * context_len))
        mlp_flops = 4 * hidden_dim * ffn_dim * seq_len
        self_attn_flops = 4 * seq_len * seq_len * hidden_dim
        cross_attn_flops = 4 * seq_len * context_len * hidden_dim
        flops_per_layer = (qkv_out_flops + cross_attn_proj_flops + mlp_flops + self_attn_flops + cross_attn_flops)

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
        test_distributed_training()
