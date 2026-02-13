import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29514"
import sys
import subprocess
from pathlib import Path
import torch
import json
from huggingface_hub import snapshot_download

wandb_name = "test_lora_training"
l40s_reference_wandb_summary_file = "fastvideo/tests/training/lora/l40s_reference_lora_wandb_summary.json"

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "2"


def test_lora_training():
    """Test the LoRA training setup"""
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

    # Run torchrun command directly on the training pipeline like the shell script
    cmd = [
        "torchrun",
        "--nnodes", NUM_NODES,
        "--nproc_per_node", NUM_GPUS_PER_NODE,
        "--master_port", os.environ["MASTER_PORT"],
        "fastvideo/training/wan_training_pipeline.py",
        "--model_path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "--inference_mode", "False",
        "--pretrained_model_name_or_path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "--data_path", "data/crush-smol_processed_t2v/combined_parquet_dataset",
        "--validation_dataset_file", "examples/training/finetune/wan_t2v_1.3B/crush_smol/validation.json",
        "--train_batch_size", "1",
        "--num_latent_t", "8",
        "--num_gpus", NUM_GPUS_PER_NODE,
        "--sp_size", NUM_GPUS_PER_NODE,
        "--tp_size", NUM_GPUS_PER_NODE,
        "--hsdp_replicate_dim", "1",
        "--hsdp_shard_dim", "2",
        "--train_sp_batch_size", "1",
        "--dataloader_num_workers", "1",
        "--gradient_accumulation_steps", "8",
        "--max_train_steps", "5",
        "--learning_rate", "5e-5",
        "--mixed_precision", "bf16",
        "--weight_only_checkpointing_steps", "6000",
        "--training_state_checkpointing_steps", "6000",
        "--validation_steps", "50",
        "--validation_sampling_steps", "50",
        "--log_validation",
        "--checkpoints_total_limit", "3",
        "--ema_start_step", "0",
        "--training_cfg_rate", "0.1",
        "--output_dir", "/workspace",
        "--tracker_project_name", "wan_lora_finetune_ci",
        "--wandb_run_name", wandb_name,
        "--num_height", "480",
        "--num_width", "832",
        "--num_frames", "77",
        "--validation_guidance_scale", "1.0",
        "--num_euler_timesteps", "50",
        "--multi_phased_distill_schedule", "4000-1",
        "--weight_decay", "1e-4",
        "--not_apply_cfg_solver",
        "--dit_precision", "fp32",
        "--max_grad_norm", "1.0",
        "--lora_rank", "32",
        "--lora_training", "True",
        "--seed", "42"
    ]
    
    process = subprocess.run(cmd, check=True)

    summary_file = '/workspace/tracker/wandb/latest-run/files/wandb-summary.json'

    device_name = torch.cuda.get_device_name()
    assert "L40S" in device_name, "Test must be run on L40S"
    reference_wandb_summary_file = l40s_reference_wandb_summary_file
    reference_wandb_summary = json.load(open(reference_wandb_summary_file))

    wandb_summary = json.load(open(summary_file))

    # Define thresholds for LoRA training based on the provided console outputs
    fields_and_thresholds = {
        'avg_step_time': 20.0,  # something up with modal
        # 'grad_norm': 0.05,      # too volatile for now. TODO: fix nondeterminism in training
        'step_time': 20.0,      # something up with modal
        'train_loss': 0.05    
    }

    failures = []
    for field, threshold in fields_and_thresholds.items():
        if field in reference_wandb_summary and field in wandb_summary:
            ref_value = reference_wandb_summary[field]
            current_value = wandb_summary[field]
            diff = abs(ref_value - current_value)
            print(f"INFO: {field}, diff: {diff}, threshold: {threshold}, reference: {ref_value}, current: {current_value}")
            if diff > threshold:
                failures.append(f"FAILED: {field} difference {diff} exceeds threshold of {threshold} (reference: {ref_value}, current: {current_value})")
        else:
            print(f"WARNING: Field {field} not found in one or both summary files")

    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    test_lora_training() 