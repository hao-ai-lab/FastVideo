import os

import fastvideo.envs as envs

envs.setdefault_external("MASTER_ADDR", "localhost")
envs.setdefault_external("MASTER_PORT", "29514")
import sys
import subprocess
from pathlib import Path
import torch
import json
import yaml
from huggingface_hub import snapshot_download

wandb_name = "test_lora_training"
l40s_reference_wandb_summary_file = "fastvideo/tests/training/lora/l40s_reference_lora_wandb_summary.json"

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "2"


def test_lora_training():
    """Test the LoRA training setup"""
    data_dir = Path("data/crush-smol_processed_t2v")

    if not data_dir.exists():
        print(f"Downloading test dataset to {data_dir}...")
        snapshot_download(repo_id="wlsaidhi/crush-smol_processed_t2v",
                          local_dir=str(data_dir),
                          repo_type="dataset",
                          local_dir_use_symlinks=False)

    training_run_config = {
        "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "engine": {
            "num_gpus": int(NUM_GPUS_PER_NODE),
            "parallelism": {
                "sp_size": int(NUM_GPUS_PER_NODE),
                "tp_size": int(NUM_GPUS_PER_NODE),
                "hsdp_replicate_dim": 1,
                "hsdp_shard_dim": 2,
            },
            "precision": {
                "dit": "fp32",
            },
        },
        "training": {
            "data": {
                "data_path": "data/crush-smol_processed_t2v/combined_parquet_dataset",
                "train_batch_size": 1,
                "num_latent_t": 8,
                "train_sp_batch_size": 1,
                "dataloader_num_workers": 1,
                "training_cfg_rate": 0.1,
                "num_height": 480,
                "num_width": 832,
                "num_frames": 77,
                "seed": 42,
            },
            "optimizer": {
                "learning_rate": 5e-5,
                "weight_decay": 1e-4,
                "max_grad_norm": 1.0,
            },
            "loop": {
                "gradient_accumulation_steps": 8,
                "max_train_steps": 5,
            },
            "checkpoint": {
                "output_dir": "/workspace",
                "weight_only_checkpointing_steps": 6000,
                "training_state_checkpointing_steps": 6000,
                "checkpoints_total_limit": 3,
            },
            "tracker": {
                "project_name": "wan_lora_finetune_ci",
                "run_name": wandb_name,
            },
            "validation": {
                "enabled": True,
                "dataset_file": "examples/training/finetune/wan_t2v_1.3B/crush_smol/validation.json",
                "every_steps": 50,
                "sampling_steps": [50],
                "guidance_scale": 1.0,
            },
            "ema": {
                "start_step": 0,
            },
            "lora": {
                "enabled": True,
                "rank": 32,
            },
        },
    }
    training_config_path = Path("/workspace/test_lora_training.yaml")
    training_config_path.parent.mkdir(parents=True, exist_ok=True)
    training_config_path.write_text(yaml.safe_dump(training_run_config, sort_keys=False))

    # Run torchrun command directly on the training pipeline like the shell script
    cmd = [
        "torchrun", "--nnodes", NUM_NODES, "--nproc_per_node", NUM_GPUS_PER_NODE, "--master_port",
        os.environ["MASTER_PORT"], "fastvideo/training/wan_training_pipeline.py", "--config",
        str(training_config_path)
    ]

    with envs.override_external("WANDB_MODE", "offline"):
        process = subprocess.run(cmd, check=True)

    summary_file = '/workspace/tracker/wandb/latest-run/files/wandb-summary.json'

    device_name = torch.cuda.get_device_name()
    assert "L40S" in device_name or "B200" in device_name, (f"LoRA training regression supports L40S and the "
                                                               f"GB200 Slurm CI target, got {device_name}")
    reference_wandb_summary_file = l40s_reference_wandb_summary_file
    reference_wandb_summary = json.load(open(reference_wandb_summary_file))

    wandb_summary = json.load(open(summary_file))

    # Define thresholds for LoRA training based on the provided console outputs
    fields_and_thresholds = {
        'avg_step_time': 30.0,  # something up with modal
        # 'grad_norm': 0.05,      # too volatile for now. TODO: fix nondeterminism in training
        'step_time': 30.0,  # something up with modal
        'train_loss': 0.05
    }

    failures = []
    for field, threshold in fields_and_thresholds.items():
        if field in reference_wandb_summary and field in wandb_summary:
            ref_value = reference_wandb_summary[field]
            current_value = wandb_summary[field]
            diff = abs(ref_value - current_value)
            print(
                f"INFO: {field}, diff: {diff}, threshold: {threshold}, reference: {ref_value}, current: {current_value}"
            )
            if diff > threshold:
                failures.append(
                    f"FAILED: {field} difference {diff} exceeds threshold of {threshold} (reference: {ref_value}, current: {current_value})"
                )
        else:
            print(f"WARNING: Field {field} not found in one or both summary files")

    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    test_lora_training()
