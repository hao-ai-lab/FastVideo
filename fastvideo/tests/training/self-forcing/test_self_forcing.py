import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29513"
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
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.utils import FlexibleArgumentParser

wandb_name = "test_self_forcing_distill"

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "2"


def run_worker():
    """Worker function that will be run on each GPU"""
    # Create and populate args
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    
    # Set the arguments based on the distill_dmd_t2v_1.3B.sh script
    args = parser.parse_args([
        "--model_path", "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers",
        "--inference_mode", "False",
        "--pretrained_model_name_or_path", "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers",
        "--real_score_model_path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "--fake_score_model_path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "--data_path", "data/crush-smol_processed_t2v/combined_parquet_dataset",
        "--validation_dataset_file", "examples/training/finetune/wan_t2v_1.3B/crush_smol/validation.json",
        "--train_batch_size", "1",
        "--num_latent_t", "21", 
        "--num_gpus", "2",
        "--sp_size", "1",
        "--tp_size", "1",
        "--hsdp_replicate_dim", "1",
        "--hsdp_shard_dim", "2",
        "--train_sp_batch_size", "1",
        "--dataloader_num_workers", "1",
        "--gradient_accumulation_steps", "1", 
        "--max_train_steps", "2",
        "--learning_rate", "1e-5",
        "--mixed_precision", "bf16",
        "--training_state_checkpointing_steps", "30",
        "--weight_only_checkpointing_steps", "30",
        "--validation_steps", "10",
        "--validation_sampling_steps", "3",
        "--log_validation",
        "--checkpoints_total_limit", "3",
        "--ema_start_step", "0",
        "--training_cfg_rate", "0.0",
        "--output_dir", "data/wan_self_forcing_test",
        "--tracker_project_name", "wan_self_forcing_ci",
        "--wandb_run_name", wandb_name,
        "--num_height", "480",
        "--num_width", "832",
        "--num_frames", "21",
        "--flow_shift", "5",
        "--validation_guidance_scale", "1.0",
        "--weight_decay", "0.01",
        "--dit_precision", "fp32",
        "--max_grad_norm", "1.0",
        # DMD args
        "--dmd_denoising_steps", "1000,750,500",  # Reduced steps for testing
        "--min_timestep_ratio", "0.02",
        "--max_timestep_ratio", "0.98",
        "--dfake_gen_update_ratio", "5",
        "--real_score_guidance_scale", "3.0",
        "--fake_score_learning_rate", "8e-6",
        "--fake_score_betas", "0.0,0.999",
        "--warp_denoising_step",
        "--enable_gradient_checkpointing_type", "full",
        # Self-forcing specific args
        "--log_visualization",
        "--simulate_generator_forward",
        "--num_frame_per_block", "3",
        "--enable_gradient_masking",
        "--gradient_mask_last_n_frames", "21",
        "--independent_first_frame", "False",
        "--same_step_across_blocks", "True",
        "--last_step_only", "False",
        "--context_noise", "0",
        "--use_ema", "True",
        "--ema_decay", "0.99",
        "--ema_start_step", "100",
    ])
    
    # Call the main training function
    pipeline = WanSelfForcingDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Self-forcing distillation training pipeline done")

def test_distributed_training():
    """Test the distributed self-forcing training setup"""
    os.environ["WANDB_MODE"] = "offline"

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

if __name__ == "__main__":
    if os.environ.get("LOCAL_RANK") is not None:
        # We're being run by torchrun
        run_worker()
    else:
        # We're being run directly
        test_distributed_training()

