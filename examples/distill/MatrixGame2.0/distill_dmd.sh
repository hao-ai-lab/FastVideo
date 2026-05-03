#!/bin/bash

export WANDB_API_KEY="7ff8b6e8356924f7a6dd51a0342dd1a422ea9352"
export WANDB_MODE="offline"
export TOKENIZERS_PARALLELISM=false

RUN_NAME=$(date +"%m%d_%H%M")
echo "RUN_NAME: $RUN_NAME"

# Model paths for Self-Forcing DMD distillation:
# GENERATOR_MODEL_PATH="../mg_models/Solaris-Causal-30K-8K-18K"
GENERATOR_MODEL_PATH="./mg_causal_Solaris"
REAL_SCORE_MODEL_PATH="./mg_bidirectional_Solaris"  # Teacher mode
FAKE_SCORE_MODEL_PATH="./mg_causal_Solaris"  # Critic model

DATA_DIR="/home/hal-kaiqin/solaris_dataset_unzip/solaris"
VALIDATION_DATASET_FILE="examples/distill/MatrixGame2.0/validation.json"
NUM_GPUS=1

# Training arguments
training_args=(
  --tracker_project_name "matrixgame_sf"
  --output_dir "checkpoints/matrixgame_sf_${RUN_NAME}"
  --wandb_run_name "${RUN_NAME}_test"
  --max_train_steps 5
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 352
  --num_width 640
  # --enable_gradient_checkpointing_type "full"
  --simulate_generator_forward
  --num_frames 81
  --num_frame_per_block 3  # Frame generation block size for self-forcing
  # --enable_gradient_masking
  # --gradient_mask_last_n_frames 21
  --init_weights_from_safetensors "checkpoints/matrixgame_sf_0413_1109/checkpoint-1200_weight_only/ema/generator_ema.safetensors"
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_GPUS
)

model_args=(
  --model_path $GENERATOR_MODEL_PATH  # TODO: check if you can remove this in this script
  --pretrained_model_name_or_path $GENERATOR_MODEL_PATH
  --real_score_model_path $REAL_SCORE_MODEL_PATH
  --fake_score_model_path $FAKE_SCORE_MODEL_PATH
)

dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

# Validation arguments
validation_args=(
  --log_validation
  --log_visualization
  --visualization-steps 100
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 100
  --validation_sampling_steps "4"
  --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 3e-6
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 400
  --training_state_checkpointing_steps 400
  --weight_decay 0
  --betas "0.9,0.95"
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --flow_shift 5
  --seed 1000
  --use_ema True
  --ema_decay 0.99
  --ema_start_step 200
)

dmd_args=(
  --dmd_denoising_steps '1000,750,500,250'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --dfake_gen_update_ratio 5
  --real_score_guidance_scale 3.0
  --fake_score_learning_rate 3e-7
  --fake_score_betas "0.9,0.95"
  --warp_denoising_step
)

self_forcing_args=(
  --independent_first_frame False  # Whether to treat first frame independently
  --same_step_across_blocks True  # Whether to use same denoising step across all blocks
  --last_step_only False  # Whether to only use the last denoising step
  --context_noise 0  # Amount of noise to add during context caching (0 = no noise)
)

torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/matrixgame_self_forcing_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}" \
    "${self_forcing_args[@]}"