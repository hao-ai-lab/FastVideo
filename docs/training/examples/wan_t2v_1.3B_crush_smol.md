**Source:** [examples/training/finetune/wan_t2v_1.3B/crush_smol](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/training/finetune/wan_t2v_1.3B/crush_smol)

# Wan2.1-T2V-1.3B Crush-Smol Example
These are e2e example scripts for finetuning Wan2.1 T2V 1.3B on the crush-smol dataset.

## Execute the following commands from `FastVideo/` to run training:

### Download crush-smol dataset:

`bash examples/training/finetune/wan_t2v_1_3b/crush_smol/download_dataset.sh`

### Preprocess the videos and captions into latents:

`bash examples/training/finetune/wan_t2v_1_3b/crush_smol/preprocess_wan_data_t2v.sh`

### Edit the following file and run finetuning:

`bash examples/training/finetune/wan_t2v_1_3b/crush_smol/finetune_t2v.sh`


## Additional Files

??? note "download_dataset.sh"

    ```sh
    #!/bin/bash
    
    python scripts/huggingface/download_hf.py --repo_id "wlsaidhi/crush-smol-merged" --local_dir "data/crush-smol" --repo_type "dataset"
    ```

??? note "finetune_t2v.sh"

    ```sh
    #!/bin/bash
    
    export WANDB_BASE_URL="https://api.wandb.ai"
    export WANDB_MODE=online
    export TOKENIZERS_PARALLELISM=false
    # export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
    
    MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    DATA_DIR="data/crush-smol_processed_t2v/combined_parquet_dataset/"
    VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"
    NUM_GPUS=4
    # export CUDA_VISIBLE_DEVICES=4,5
    
    
    # Training arguments
    training_args=(
      --tracker_project_name "wan_t2v_finetune"
      --output_dir "checkpoints/wan_t2v_finetune"
      --max_train_steps 5000
      --train_batch_size 1
      --train_sp_batch_size 1
      --gradient_accumulation_steps 8
      --num_latent_t 20
      --num_height 480
      --num_width 832
      --num_frames 77
      --enable_gradient_checkpointing_type "full"
    )
    
    # Parallel arguments
    parallel_args=(
      --num_gpus $NUM_GPUS 
      --sp_size $NUM_GPUS 
      --tp_size 1
      --hsdp_replicate_dim 1
      --hsdp_shard_dim $NUM_GPUS
    )
    
    # Model arguments
    model_args=(
      --model_path $MODEL_PATH
      --pretrained_model_name_or_path $MODEL_PATH
    )
    
    # Dataset arguments
    dataset_args=(
      --data_path $DATA_DIR
      --dataloader_num_workers 1
    )
    
    # Validation arguments
    validation_args=(
      --log_validation 
      --validation_dataset_file $VALIDATION_DATASET_FILE
      --validation_steps 200
      --validation_sampling_steps "50" 
      --validation_guidance_scale "6.0"
    )
    
    # Optimizer arguments
    optimizer_args=(
      --learning_rate 5e-5
      --mixed_precision "bf16"
      --weight_only_checkpointing_steps 1000
      --training_state_checkpointing_steps 1000
      --weight_decay 1e-4
      --max_grad_norm 1.0
    )
    
    # Miscellaneous arguments
    miscellaneous_args=(
      --inference_mode False
      --checkpoints_total_limit 3
      --training_cfg_rate 0.1
      --multi_phased_distill_schedule "4000-1"
      --not_apply_cfg_solver
      --dit_precision "fp32"
      --num_euler_timesteps 50
      --ema_start_step 0
      --enable_gradient_checkpointing_type "full"
      # --resume_from_checkpoint "checkpoints/wan_t2v_finetune/checkpoint-2500"
    )
    
    torchrun \
      --nnodes 1 \
      --nproc_per_node $NUM_GPUS \
        fastvideo/training/wan_training_pipeline.py \
        "${parallel_args[@]}" \
        "${model_args[@]}" \
        "${dataset_args[@]}" \
        "${training_args[@]}" \
        "${optimizer_args[@]}" \
        "${validation_args[@]}" \
        "${miscellaneous_args[@]}"
    
    ```

??? note "finetune_t2v.slurm"

    ```slurm
    #!/bin/bash
    #SBATCH --job-name=t2v
    #SBATCH --partition=main
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --gres=gpu:8
    #SBATCH --cpus-per-task=128
    #SBATCH --mem=1440G
    #SBATCH --output=t2v_output/t2v_%j.out
    #SBATCH --error=t2v_output/t2v_%j.err
    #SBATCH --exclusive
    set -e -x
    
    # Environment Setup
    source ~/conda/miniconda/bin/activate
    conda activate will-fv
    
    # Basic Info
    export WANDB_MODE="online"
    export NCCL_P2P_DISABLE=1
    export TORCH_NCCL_ENABLE_MONITORING=0
    # different cache dir for different processes
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
    export MASTER_PORT=29500
    export NODE_RANK=$SLURM_PROCID
    nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
    export MASTER_ADDR=${nodes[0]}
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    export TOKENIZERS_PARALLELISM=false
    export WANDB_BASE_URL="https://api.wandb.ai"
    export WANDB_MODE=online
    # export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
    
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "NODE_RANK: $NODE_RANK"
    
    # Configs
    NUM_GPUS=8
    MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    DATA_DIR="data/crush-smol_processed_t2v/combined_parquet_dataset/"
    VALIDATION_DATASET_FILE="examples/training/finetune/wan_t2v_1_3b/crush_smol/validation.json"
    # export CUDA_VISIBLE_DEVICES=4,5
    # IP=[MASTER NODE IP]
    
    # Training arguments
    training_args=(
      --tracker_project_name wan_t2v_finetune
      --output_dir "checkpoints/wan_t2v_finetune"
      --max_train_steps 1000
      --train_batch_size 4
      --train_sp_batch_size 1
      --gradient_accumulation_steps 1
      --num_latent_t 8
      --num_height 480
      --num_width 832
      --num_frames 77
      --enable_gradient_checkpointing_type "full"
    )
    
    # Parallel arguments
    parallel_args=(
      --num_gpus $NUM_GPUS
      --sp_size 4
      --tp_size 1
      --hsdp_replicate_dim 2
      --hsdp_shard_dim 4
    )
    
    # Model arguments
    model_args=(
      --model_path $MODEL_PATH
      --pretrained_model_name_or_path $MODEL_PATH
    )
    
    # Dataset arguments
    dataset_args=(
      --data_path "$DATA_DIR"
      --dataloader_num_workers 10
    )
    
    # Validation arguments
    validation_args=(
      --log_validation
      --validation_dataset_file "$VALIDATION_DATASET_FILE"
      --validation_steps 100
      --validation_sampling_steps "50"
      --validation_guidance_scale "6.0"
    )
    
    # Optimizer arguments
    optimizer_args=(
      --learning_rate 5e-5
      --mixed_precision "bf16"
      --weight_only_checkpointing_steps 400
      --training_state_checkpointing_steps 400
      --weight_decay 1e-4
      --max_grad_norm 1.0
    )
    
    # Miscellaneous arguments
    miscellaneous_args=(
      --inference_mode False
      --checkpoints_total_limit 3
      --training_cfg_rate 0.1
      --multi_phased_distill_schedule "4000-1"
      --not_apply_cfg_solver
      --dit_precision "fp32"
      --num_euler_timesteps 50
      --ema_start_step 0
      --enable_gradient_checkpointing_type "full"
    )
    
    srun torchrun \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc_per_node $NUM_GPUS \
    --node_rank $SLURM_PROCID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        fastvideo/training/wan_training_pipeline.py \
        "${parallel_args[@]}" \
        "${model_args[@]}" \
        "${dataset_args[@]}" \
        "${training_args[@]}" \
        "${optimizer_args[@]}" \
        "${validation_args[@]}" \
        "${miscellaneous_args[@]}"
    ```

??? note "finetune_t2v_lora.sh"

    ```sh
    #!/bin/bash
    
    export WANDB_BASE_URL="https://api.wandb.ai"
    export WANDB_MODE=online
    # export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
    
    MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    DATA_DIR="data/crush-smol_processed_t2v/combined_parquet_dataset/"
    VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"
    NUM_GPUS=1
    # export CUDA_VISIBLE_DEVICES=4,5
    
    
    # Training arguments
    training_args=(
      --tracker_project_name "wan_t2v_finetune"
      --output_dir "checkpoints/wan_t2v_finetune_lora"
      --max_train_steps 5000
      --train_batch_size 1
      --train_sp_batch_size 1
      --gradient_accumulation_steps 8
      --num_latent_t 20
      --num_height 480
      --num_width 832
      --num_frames 77
      --lora_rank 32
      --lora_training True
    )
    
    # Parallel arguments
    parallel_args=(
      --num_gpus $NUM_GPUS 
      --sp_size $NUM_GPUS 
      --tp_size $NUM_GPUS
      --hsdp_replicate_dim 1
      --hsdp_shard_dim $NUM_GPUS
    )
    
    # Model arguments
    model_args=(
      --model_path $MODEL_PATH
      --pretrained_model_name_or_path $MODEL_PATH
    )
    
    # Dataset arguments
    dataset_args=(
      --data_path $DATA_DIR
      --dataloader_num_workers 1
    )
    
    # Validation arguments
    validation_args=(
      --log_validation 
      --validation_dataset_file $VALIDATION_DATASET_FILE
      --validation_steps 200
      --validation_sampling_steps "50" 
      --validation_guidance_scale "6.0"
    )
    
    # Optimizer arguments
    optimizer_args=(
      --learning_rate 5e-5
      --mixed_precision "bf16"
      --weight_only_checkpointing_steps 400
      --training_state_checkpointing_steps 400
      --weight_decay 1e-4
      --max_grad_norm 1.0
    )
    
    # Miscellaneous arguments
    miscellaneous_args=(
      --inference_mode False
      --checkpoints_total_limit 3
      --training_cfg_rate 0.1
      --multi_phased_distill_schedule "4000-1"
      --not_apply_cfg_solver
      --dit_precision "fp32"
      --num_euler_timesteps 50
      --ema_start_step 0
      --resume_from_checkpoint "checkpoints/wan_t2v_finetune_lora/checkpoint-160"
    )
    
    torchrun \
      --nnodes 1 \
      --nproc_per_node $NUM_GPUS \
      --master_port 29501 \
        fastvideo/training/wan_training_pipeline.py \
        "${parallel_args[@]}" \
        "${model_args[@]}" \
        "${dataset_args[@]}" \
        "${training_args[@]}" \
        "${optimizer_args[@]}" \
        "${validation_args[@]}" \
        "${miscellaneous_args[@]}"
    
    ```

??? note "preprocess_wan_data_t2v.sh"

    ```sh
    #!/bin/bash
    
    GPU_NUM=1 # 2,4,8
    MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    MODEL_TYPE="wan"
    DATA_MERGE_PATH="data/crush-smol/merge.txt"
    OUTPUT_DIR="data/crush-smol_processed_t2v/"
    
    torchrun --nproc_per_node=$GPU_NUM \
        fastvideo/pipelines/preprocess/v1_preprocess.py \
        --model_path $MODEL_PATH \
        --data_merge_path $DATA_MERGE_PATH \
        --preprocess_video_batch_size 8 \
        --seed 42 \
        --max_height 480 \
        --max_width 832 \
        --num_frames 77 \
        --dataloader_num_workers 0 \
        --output_dir=$OUTPUT_DIR \
        --train_fps 16 \
        --samples_per_file 8 \
        --flush_frequency 8 \
        --video_length_tolerance_range 5 \
        --preprocess_task "t2v" 
    ```

??? note "preprocess_wan_data_t2v_new.sh"

    ```sh
    #!/bin/bash
    
    GPU_NUM=2 # 2,4,8
    MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    DATASET_PATH="data/crush-smol/"
    OUTPUT_DIR="data/crush-smol_processed_t2v/"
    
    torchrun --nproc_per_node=$GPU_NUM \
        -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
        --model_path $MODEL_PATH \
        --mode preprocess \
        --workload_type t2v \
        --preprocess.video_loader_type torchvision \
        --preprocess.dataset_type merged \
        --preprocess.dataset_path $DATASET_PATH \
        --preprocess.dataset_output_dir $OUTPUT_DIR \
        --preprocess.preprocess_video_batch_size 2 \
        --preprocess.dataloader_num_workers 0 \
        --preprocess.max_height 480 \
        --preprocess.max_width 832 \
        --preprocess.num_frames 77 \
        --preprocess.train_fps 16 \
        --preprocess.samples_per_file 8 \
        --preprocess.flush_frequency 8 \
        --preprocess.video_length_tolerance_range 5
    
    ```

??? note "validation.json"

    ```json
    {
      "data": [
        {
          "caption": "A large metal cylinder is seen pressing down on a pile of Oreo cookies, flattening them as if they were under a hydraulic press.",
          "image_path": null,
          "video_path": null,
          "num_inference_steps": 50,
          "height": 480,
          "width": 832,
          "num_frames": 77
        },
        {
          "caption": "A large metal cylinder is seen compressing colorful clay into a compact shape, demonstrating the power of a hydraulic press.",
          "image_path": null,
          "video_path": null,
          "num_inference_steps": 50,
          "height": 480,
          "width": 832,
          "num_frames": 77
        },
        {
          "caption": "A large metal cylinder is seen pressing down on a pile of colorful candies, flattening them as if they were under a hydraulic press. The candies are crushed and broken into small pieces, creating a mess on the table.",
          "image_path": null,
          "video_path": null,
          "num_inference_steps": 50,
          "height": 480,
          "width": 832,
          "num_frames": 77
        }
      ]
    }
    
    ```

