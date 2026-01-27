export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
#export MUDNN_LOG_LEVEL=INFO
export WANDB_MODE=disabled
# export TORCH_PROFILING_TRACE=/data/yanguo.sun/hunyuan-video/FastVideo/profiling
torchrun --nnodes 1 --nproc_per_node 8 \
    fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path /data/yanguo.sun/hunyuan-video/HunyuanVideo/ckpts \
    --dit_model_name_or_path /data/yanguo.sun/hunyuan-video/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --model_type "hunyuan" \
    --cache_dir /data/yanguo.sun/hunyuan-video/.cache \
    --data_json_path /data/yanguo.sun/hunyuan-video/datasets/videos2caption.json \
    --validation_prompt_dir /data/yanguo.sun/hunyuan-video/datasets/validation \
    --gradient_checkpointing \
    --train_batch_size=2 \
    --num_latent_t 32 \
    --sp_size 8 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=20 \
    --learning_rate=1e-5 \
    --mixed_precision=bf16 \
    --checkpointing_steps=200 \
    --validation_steps 100 \
    --validation_sampling_steps 50 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir=./outputs/OpenVidHD-Finetune-Hunyuan \
    --tracker_project_name OpenVidHD-Finetune-Hunyuan \
    --num_frames 125 \
    --num_height 720 \
    --num_width 1280 \
    --shift 7 \
    --validation_guidance_scale "1.0" \
    --use_fused_rmsnorm \
    --use_fused_rope \
