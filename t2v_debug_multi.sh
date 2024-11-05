
accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path data/mochi \
    --cache_dir "data/.cache" \
    --data_json_path "data/Encoder_Overfit_Data/videos2caption.json" \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 14 \
    --sp_size 2 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps=2 \
    --max_train_steps=2000 \
    --learning_rate=1e-5 \
    --mixed_precision="bf16" \
    --weighting_scheme "uniform" \
    --checkpointing_steps=500 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.1 \
    --ema_decay 0.999 \
    --output_dir="data/outputs/debug_uniform_14_sp2_spb2_4videos_2K"




accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path data/mochi \
    --cache_dir "data/.cache" \
    --data_json_path "data/Encoder_Overfit_Data/videos2caption.json" \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 20 \
    --sp_size 4 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps=2 \
    --max_train_steps=500 \
    --learning_rate=0.0 \
    --mixed_precision="bf16" \
    --weighting_scheme "uniform" \
    --checkpointing_steps=100 \
    --validation_steps 2 \
    --validation_sampling_steps 64 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.1 \
    --ema_decay 0.999 \
    --log_validation \
    --validation_prompt_dir "data/Encoder_Overfit_Data/validation_prompt_embed_mask" \
    --uncond_prompt_dir "data/Encoder_Overfit_Data/uncond_prompt_embed_mask" \
    --output_dir="data/outputs/debug_logvalidation_sp2"



torchrun --nnodes=1 --nproc_per_node=4 --master_port 29503 debug_mochi_sp.py
torchrun --nnodes=1 --nproc_per_node=4 --master_port 29503  debug_OSP_A2A.py


for seed in 42 
do
torchrun --nnodes=1 --nproc_per_node=4 --master_port 29503 fastvideo/model/test.py --seed $seed
done


num_gpus=1

torchrun --nproc_per_node=$num_gpus fastvideo/sample/generate_synthetic.py \
    --model_path data/mochi \
    --num_frames 163 \
    --height 480 \
    --width 848 \
    --num_inference_steps 2 \
    --guidance_scale 4.5 \
    --prompt_path "data/prompt.txt" \
    --dataset_output_dir data/synthetic_debug3

    


"A person is sawing a piece of wood placed on a support, using a large hand saw.
The saw moves rapidly back and forth across the top of the wood, generating small wood chips and dust. 
The person's hands are visible, gripping the saw firmly. The background shows a workshop environment with indistinct objects and a bench.
The focus remains on the action of sawing, with no significant change in the camera angle or background.
The sawing action continues with the saw moving through the wood, which shows a visible cut deepening as the saw progresses. 
Wood chips continue to scatter from the cutting area. 
The person's grip adjusts slightly for better control and force.
The background remains consistent with the workshop setting, and the camera stays focused on the sawing action without any shifts in perspective
No new objects or persons enter the scene. 
The sawing action reaches a deeper cut into the wood, indicating progress in the task. 
The person's movements become more pronounced as they apply more force. 
The saw blade appears more prominently as it moves back and forth. 
The background and setting remain unchanged, maintaining the focus on the sawing activity.
The camera continues to capture the action closely, emphasizing the interaction between the saw and the wood."
