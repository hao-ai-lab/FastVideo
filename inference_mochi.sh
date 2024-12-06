


num_gpus=4

torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/mochi \
    --prompt_embed_path "data/synthetic_debug2/prompt_embed/2.pt" \
    --encoder_attention_mask_path "data/synthetic_debug2/prompt_attention_mask/1.pt" \
    --num_frames  163 \
    --height 480 \
    --width 848 \
    --num_inference_steps 32 \
    --guidance_scale 4.5 \
    --output_path outputs_video/debug \
    --shift 8 \
    --seed 12345 \
    --scheduler_type "pcm_linear_quadratic" 

