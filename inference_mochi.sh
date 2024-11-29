

num_gpus=4

torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/mochi \
    --prompt_path data/prompt.txt \
    --transformer_path data/outputs/video_distill_synthetic/checkpoint-1500 \
    --num_frames  163 \
    --height 480 \
    --width 848 \
    --num_inference_steps 8 \
    --guidance_scale 4.5 \
    --output_path outputs_video/distill_lq_163_1500_precision_stochastic_0.7 \
    --shift 8 \
    --seed 12345 \
    --scheduler_type "pcm_linear_quadratic" 




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

