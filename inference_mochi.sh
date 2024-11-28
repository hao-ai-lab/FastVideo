

num_gpus=4

torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/mochi \
    --prompt_path data/prompt.txt \
    --transformer_path data/outputs/video_distill_synthetic/checkpoint-250\
    --num_frames  163 \
    --height 480 \
    --width 848 \
    --num_inference_steps 8 \
    --guidance_scale 6 \
    --output_path outputs_video/distill_lq_163_250_precision_correct_guidance_6 \
    --shift 8 \
    --seed 12345 \
    --scheduler_type "pcm_linear_quadratic" 

