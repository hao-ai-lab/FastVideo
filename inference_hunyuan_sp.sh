num_gpus=1

torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_hunyuan.py \
    --height 512 \
    --width 512 \
    --num_frames 29 \
    --num_inference_steps 50 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow-reverse \
    --prompts "A cat walks on the grass, realistic style." \
    --seed 42 \
    --output_path outputs_video/hunyuan/
