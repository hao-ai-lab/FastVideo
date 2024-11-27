


num_gpus=4
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/mochi \
    --prompt_path data/prompt.txt \
    --num_frames  163 \
    --height 480 \
    --width 848 \
    --num_inference_steps 64 \
    --guidance_scale 4.5 \
    --output_path outputs_video/distill_baseline_debug_fp32cast/ \
    --shift 4 \
    --seed 12345 \
    --scheduler_type "euler" 






