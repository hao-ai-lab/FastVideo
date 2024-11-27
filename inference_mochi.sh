
checkpoint_iter_list=(
    "250"
    "500"
    "750"
    "1000"
)
num_gpus=4
for checkpoint_iter in "${checkpoint_iter_list[@]}"; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
        fastvideo/sample/sample_t2v_mochi.py \
        --model_path data/mochi \
        --prompt_path data/prompt.txt \
        --transformer_path data/outputs/video_distill_synthetic/checkpoint-${checkpoint_iter}\
        --num_frames  163 \
        --height 480 \
        --width 848 \
        --num_inference_steps 8 \
        --guidance_scale 4.5 \
        --output_path outputs_video/distill_synthetic_${checkpoint_iter} \
        --shift 4 \
        --seed 12345 \
        --scheduler_type "pcm_linear_quadratic" 
done
