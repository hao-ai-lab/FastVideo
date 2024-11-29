num_gpus=1

torchrun --nproc_per_node=$num_gpus fastvideo/sample/generate_synthetic.py \
    --model_path data/mochi \
    --num_frames 1 \
    --height 480 \
    --width 848 \
    --num_inference_steps 4 \
    --guidance_scale 4.5 \
    --prompt_path "data/prompt.txt" \
    --dataset_output_dir data/synthetic_debug2

    