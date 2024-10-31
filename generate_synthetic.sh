num_gpus=8

torchrun --nproc_per_node=$num_gpus fastvideo/sample/generate_synthetic.py \
    --model_path data/mochi \
    --num_frames 163 \
    --height 480 \
    --width 848 \
    --num_inference_steps 64 \
    --guidance_scale 4.5 \
    --prompt_path "/path/to/prompt.txt" \
    --dataset_output_dir test

    