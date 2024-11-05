num_gpus=1

torchrun --nproc_per_node=$num_gpus fastvideo/sample/generate_synthetic.py \
    --model_path data/mochi \
    --num_frames 163 \
    --height 480 \
    --width 848 \
    --num_inference_steps 64 \
    --guidance_scale 4.5 \
<<<<<<<< HEAD:scripts/data_preprocess/generate_synthetic.sh
    --prompt_path "./prompt.txt" \
    --dataset_output_dir "./data/Mochi-Synthetic-Data_new"
========
    --prompt_path "data/prompt.txt" \
    --dataset_output_dir data/synthetic_debug2
>>>>>>>> 6c6d4b34e7a21b7b6996073ac3e8a6feaefee17a:scripts/generate_synthetic.sh

    