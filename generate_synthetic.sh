python fastvideo/sample/generate_synthetic.py \
    --model_path data/mochi \
    --num_frames 163 \
    --height 480 \
    --width 848 \
    --num_inference_steps 64 \
    --guidance_scale 4.5 \
    --prompt_path data/dummyVid/videos2caption_5video.json \
    --dataset_output_dir data/Mochi-Synthetic-Data

    