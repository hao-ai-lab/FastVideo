parallel=1
url='127.0.0.1'
model_dir=/workspace/FastVideo-StepV/data/stepvideo-t2v
nsys profile --output=my_profile torchrun --nproc_per_node $parallel quant_model.py --model_dir $model_dir --vae_url $url --caption_url $url --prompt "A police helicopter hovers above a high-speed chase, guiding officers on the ground to apprehend a suspect." --num_frames 136  --save_path "/workspace/FastVideo-StepV/fastvideo/quant/stepvideo" --height 544 --width 992 --infer_steps 1