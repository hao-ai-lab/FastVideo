CUDA_VISIBLE_DEVICES=7 python api/call_remote_server.py --model_dir data/stepvideo-t2v/ &

# inference
parallel=4
url='127.0.0.1'
model_dir=data/stepvideo-t2v
skip_time_steps=15
mask_strategy_selected=0,1,2,3
mask_search_files_path=mask_research_json_7m
torchrun --nproc_per_node $parallel fastvideo/sample/sample_t2v_stepvideo_STA.py \
    --model_dir $model_dir \
    --vae_url $url \
    --caption_url $url  \
    --prompt assets/MovieGen_s8/prompts_s8.txt \
    --infer_steps 50  \
    --width 768 \
    --height 768 \
    --num_frames 204 \
    --cfg_scale 9.0 \
    --save_path outputs_STA/MovieGen_s8_${mask_strategy_selected}_${skip_time_steps}/ \
    --time_shift 13.0 \
    --skip_time_steps $skip_time_steps \
    --mask_strategy_selected $mask_strategy_selected \
    --mask_search_files_path $mask_search_files_path

parallel=4
url='127.0.0.1'
model_dir=data/stepvideo-t2v
skip_time_steps=12
mask_strategy_selected=4,5,6
mask_search_files_path=mask_research_json_7m
torchrun --nproc_per_node $parallel fastvideo/sample/sample_t2v_stepvideo_STA.py \
    --model_dir $model_dir \
    --vae_url $url \
    --caption_url $url  \
    --prompt assets/MovieGen_s8/prompts_s8.txt \
    --infer_steps 50  \
    --width 768 \
    --height 768 \
    --num_frames 204 \
    --cfg_scale 9.0 \
    --save_path outputs_STA/MovieGen_s8_${mask_strategy_selected}_${skip_time_steps}/ \
    --time_shift 13.0 \
    --skip_time_steps $skip_time_steps \
    --mask_strategy_selected $mask_strategy_selected \
    --mask_search_files_path $mask_search_files_path

CUDA_VISIBLE_DEVICES=7 python api/call_remote_server.py --model_dir data/stepvideo-t2v/ &

# inference
parallel=6
url='127.0.0.1'
model_dir=data/stepvideo-t2v
skip_time_steps=15
mask_strategy_selected=4,5,6
mask_search_files_path=mask_research_json_7m
torchrun --nproc_per_node $parallel run_parallel_STA.py \
    --model_dir $model_dir \
    --vae_url $url \
    --caption_url $url  \
    --ulysses_degree $parallel \
    --prompt assets/MovieGen_s8/prompts_s8.txt \
    --infer_steps 50  \
    --width 768 \
    --height 768 \
    --num_frames 204 \
    --cfg_scale 9.0 \
    --save_path outputs_STA/results_sp_save_ms_s8/ \
    --time_shift 13.0 \
    --skip_time_steps $skip_time_steps \
    --mask_strategy_selected $mask_strategy_selected \
    --mask_search_files_path $mask_search_files_path

# --mask_strategy_file_path $mask_strategy_file_path \

parallel=6
url='127.0.0.1'
model_dir=data/stepvideo-t2v-turbo
torchrun --nproc_per_node $parallel run_parallel_STA.py \
    --model_dir $model_dir \
    --vae_url $url \
    --caption_url $url  \
    --ulysses_degree $parallel \
    --prompt assets/MovieGen_s8/prompts_s8.txt \
    --infer_steps 15  \
    --width 992 \
    --height 544 \
    --num_frames 136 \
    --cfg_scale 5.0 \
    --save_path results_distill_136/ \
    --time_shift 17.0 \

parallel=1
url='127.0.0.1'
model_dir=data/stepvideo-t2v

torchrun --nproc_per_node $parallel run_parallel_STA.py \
    --model_dir $model_dir \
    --vae_url $url \
    --caption_url $url  \
    --ulysses_degree $parallel \
    --prompt assets/MovieGen_s8/prompts_s8.txt \
    --infer_steps 50  \
    --width 256 \
    --height 256 \
    --num_frames 68 \
    --cfg_scale 9.0 \
    --save_path results_test/ \
    --time_shift 13.0 \

parallel=6
url='127.0.0.1'
model_dir=data/stepvideo-t2v
torchrun --nproc_per_node $parallel run_parallel_STA.py \
    --model_dir $model_dir \
    --vae_url $url \
    --caption_url $url  \
    --ulysses_degree $parallel \
    --prompt assets/MovieGen_s8/prompts_s8.txt \
    --infer_steps 50  \
    --width 768 \
    --height 768 \
    --num_frames 204 \
    --cfg_scale 9.0 \
    --save_path results_sp_save_ms_s8/ \
    --time_shift 13.0

parallel=6
url='127.0.0.1'
model_dir=data/stepvideo-t2v
torchrun --nproc_per_node $parallel fastvideo/sample/sample_t2v_stepvideo_STA.py \
    --model_dir $model_dir \
    --vae_url $url \
    --caption_url $url  \
    --prompt assets/MovieGen_f8/prompts_f8.txt \
    --infer_steps 2  \
    --width 768 \
    --height 768 \
    --num_frames 204 \
    --cfg_scale 9.0 \
    --save_path results_sp_save_ms_f8/ \
    --time_shift 13.0

parallel=1
url='127.0.0.1'
model_dir=data/stepvideo-t2v

torchrun --nproc_per_node $parallel run_parallel_STA.py \
    --model_dir $model_dir \
    --vae_url $url \
    --caption_url $url  \
    --ulysses_degree $parallel \
    --prompt assets/MovieGen_s8/prompts_s8.txt \
    --infer_steps 50  \
    --width 768 \
    --height 768 \
    --num_frames 136 \
    --cfg_scale 9.0 \
    --save_path results_768_f204/ \
    --time_shift 13.0