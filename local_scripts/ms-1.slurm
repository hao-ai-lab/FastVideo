#!/bin/bash
#SBATCH --job-name=full-1
#SBATCH --partition=mbzuai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=960G
#SBATCH --output=slurm_log_mask_search_200/full-1.out
#SBATCH --error=slurm_log_mask_search_200/full-1.err
#SBATCH --exclusive
#SBATCH --time=72:00:00

conda init
source ~/conda/miniconda/bin/activate
PYTHON_VIRTUAL_ENVIRONMENT=yq-stepvideo-t250 
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

echo " "
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date



# Create an array of CUDA device IDs
gpu_ids=(0 1 2 3 4 5 6 7)

cd ~/yongqi/FastVideo-StepV
CUDA_VISIBLE_DEVICES=7 python fastvideo/sample/call_remote_server_stepvideo.py  --model_dir data/stepvideo-t2v/ &

# inference
parallel=4
url='127.0.0.1'
model_dir=data/stepvideo-t2v
skip_time_steps=50
mask_strategy_selected=0,1,2,3
mask_search_files_path=mask_research_json_7m
save_path=outputs_STA/MovieGen_200_400_${mask_strategy_selected}_${skip_time_steps}
torchrun --nproc_per_node $parallel fastvideo/sample/sample_t2v_stepvideo_STA.py \
    --model_dir $model_dir \
    --vae_url $url \
    --caption_url $url  \
    --prompt assets/prompts_200_300.txt \
    --infer_steps 50  \
    --width 768 \
    --height 768 \
    --num_frames 204 \
    --cfg_scale 9.0 \
    --save_path $save_path \
    --time_shift 13.0 \
    --skip_time_steps $skip_time_steps \
    --mask_strategy_selected $mask_strategy_selected \
    --mask_search_files_path $mask_search_files_path

# Wait for all background jobs to complete
wait
echo "${save_path}"
echo "All jobs completed!"

# cd ~/yongqi/common_metrics_on_video_quality
# conda activate video_metric
# python demo.py ../original_hunyuan_moviegen200_name/ ../FastVideo-Internal-inference/${output_path}