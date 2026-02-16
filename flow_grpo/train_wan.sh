export PYTHONPATH="${PYTHONPATH}:/home/hao_lab/miniconda3/envs/flow_grpo/bin/python"
export WANDB_API_KEY="84fb0deb6b40f77a0f1ceda0029efbe67164395f"
export CUDA_VISIBLE_DEVICES=4,5,6,7
# Set to a directory where align_logs/flow_logs (and fv_logs when running FastVideo) will be written
export ALIGN_LOGS_ROOT="/mnt/fast-disks/hao_lab/shijie/FastVideo"

accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --main_process_port 29503 \
    scripts/train_wan2_1.py \
    --config config/grpo.py:general_ocr_wan2_1

# accelerate launch \The
#     --config_file scripts/accelerate_configs/multi_gpu.yaml \
#     --num_processes 1 \
#     --main_process_port 29503 \
#     scripts/train_wan2_1.py \
#     --config config/grpo.py:test_wan2_1_videoalign
