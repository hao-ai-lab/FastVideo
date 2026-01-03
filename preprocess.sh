export MASTER_ADDR=localhost
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1

python fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path /workspace/Matrix-Game-2.0-Base-Diffusers \
    --data_merge_path /workspace/footsies-dataset/merge.txt \
    --output_dir /workspace/footsies-dataset/preprocessed/ \
    --preprocess_task matrixgame \
    --num_frames 77 \
    --max_height 480 \
    --max_width 832 \
    --preprocess_video_batch_size 2 \
    --dataloader_num_workers 4 \
    --samples_per_file 64
